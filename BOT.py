import cv2
import numpy as np
import telebot
from telebot import types
import tempfile
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Initialize bot with your token
API_TOKEN = 'YOUR TELEGRAM BOT'
bot = telebot.TeleBot(API_TOKEN)

# User sessions to store original images
user_sessions = {}


class UserSession:
    def __init__(self):
        self.original_image = None
        self.current_mode = 'grayscale'
        self.processed_image = None  # Store the last processed image


# Default threshold for black & white conversion
DEFAULT_THRESHOLD = 128


class ImageProcessor:
    @staticmethod
    def convert_to_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def convert_to_bw(image, threshold=DEFAULT_THRESHOLD):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return bw

    @staticmethod
    def process_image(image, mode='grayscale'):
        if mode == 'grayscale':
            return ImageProcessor.convert_to_grayscale(image)
        elif mode == 'bw':
            return ImageProcessor.convert_to_bw(image)
        elif mode == 'both':
            gray = ImageProcessor.convert_to_grayscale(image)
            bw = ImageProcessor.convert_to_bw(image)
            return np.hstack((gray, bw))  # Combine side by side
        return image

    @staticmethod
    def generate_colorful_histogram(image, mode):
        plt.figure(figsize=(10, 5))

        # Create custom colorful style for grayscale/BW images
        if mode in ['grayscale', 'bw', 'both']:
            # Create a gradient colormap from dark to light gray
            colors = [(0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1)]  # Black to White
            cmap_name = 'grayscale_cmap'
            grayscale_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

            if len(image.shape) == 3:  # Convert to grayscale if needed
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Plot with gradient fill
            hist, bins = np.histogram(image.flatten(), 256, [0, 256])
            plt.fill_between(range(256), hist, color='skyblue', alpha=0.4)
            plt.plot(hist, color='navy', linewidth=2)
            plt.gca().set_facecolor('lightcyan')
            plt.title('Grayscale Histogram', color='darkblue', fontsize=14)
        else:
            # Colorful histogram for color images
            colors = ('blue', 'green', 'red')
            channel_names = ('Blue', 'Green', 'Red')
            for i, (color, name) in enumerate(zip(colors, channel_names)):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                plt.plot(hist, color=color, label=name, linewidth=2)
                plt.fill_between(range(256), hist.flatten(), color=color, alpha=0.3)
            plt.legend()
            plt.gca().set_facecolor('linen')
            plt.title('RGB Color Histogram', color='darkred', fontsize=14)

        # Common styling
        plt.xlabel('Pixel Intensity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(temp_file.name, bbox_inches='tight', dpi=120, facecolor=plt.gca().get_facecolor())
        plt.close()

        return temp_file.name


# Generate keyboard for mode selection
def get_mode_keyboard(show_histogram=True):
    keyboard = types.InlineKeyboardMarkup()
    keyboard.row(
        types.InlineKeyboardButton("Grayscale", callback_data="grayscale"),
        types.InlineKeyboardButton("Black & White", callback_data="bw"),
        types.InlineKeyboardButton("Both", callback_data="both")
    )
    if show_histogram:
        keyboard.row(
            types.InlineKeyboardButton("🌈 Show Colorful Histogram", callback_data="histogram")
        )
    return keyboard


# Start command handler
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = """
    🎨 *Advanced Image Converter Bot* 🎨

    Send me an image and I'll convert it to:
    - Grayscale
    - Black & White
    - Or both side by side

    *New Feature:* 🌈 Colorful histograms that match your image type!

    Just send me any image to begin!
    """
    bot.send_message(message.chat.id, welcome_text, parse_mode="Markdown")


# Handle all image messages
@bot.message_handler(content_types=['photo'])
def handle_image(message):
    try:
        chat_id = message.chat.id
        if chat_id not in user_sessions:
            user_sessions[chat_id] = UserSession()

        # Get the highest resolution photo
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Convert to OpenCV format and store original
        image = cv2.imdecode(np.frombuffer(downloaded_file, np.uint8), cv2.IMREAD_COLOR)
        user_sessions[chat_id].original_image = image
        user_sessions[chat_id].current_mode = 'original'

        # Send original image first
        _, img_encoded = cv2.imencode('.jpg', image)
        img_bytes = img_encoded.tobytes()

        bot.send_photo(
            chat_id,
            img_bytes,
            caption="Original image. Select processing mode:",
            reply_markup=get_mode_keyboard()
        )

    except Exception as e:
        bot.reply_to(message, f"❌ Error processing image: {str(e)}")


# Callback query handler for mode selection
@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    try:
        chat_id = call.message.chat.id
        data = call.data

        if chat_id not in user_sessions or user_sessions[chat_id].original_image is None:
            bot.answer_callback_query(call.id, "Please send a new image first", show_alert=True)
            return

        session = user_sessions[chat_id]

        if data in ['grayscale', 'bw', 'both']:
            # Mode selection
            session.current_mode = data
            processed = ImageProcessor.process_image(session.original_image, data)
            session.processed_image = processed

            # Create caption based on mode
            captions = {
                'grayscale': "🎨 Grayscale conversion",
                'bw': "⚫⚪ Black & White conversion",
                'both': "⬜ Left: Grayscale | ⬛ Right: Black & White"
            }
            caption = captions.get(data, "Processed image")

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                cv2.imwrite(temp_file.name, processed)
                temp_file_path = temp_file.name

            # Edit the original message with new image
            with open(temp_file_path, 'rb') as photo:
                bot.edit_message_media(
                    chat_id=chat_id,
                    message_id=call.message.message_id,
                    media=types.InputMediaPhoto(photo, caption=f"{caption}\nSelect another mode:")
                )

            # Update the reply markup (buttons)
            bot.edit_message_reply_markup(
                chat_id=chat_id,
                message_id=call.message.message_id,
                reply_markup=get_mode_keyboard()
            )

            # Clean up temporary file
            os.unlink(temp_file_path)

        elif data == 'histogram':
            # Generate appropriate histogram
            if session.current_mode in ['grayscale', 'bw']:
                hist_image = session.processed_image
                hist_path = ImageProcessor.generate_colorful_histogram(hist_image, session.current_mode)
                caption = f"🌈 Colorful Histogram ({session.current_mode})"
            elif session.current_mode == 'both':
                # Use just the grayscale part for histogram
                gray_part = session.processed_image[:, :session.processed_image.shape[1] // 2]
                hist_path = ImageProcessor.generate_colorful_histogram(gray_part, 'grayscale')
                caption = "🌈 Histogram (Grayscale part)"
            else:  # Original color image
                hist_path = ImageProcessor.generate_colorful_histogram(session.original_image, 'color')
                caption = "🌈 RGB Color Histogram"

            # Send histogram as a separate message
            with open(hist_path, 'rb') as hist_file:
                bot.send_photo(
                    chat_id,
                    hist_file,
                    caption=caption,
                    reply_to_message_id=call.message.message_id
                )

            # Clean up histogram file
            os.unlink(hist_path)

        # Acknowledge the callback
        bot.answer_callback_query(call.id)

    except Exception as e:
        bot.answer_callback_query(call.id, f"Error: {str(e)}", show_alert=True)


# Start the bot
if __name__ == '__main__':
    print("Bot is running...")
    bot.infinity_polling()
