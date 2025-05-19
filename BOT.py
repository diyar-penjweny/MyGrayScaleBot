import cv2
import numpy as np
import telebot
from telebot import types
import tempfile
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import torch
from sklearn.cluster import KMeans
from collections import Counter
from typing import Optional, Tuple, Dict, Any

# Initialize bot with your token
API_TOKEN = 'Your telegram bot'
bot = telebot.TeleBot(API_TOKEN)

# Check for GPU availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# User sessions to store original images
user_sessions = {}


class UserSession:
    def __init__(self):
        self.original_image: Optional[np.ndarray] = None
        self.current_mode: str = 'grayscale'
        self.processed_image: Optional[np.ndarray] = None
        self.history: list = []  # To store processing history for undo
        self.settings: Dict[str, Any] = {
            'bw_threshold': 128,
            'edge_low': 100,
            'edge_high': 200,
            'enhance_factor': 1.0,
            'style_transfer': 'none'
        }


class ImageProcessor:
    @staticmethod
    def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale with luminosity method"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def convert_to_bw(image: np.ndarray, threshold: int = 128) -> np.ndarray:
        """Convert image to black and white with adaptive thresholding"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Use adaptive thresholding for better results
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
        return bw

    @staticmethod
    def apply_edge_detection(image: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
        """Apply Canny edge detection with automatic threshold calculation"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Auto calculate thresholds using median
        v = np.median(gray)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(gray, lower, upper, L2gradient=True)
        return edges

    @staticmethod
    def apply_cartoon_effect(image: np.ndarray) -> np.ndarray:
        """Apply cartoon effect to the image"""
        # Convert to grayscale and apply median blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)

        # Detect edges with adaptive threshold
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)

        # Apply bilateral filter for color smoothing
        color = cv2.bilateralFilter(image, 9, 300, 300)

        # Combine edges with color image
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon

    @staticmethod
    def apply_sketch_effect(image: np.ndarray) -> np.ndarray:
        """Convert image to pencil sketch effect"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inv_gray = 255 - gray
        blurred = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        inv_blurred = 255 - blurred
        sketch = cv2.divide(gray, inv_blurred, scale=256.0)
        return sketch

    @staticmethod
    def enhance_image(image: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """Enhance image using PIL for better quality adjustments"""
        # Convert to PIL format
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Apply enhancements
        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced = enhancer.enhance(factor)

        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(factor)

        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)

    @staticmethod
    def apply_style_transfer(image: np.ndarray, style: str = 'watercolor') -> np.ndarray:
        """Apply simple style transfer effect (simulated)"""
        if style == 'watercolor':
            result = cv2.stylization(image, sigma_s=60, sigma_r=0.6)
        elif style == 'oil_painting':
            result = cv2.xphoto.oilPainting(image, size=7, dynRatio=1)
        else:
            result = image
        return result

    @staticmethod
    def extract_dominant_colors(image: np.ndarray, num_colors: int = 5) -> Tuple[list, list]:
        """Extract dominant colors from image using K-means clustering"""
        # Resize image for faster processing
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
        pixels = img.reshape(-1, 3)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_colors, random_state=42)
        kmeans.fit(pixels)

        # Get colors and their proportions
        counts = Counter(kmeans.labels_)
        total = sum(counts.values())
        colors = [kmeans.cluster_centers_[i].astype(int) for i in range(num_colors)]
        proportions = [counts[i] / total for i in range(num_colors)]

        return colors, proportions

    @staticmethod
    def process_image(image: np.ndarray, mode: str = 'grayscale', settings: Optional[Dict] = None) -> np.ndarray:
        """Main image processing function with multiple modes"""
        if settings is None:
            settings = {}

        if mode == 'grayscale':
            return ImageProcessor.convert_to_grayscale(image)
        elif mode == 'bw':
            threshold = settings.get('bw_threshold', 128)
            return ImageProcessor.convert_to_bw(image, threshold)
        elif mode == 'both':
            gray = ImageProcessor.convert_to_grayscale(image)
            bw = ImageProcessor.convert_to_bw(image, settings.get('bw_threshold', 128))
            return np.hstack((gray, bw))
        elif mode == 'edge':
            low = settings.get('edge_low', 100)
            high = settings.get('edge_high', 200)
            return ImageProcessor.apply_edge_detection(image, low, high)
        elif mode == 'cartoon':
            return ImageProcessor.apply_cartoon_effect(image)
        elif mode == 'sketch':
            return ImageProcessor.apply_sketch_effect(image)
        elif mode == 'enhance':
            factor = settings.get('enhance_factor', 1.5)
            return ImageProcessor.enhance_image(image, factor)
        elif mode.startswith('style_'):
            style = mode.split('_')[1]
            return ImageProcessor.apply_style_transfer(image, style)
        return image

    @staticmethod
    def generate_colorful_histogram(image: np.ndarray, mode: str) -> str:
        """Generate enhanced histogram visualization with dominant colors"""
        plt.figure(figsize=(12, 6))

        if mode in ['grayscale', 'bw', 'both']:
            # Grayscale histogram with modern design
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            hist, bins = np.histogram(image.flatten(), 256, [0, 256])
            plt.fill_between(range(256), hist, color='#4B8BBE', alpha=0.6)
            plt.plot(hist, color='#306998', linewidth=2)

            # Add statistics
            mean_val = np.mean(image)
            median_val = np.median(image)
            plt.axvline(mean_val, color='#FFD43B', linestyle='--', linewidth=2,
                        label=f'Mean: {mean_val:.1f}')
            plt.axvline(median_val, color='#FF872C', linestyle=':', linewidth=2,
                        label=f'Median: {median_val:.1f}')

            plt.gca().set_facecolor('#F5F5F5')
            plt.title('Grayscale Histogram with Statistics', color='#333333', fontsize=14)
            plt.legend()
        else:
            # Color histogram with dominant colors
            colors, proportions = ImageProcessor.extract_dominant_colors(image)

            # Plot RGB channels
            channel_colors = ('#FF0000', '#00FF00', '#0000FF')
            channel_names = ('Red', 'Green', 'Blue')
            for i, (color, name) in enumerate(zip(channel_colors, channel_names)):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                plt.plot(hist, color=color, label=name, linewidth=2, alpha=0.7)

            # Add dominant color information
            ax2 = plt.gca().twinx()
            for i, (color, prop) in enumerate(zip(colors, proportions)):
                rgb_color = [c / 255 for c in color]
                ax2.bar(i, prop * 100, color=rgb_color, alpha=0.6,
                        label=f'Color {i + 1}: {prop * 100:.1f}%')

            plt.gca().set_facecolor('#FAFAFA')
            plt.title('RGB Histogram with Dominant Colors', color='#333333', fontsize=14)
            plt.legend(loc='upper right')
            ax2.set_ylabel('Dominant Color Percentage (%)')
            ax2.set_ylim(0, 100)
            ax2.legend(loc='upper left')

        plt.xlabel('Pixel Intensity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.4)

        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(temp_file.name, bbox_inches='tight', dpi=150,
                    facecolor=plt.gca().get_facecolor())
        plt.close()
        return temp_file.name


def get_mode_keyboard(show_advanced: bool = False) -> types.InlineKeyboardMarkup:
    """Create interactive keyboard with processing options"""
    keyboard = types.InlineKeyboardMarkup(row_width=2)

    # Basic modes
    basic_buttons = [
        types.InlineKeyboardButton("Grayscale", callback_data="grayscale"),
        types.InlineKeyboardButton("Black & White", callback_data="bw"),
        types.InlineKeyboardButton("Both", callback_data="both"),
        types.InlineKeyboardButton("Edge Detection", callback_data="edge")
    ]
    keyboard.add(*basic_buttons)

    # Advanced modes
    if show_advanced:
        advanced_buttons = [
            types.InlineKeyboardButton("🎨 Cartoon", callback_data="cartoon"),
            types.InlineKeyboardButton("✏️ Sketch", callback_data="sketch"),
            types.InlineKeyboardButton("✨ Enhance", callback_data="enhance"),
            types.InlineKeyboardButton("🌊 Watercolor", callback_data="style_watercolor"),
            types.InlineKeyboardButton("🖼️ Oil Painting", callback_data="style_oil_painting")
        ]
        keyboard.add(*advanced_buttons)

    # Always show histogram and settings
    keyboard.row(
        types.InlineKeyboardButton("🌈 Show Histogram", callback_data="histogram"),
        types.InlineKeyboardButton("⚙️ Settings", callback_data="settings")
    )

    if show_advanced:
        keyboard.row(types.InlineKeyboardButton("⬅️ Basic Modes", callback_data="basic_modes"))
    else:
        keyboard.row(types.InlineKeyboardButton("➡️ Advanced Modes", callback_data="advanced_modes"))

    return keyboard


def get_settings_keyboard(chat_id: int) -> types.InlineKeyboardMarkup:
    """Create keyboard for adjusting settings"""
    if chat_id not in user_sessions:
        return get_mode_keyboard()

    settings = user_sessions[chat_id].settings
    keyboard = types.InlineKeyboardMarkup()

    # Threshold settings
    keyboard.row(
        types.InlineKeyboardButton(f"BW Threshold: {settings['bw_threshold']}",
                                   callback_data="set_bw_threshold")
    )

    # Edge detection settings
    keyboard.row(
        types.InlineKeyboardButton(f"Edge Low: {settings['edge_low']}",
                                   callback_data="set_edge_low"),
        types.InlineKeyboardButton(f"Edge High: {settings['edge_high']}",
                                   callback_data="set_edge_high")
    )

    # Enhancement factor
    keyboard.row(
        types.InlineKeyboardButton(f"Enhance Factor: {settings['enhance_factor']:.1f}",
                                   callback_data="set_enhance_factor")
    )

    # Back button
    keyboard.row(
        types.InlineKeyboardButton("⬅️ Back to Modes", callback_data="back_to_modes")
    )

    return keyboard


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message: types.Message) -> None:
    """Send welcome message with bot instructions"""
    welcome_text = """
    🎨 *Advanced Image Processing Bot* 🎨

    *Features:*
    - Basic conversions: Grayscale, B&W, Edge Detection
    - Advanced effects: Cartoon, Sketch, Watercolor, Oil Painting
    - Image enhancement and quality improvement
    - Color analysis with histograms and dominant colors
    - Customizable processing parameters

    *How to use:*
    1. Send me any image
    2. Choose processing mode
    3. Adjust settings if needed
    4. Download your processed image

    Try the *Advanced Modes* for creative effects!
    """
    bot.send_message(message.chat.id, welcome_text, parse_mode="Markdown")


@bot.message_handler(content_types=['photo'])
def handle_image(message: types.Message) -> None:
    """Handle incoming image messages"""
    try:
        chat_id = message.chat.id
        if chat_id not in user_sessions:
            user_sessions[chat_id] = UserSession()

        # Get the highest quality photo available
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Process the image
        image = cv2.imdecode(np.frombuffer(downloaded_file, np.uint8), cv2.IMREAD_COLOR)

        # Store in user session
        user_sessions[chat_id].original_image = image
        user_sessions[chat_id].current_mode = 'original'
        user_sessions[chat_id].history = [image.copy()]  # Initialize history

        # Send confirmation with processing options
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


@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call: types.CallbackQuery) -> None:
    """Handle all callback queries from inline keyboards"""
    try:
        chat_id = call.message.chat.id
        data = call.data

        if chat_id not in user_sessions or user_sessions[chat_id].original_image is None:
            bot.answer_callback_query(call.id, "Please send a new image first", show_alert=True)
            return

        session = user_sessions[chat_id]

        if data in ['grayscale', 'bw', 'both', 'edge', 'cartoon', 'sketch', 'enhance',
                    'style_watercolor', 'style_oil_painting']:
            # Handle processing modes
            session.current_mode = data
            processed = ImageProcessor.process_image(session.original_image, data, session.settings)
            session.processed_image = processed
            session.history.append(processed.copy())  # Add to history

            captions = {
                'grayscale': "🎨 Grayscale conversion",
                'bw': "⚫⚪ Black & White conversion (adaptive threshold)",
                'both': "⬜ Left: Grayscale | ⬛ Right: Black & White",
                'edge': "🔍 Edge Detection (Canny with auto thresholds)",
                'cartoon': "🎨 Cartoon Effect",
                'sketch': "✏️ Pencil Sketch Effect",
                'enhance': "✨ Enhanced Image (contrast + sharpness)",
                'style_watercolor': "🌊 Watercolor Style Effect",
                'style_oil_painting': "🖼️ Oil Painting Effect"
            }
            caption = captions.get(data, "Processed image")

            # Send processed image
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                cv2.imwrite(temp_file.name, processed)
                temp_file_path = temp_file.name

            with open(temp_file_path, 'rb') as photo:
                bot.edit_message_media(
                    chat_id=chat_id,
                    message_id=call.message.message_id,
                    media=types.InputMediaPhoto(photo, caption=f"{caption}\nSelect another mode:")
                )

            # Update keyboard based on current mode
            show_advanced = data in ['cartoon', 'sketch', 'enhance', 'style_watercolor', 'style_oil_painting']
            bot.edit_message_reply_markup(
                chat_id=chat_id,
                message_id=call.message.message_id,
                reply_markup=get_mode_keyboard(show_advanced)
            )
            os.unlink(temp_file_path)

        elif data == 'histogram':
            # Generate and send histogram
            if session.current_mode in ['grayscale', 'bw', 'cartoon', 'sketch']:
                hist_image = session.processed_image if session.processed_image is not None else session.original_image
                hist_path = ImageProcessor.generate_colorful_histogram(hist_image, session.current_mode)
                caption = f"🌈 Colorful Histogram ({session.current_mode})"
            elif session.current_mode == 'both':
                gray_part = session.processed_image[:, :session.processed_image.shape[1] // 2]
                hist_path = ImageProcessor.generate_colorful_histogram(gray_part, 'grayscale')
                caption = "🌈 Histogram (Grayscale part)"
            else:
                hist_path = ImageProcessor.generate_colorful_histogram(session.original_image, 'color')
                caption = "🌈 RGB Color Histogram with Dominant Colors"

            with open(hist_path, 'rb') as hist_file:
                bot.send_photo(
                    chat_id,
                    hist_file,
                    caption=caption,
                    reply_to_message_id=call.message.message_id
                )
            os.unlink(hist_path)

        elif data in ['advanced_modes', 'basic_modes']:
            # Switch between basic and advanced modes
            show_advanced = data == 'advanced_modes'
            bot.edit_message_reply_markup(
                chat_id=chat_id,
                message_id=call.message.message_id,
                reply_markup=get_mode_keyboard(show_advanced)
            )

        elif data == 'settings':
            # Show settings keyboard
            bot.edit_message_reply_markup(
                chat_id=chat_id,
                message_id=call.message.message_id,
                reply_markup=get_settings_keyboard(chat_id)
            )

        elif data == 'back_to_modes':
            # Return to mode selection
            show_advanced = session.current_mode in ['cartoon', 'sketch', 'enhance',
                                                     'style_watercolor', 'style_oil_painting']
            bot.edit_message_reply_markup(
                chat_id=chat_id,
                message_id=call.message.message_id,
                reply_markup=get_mode_keyboard(show_advanced)
            )

        elif data.startswith('set_'):
            # Handle setting adjustments
            setting = data[4:]
            current_value = session.settings.get(setting, 0)

            if setting == 'bw_threshold':
                new_value = max(0, min(255, current_value + 10 if call.message.text == '+' else current_value - 10))
            elif setting in ['edge_low', 'edge_high']:
                new_value = max(1, min(255, current_value + 5 if call.message.text == '+' else current_value - 5))
            elif setting == 'enhance_factor':
                new_value = max(0.1, min(3.0,
                                         round(current_value + 0.1 if call.message.text == '+' else current_value - 0.1,
                                               1)))

            session.settings[setting] = new_value
            bot.edit_message_reply_markup(
                chat_id=chat_id,
                message_id=call.message.message_id,
                reply_markup=get_settings_keyboard(chat_id)
            )

        bot.answer_callback_query(call.id)

    except Exception as e:
        bot.answer_callback_query(call.id, f"Error: {str(e)}", show_alert=True)


if __name__ == '__main__':
    print("Bot is running with advanced features...")
    bot.infinity_polling()
