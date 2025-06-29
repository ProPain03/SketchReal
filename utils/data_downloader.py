

import os
import sys
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from tqdm import tqdm
import pickle
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from config.config import Config

class EnhancedDataDownloader:
    def __init__(self):
        self.config = Config()
        self.download_log_file = os.path.join(self.config.RAW_DATA_DIR, 'download_log.pkl')
    
    def _load_download_log(self):
        """Load previous download information"""
        if os.path.exists(self.download_log_file):
            with open(self.download_log_file, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def _save_download_log(self, log_data):
        """Save download information"""
        os.makedirs(self.config.RAW_DATA_DIR, exist_ok=True)
        with open(self.download_log_file, 'wb') as f:
            pickle.dump(log_data, f)

    def download_quickdraw_data(self, categories=None, max_samples=5000):
        """Download Quick Draw dataset for specified categories"""
        if categories is None:
            categories = self.config.QUICKDRAW_CATEGORIES
        
        quickdraw_dir = os.path.join(self.config.RAW_DATA_DIR, 'quickdraw')
        os.makedirs(quickdraw_dir, exist_ok=True)
        
        base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
        
        for category in tqdm(categories, desc="Downloading Quick Draw data"):
            filename = f"{category}.npy"
            file_path = os.path.join(quickdraw_dir, filename)
            
            if not os.path.exists(file_path):
                url = f"{base_url}{filename}"
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    print(f"Downloaded {category}.npy")
                except Exception as e:
                    print(f"Error downloading {category}: {e}")

    def create_enhanced_synthetic_ui_data(self, force_create=False):
        """Create enhanced synthetic UI data for all UI labels"""
        download_log = self._load_download_log()
        synthetic_dir = os.path.join(self.config.RAW_DATA_DIR, 'synthetic_ui_enhanced')
        cache_key = f"synthetic_ui_enhanced_{len(self.config.UI_LABELS)}"
        
        if not force_create and cache_key in download_log:
            if self._check_synthetic_data_exists(synthetic_dir):
                print(f"✅ Enhanced synthetic UI data already exists")
                return
        
        os.makedirs(synthetic_dir, exist_ok=True)
        
        # Create synthetic data for each UI label
        for label_id, label_name in self.config.UI_LABELS.items():
            if label_name == 'background':  # Skip background for now
                continue
                
            label_dir = os.path.join(synthetic_dir, label_name)
            os.makedirs(label_dir, exist_ok=True)
            
            print(f"Generating {self.config.SYNTHETIC_SAMPLES_PER_LABEL} synthetic images for {label_name}...")
            
            for i in range(self.config.SYNTHETIC_SAMPLES_PER_LABEL):
                img = self._create_synthetic_ui_image(label_name, i)
                img.save(os.path.join(label_dir, f'{label_name}_{i:04d}.png'))
        
        # Save creation log
        download_log[cache_key] = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ui_labels': list(self.config.UI_LABELS.values()),
            'samples_per_label': self.config.SYNTHETIC_SAMPLES_PER_LABEL,
            'total_images': len(self.config.UI_LABELS) * self.config.SYNTHETIC_SAMPLES_PER_LABEL
        }
        self._save_download_log(download_log)
        print(f"✅ Enhanced synthetic UI data created successfully")

    def _check_synthetic_data_exists(self, synthetic_dir):
        """Check if synthetic data already exists"""
        if not os.path.exists(synthetic_dir):
            return False
        
        for label_name in self.config.UI_LABELS.values():
            if label_name == 'background':
                continue
            label_dir = os.path.join(synthetic_dir, label_name)
            if not os.path.exists(label_dir):
                return False
            images = [f for f in os.listdir(label_dir) if f.endswith('.png')]
            if len(images) < self.config.SYNTHETIC_SAMPLES_PER_LABEL:
                return False
        return True

    def _create_synthetic_ui_image(self, label_name, seed):
        """Create a synthetic UI image for the given label"""
        np.random.seed(seed)
        random.seed(seed)
        
        # Base image
        bg_colors = ['white', 'lightgray', 'aliceblue', 'lavender']
        img = Image.new('RGB', (64, 64), random.choice(bg_colors))
        draw = ImageDraw.Draw(img)
        colors = ['black', 'blue', 'green', 'red', 'purple', 'orange', 'brown']
        
        if label_name == 'Icon':
            self._draw_icon(draw, colors)
        elif label_name == 'Text Button':
            self._draw_text_button(draw, colors)
        elif label_name == 'Input Field':
            self._draw_input_field(draw, colors)
        elif label_name == 'Checked View':
            self._draw_checked_view(draw, colors)
        elif label_name == 'Switch':
            self._draw_switch(draw, colors)
        elif label_name == 'Page Indicator':
            self._draw_page_indicator(draw, colors)
        elif label_name == 'Pop-Up Window':
            self._draw_popup_window(draw, colors)
        elif label_name == 'Background Image':
            self._draw_image(draw, colors)
        elif label_name == 'Text':
            self._draw_text(draw, colors)
        elif label_name == 'Image':
            self._draw_image_placeholder(draw, colors)
        elif label_name == 'Sliding Menu':
            self._draw_sliding_menu(draw, colors)
        elif label_name == 'Other':
            self._draw_abstract_shape(draw, colors)
            
        return img

    def _draw_icon(self, draw, colors):
        """Draw various icon types"""
        center_x, center_y = 32, 32
        size = random.randint(8, 16)
        color = random.choice(colors)
        
        icon_type = random.randint(0, 4)
        if icon_type == 0:  # Star
            points = []
            for i in range(10):
                angle = i * 36
                radius = size if i % 2 == 0 else size // 2
                x = center_x + radius * np.cos(np.radians(angle))
                y = center_y + radius * np.sin(np.radians(angle))
                points.append((x, y))
            draw.polygon(points, outline=color, width=2)
        elif icon_type == 1:  # Heart
            draw.arc([center_x-size, center_y-size//2, center_x, center_y+size//2], 0, 180, fill=color, width=3)
            draw.arc([center_x, center_y-size//2, center_x+size, center_y+size//2], 0, 180, fill=color, width=3)
            draw.polygon([(center_x-size, center_y), (center_x, center_y+size), (center_x+size, center_y)], outline=color, width=2)
        elif icon_type == 2:  # Gear
            draw.ellipse([center_x-size, center_y-size, center_x+size, center_y+size], outline=color, width=2)
            for i in range(8):
                angle = i * 45
                x1 = center_x + (size-3) * np.cos(np.radians(angle))
                y1 = center_y + (size-3) * np.sin(np.radians(angle))
                x2 = center_x + (size+3) * np.cos(np.radians(angle))
                y2 = center_y + (size+3) * np.sin(np.radians(angle))
                draw.line([x1, y1, x2, y2], fill=color, width=2)
        elif icon_type == 3:  # House
            draw.polygon([(center_x, center_y-size), (center_x-size, center_y), (center_x+size, center_y)], outline=color, width=2)
            draw.rectangle([center_x-size//2, center_y, center_x+size//2, center_y+size], outline=color, width=2)
            draw.rectangle([center_x-2, center_y+2, center_x+2, center_y+size-2], outline=color, width=1)
        else:  # Plus
            draw.line([center_x, center_y-size, center_x, center_y+size], fill=color, width=3)
            draw.line([center_x-size, center_y, center_x+size, center_y], fill=color, width=3)

    def _draw_text_button(self, draw, colors):
        """Draw text button"""
        width = random.randint(25, 45)
        height = random.randint(12, 20)
        x = random.randint(5, 64-width-5)
        y = random.randint(20, 44-height)
        
        button_color = random.choice(['lightblue', 'lightgreen', 'lightcoral', 'khaki'])
        text_color = random.choice(colors)
        
        draw.rectangle([x, y, x+width, y+height], fill=button_color, outline='gray', width=2)
        
        # Add button text
        button_texts = ["OK", "Cancel", "Submit", "Next", "Save", "Login"]
        text = random.choice(button_texts)
        text_x = x + width//2 - len(text)*2
        text_y = y + height//2 - 4
        draw.text((text_x, text_y), text, fill=text_color)

    def _draw_input_field(self, draw, colors):
        """Draw input field"""
        width = random.randint(35, 50)
        height = random.randint(12, 18)
        x = random.randint(5, 64-width-5)
        y = random.randint(20, 44-height)
        
        draw.rectangle([x, y, x+width, y+height], fill='white', outline='gray', width=2)
        
        # Add placeholder text or cursor
        if random.random() > 0.5:
            draw.text((x+3, y+3), "Enter text...", fill='lightgray')
        else:
            cursor_x = x + random.randint(3, width-5)
            draw.line([cursor_x, y+2, cursor_x, y+height-2], fill='black', width=1)

    def _draw_checked_view(self, draw, colors):
        """Draw checkbox or checkmark"""
        if random.random() > 0.5:  # Checkbox
            size = random.randint(12, 18)
            x = random.randint(10, 54-size)
            y = random.randint(20, 44-size)
            
            draw.rectangle([x, y, x+size, y+size], outline='black', width=2)
            
            # Add checkmark
            draw.line([x+3, y+size//2, x+size//2, y+size-4], fill='green', width=2)
            draw.line([x+size//2, y+size-4, x+size-3, y+3], fill='green', width=2)
        else:  # X mark
            center_x, center_y = 32, 32
            size = random.randint(8, 12)
            color = random.choice(['red', 'darkred'])
            
            draw.line([center_x-size, center_y-size, center_x+size, center_y+size], fill=color, width=3)
            draw.line([center_x-size, center_y+size, center_x+size, center_y-size], fill=color, width=3)

    def _draw_switch(self, draw, colors):
        """Draw toggle switch"""
        switch_width = random.randint(20, 30)
        switch_height = random.randint(10, 13)
        x = random.randint(10, 54-switch_width)
        y = random.randint(25, 39-switch_height)
        
        # Switch state
        is_on = random.random() > 0.5
        track_color = 'lightgreen' if is_on else 'lightgray'
        
        # Draw track
        draw.rectangle([x, y, x+switch_width, y+switch_height], fill=track_color, outline='gray', width=1)
        
        # Draw toggle
        toggle_size = switch_height - 4
        toggle_x = x + switch_width - toggle_size - 2 if is_on else x + 2
        toggle_y = y + 2
        
        draw.ellipse([toggle_x, toggle_y, toggle_x+toggle_size, toggle_y+toggle_size], 
                    fill='white', outline='darkgray', width=1)

    def _draw_page_indicator(self, draw, colors):
        """Draw page indicator dots"""
        num_dots = random.randint(3, 6)
        dot_size = 4
        total_width = num_dots * dot_size * 2
        start_x = (64 - total_width) // 2
        y = random.randint(45, 55)
        
        active_dot = random.randint(0, num_dots-1)
        
        for i in range(num_dots):
            x = start_x + i * dot_size * 2
            color = 'blue' if i == active_dot else 'lightgray'
            draw.ellipse([x, y, x+dot_size, y+dot_size], fill=color)

    def _draw_popup_window(self, draw, colors):
        """Draw popup window frame"""
        width = random.randint(35, 50)
        height = random.randint(25, 40)
        x = random.randint(7, 64-width-7)
        y = random.randint(7, 64-height-7)
        
        # Window frame
        draw.rectangle([x, y, x+width, y+height], fill='white', outline='black', width=2)
        
        # Title bar
        title_height = 8
        draw.rectangle([x, y, x+width, y+title_height], fill='lightblue', outline='black', width=1)
        
        # Close button
        close_x = x + width - 6
        draw.rectangle([close_x, y+1, close_x+4, y+6], fill='red', outline='darkred', width=1)
        draw.text((close_x+1, y+1), "X", fill='white')

    def _draw_image(self, draw, colors):
        """Draw background image elements"""
        # Simple landscape or abstract background
        if random.random() > 0.5:  # Landscape
            # Sky
            draw.rectangle([0, 0, 64, 32], fill='lightblue')
            # Ground
            draw.rectangle([0, 32, 64, 64], fill='lightgreen')
            # Sun
            draw.ellipse([45, 5, 60, 20], fill='yellow', outline='orange')
            # Mountain
            draw.polygon([(10, 32), (25, 15), (40, 32)], fill='gray')
        else:  # Abstract pattern
            for _ in range(8):
                x1, y1 = random.randint(0, 64), random.randint(0, 64)
                x2, y2 = random.randint(0, 64), random.randint(0, 64)
                color = random.choice(['lightcyan', 'lavender', 'mistyrose', 'honeydew'])
                draw.line([x1, y1, x2, y2], fill=color, width=3)

    def _draw_text(self, draw, colors):
        """Draw text elements"""
        text_options = ["Header Text", "Label:", "Description", "Title", "Info", "Details"]
        text = random.choice(text_options)
        
        x = random.randint(5, 20)
        y = random.randint(15, 35)
        color = random.choice(colors)
        
        draw.text((x, y), text, fill=color)
        
        # Sometimes add underline
        if random.random() > 0.7:
            text_width = len(text) * 6
            draw.line([x, y+12, x+text_width, y+12], fill=color, width=1)

    def _draw_image_placeholder(self, draw, colors):
        """Draw image placeholder"""
        width = random.randint(25, 45)
        height = random.randint(20, 35)
        x = random.randint(5, 64-width-5)
        y = random.randint(5, 64-height-5)
        
        draw.rectangle([x, y, x+width, y+height], fill='lightgray', outline='gray', width=2)
        
        # Add image icon or text
        if width > 20 and height > 15:
            if random.random() > 0.5:
                draw.text((x+width//2-8, y+height//2-4), "IMG", fill='darkgray')
            else:
                # Simple mountain icon
                draw.polygon([(x+5, y+height-5), (x+width//3, y+8), (x+2*width//3, y+height-5)], fill='darkgray')

    def _draw_sliding_menu(self, draw, colors):
        """Draw sliding menu"""
        menu_width = random.randint(15, 25)
        
        # Menu panel
        draw.rectangle([0, 0, menu_width, 64], fill='lightgray', outline='gray', width=1)
        
        # Menu items
        for i in range(4):
            item_y = 8 + i * 12
            draw.rectangle([2, item_y, menu_width-2, item_y+8], fill='white', outline='gray', width=1)
            draw.text((4, item_y+1), f"Item {i+1}", fill='black')

    def _draw_abstract_shape(self, draw, colors):
        """Draw abstract shapes for 'other' category"""
        shape_type = random.randint(0, 6) # 0-5 for specific shapes, 6 for random
        color = random.choice(colors)
        
        if shape_type == 0:  # Scribble
            points = []
            for i in range(10):
                x = random.randint(10, 54)
                y = random.randint(10, 54)
                points.extend([x, y])
            if len(points) >= 6:
                draw.line(points, fill=color, width=2)
        elif shape_type == 1:  # Zigzag
            points = []
            for i in range(8):
                x = 8 + i * 6
                y = 32 + (10 if i % 2 == 0 else -10)
                points.extend([x, y])
            draw.line(points, fill=color, width=2)
        elif shape_type == 2:  # Abstract polygon
            num_points = random.randint(5, 8)
            points = []
            center_x, center_y = 32, 32
            for i in range(num_points):
                angle = i * (360 / num_points)
                radius = random.randint(10, 20)
                x = center_x + radius * np.cos(np.radians(angle))
                y = center_y + radius * np.sin(np.radians(angle))
                points.append((x, y))
            draw.polygon(points, outline=color, width=2)
        elif shape_type == 3:  # Plain background with subtle texture
            for i in range(0, 64, 4):
                alpha = random.randint(240, 250)
                color = (alpha, alpha, alpha)
                draw.line([(i, 0), (i, 64)], fill=color, width=1)
        elif shape_type == 4:  # Background pattern
            for x in range(0, 64, 8):
                for y in range(0, 64, 8):
                    if random.random() > 0.7:
                        draw.rectangle([x, y, x+2, y+2], fill='lightgray')
        elif shape_type == 5:  # Minimal elements
            # Just add a few minimal marks
            for _ in range(random.randint(1, 3)):
                x, y = random.randint(20, 44), random.randint(20, 44)
                draw.ellipse([x, y, x+3, y+3], fill=random.choice(colors))
        else:  # Random shapes
            for _ in range(3):
                x1, y1 = random.randint(10, 52), random.randint(10, 52)
                x2, y2 = random.randint(x1+1, 54), random.randint(y1+1, 54)
                draw.ellipse([x1, y1, x2, y2], outline=color, width=2)

if __name__ == "__main__":
    downloader = EnhancedDataDownloader()
    downloader.download_quickdraw_data()
    downloader.create_enhanced_synthetic_ui_data()