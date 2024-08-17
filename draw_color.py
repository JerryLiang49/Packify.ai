import os
import json
import random
from PIL import Image, ImageDraw, ImageFont

def generate_random_hex_color():
    # Generate a random HEX color code
    range_val = 64
    step = 256 / range_val
    r = random.randint(0, range_val - 1) * step + random.randint(0, 3) * 5
    g = random.randint(0, range_val - 1) * step + random.randint(0, 3) * 5
    b = random.randint(0, range_val - 1) * step + random.randint(0, 3) * 5
    _num = int(r * 256 * 256 + g * 256 + b)
    hex_number = "{:06x}".format(_num)
    return f"#{hex_number}"

def hex_to_rgb(hex_str):
    # Convert HEX color string to RGB tuple
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))

# Load words and font paths from files
with open("./datasets/words.txt", "r", encoding="utf-8") as file:
    _words = file.read()
    words = _words.split(",")

with open("./datasets/fonts2.json", "r", encoding="utf-8") as file:
    fonts = json.load(file)

font_names = list(fonts.keys())

# List all background images
bgs = os.listdir("/home/baoxiaohe/zd/clean_background/train_data/origin")

def draw_text(img, words, font, color, hex_color, k, j):
    # Draw text on an image with specified properties
    draw = ImageDraw.Draw(img)

    # Get the bounding box of the text
    bbox = draw.textbbox((0, 0), words[0], font=font)
    text_width = bbox[2] - bbox[0]
    text_height = (bbox[3] - bbox[1]) * len(words)

    max_x = img.width - text_width
    max_y = img.height - text_height

    # Check if text fits within the image dimensions
    if max_x <= 0 or max_y <= 0:
        print(f"Text '{words[0]}' is too large to fit within the background image.")
        return

    # Randomly position the text on the image
    center_x = random.randint(0, max_x)
    center_y = random.randint(0, max_y)

    x = center_x - text_width // 2
    y = center_y - text_height // 2

    # Ensure the text does not go outside the image boundaries
    x = max(0, min(x, max_x))
    y = max(0, min(y, max_y))

    # Draw each word of the text on the image
    for idx, word in enumerate(words):
        draw.text((x, y + (bbox[3] - bbox[1]) * idx), word, font=font, fill=color)

    # Define a fixed size for the cropped image
    cropped_width = 300
    cropped_height = 300

    # Ensure the text is centered in the cropped image
    crop_center_x = x + text_width // 2
    crop_center_y = y + text_height // 2

    crop_x1 = max(0, crop_center_x - cropped_width // 2)
    crop_y1 = max(0, crop_center_y - cropped_height // 2)
    crop_x2 = min(img.width, crop_center_x + cropped_width // 2)
    crop_y2 = min(img.height, crop_center_y + cropped_height // 2)

    # Adjust the cropping box to ensure it's within the image boundaries
    if crop_x2 - crop_x1 < cropped_width:
        crop_x1 = max(0, crop_x2 - cropped_width)
        crop_x2 = crop_x1 + cropped_width
    if crop_y2 - crop_y1 < cropped_height:
        crop_y1 = max(0, crop_y2 - cropped_height)
        crop_y2 = crop_y1 + cropped_height

    # Crop the image to the fixed size
    cropped_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    # Create the output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Save the cropped image with hex color in the filename
    cropped_img.save(f"./models/color_patch/imgbg_{k}_cpy_{j}_{hex_color}.jpg")

# Generate images with text overlays
for i in range(len(font_names)):
    font_name = font_names[i]
    for k in range(200):  # Number of different images to generate
        # Create a random colored background image
        bg_color = generate_random_hex_color()
        img = Image.open(f"/home/baoxiaohe/zd/clean_background/train_data/origin/{random.choice(bgs)}")

        hex_color = generate_random_hex_color()
        color = hex_to_rgb(hex_color)
        for j in range(10):
            words_num = random.randint(1, 10)
            _words = [random.choice(words) for _ in range(words_num)]  # Randomly select words

            # Randomize the font size
            font_size = random.randint(1, 40)
            _font = ImageFont.truetype(f"./datasets/{fonts[font_name]}", font_size)

            draw_text(img, _words, _font, color, hex_color, k, j)  # Draw text on the image
        print(f"processed {k} of 200, {i} of {len(font_names)}")
