import os
import json
import random
from PIL import Image, ImageDraw, ImageFont

def generate_random_hex_color():
    # Generate a random HEX color code
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def generate_random_color_background(size=(800, 600)):
    # Create a new image with a random background color
    color = generate_random_hex_color()
    return Image.new("RGB", size, color=color)

# Directory containing background images
dir = "/home/baoxiaohe/misc/clean_mockup_bg/train"
bgs = os.listdir(dir)

bg_imgs = []

# Load words from file
with open("./words.txt", "r", encoding="utf-8") as file:
    _words = file.read()
    words = _words.split(",")

# Add JPEG background images to the list
for file in bgs:
    if file.endswith(".jpg"):
        bg_imgs.append(os.path.join(dir, file))

# Load fonts from JSON file
with open("./fonts2.json", "r", encoding="utf-8") as file:
    fonts = json.load(file)

font_names = list(fonts.keys())
texts_per_image = 10

def draw_text(img, word, font, color, k, j, font_name):
    # Draw text on an image and save a cropped version with padding
    draw = ImageDraw.Draw(img)
    
    # Get the bounding box of the text
    bbox = draw.textbbox((0, 0), word, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    max_x = img.width - text_width
    max_y = img.height - text_height
    
    if max_x <= 0 or max_y <= 0:
        print(f"Text '{word}' is too large to fit within the background image.")
        return
    
    # Randomly position the text on the image
    center_x = random.randint(0, img.width)
    center_y = random.randint(0, img.height)
    
    x = center_x - text_width // 2
    y = center_y - text_height // 2
    
    # Ensure the text does not go outside the image boundaries
    x = max(0, min(x, max_x))
    y = max(0, min(y, max_y))
    
    # Draw text on the image
    draw.text((x, y), word, font=font, fill=color)

    # Calculate the extended bounding box for padding
    extended_bbox = [
        max(0, x - 20),
        max(0, y - 20),
        min(img.width, x + text_width + 20),
        min(img.height, y + text_height + 20)
    ]
    
    # Ensure the text is centered in the cropped image
    cropped_width = extended_bbox[2] - extended_bbox[0]
    cropped_height = extended_bbox[3] - extended_bbox[1]
    crop_center_x = x + text_width // 2
    crop_center_y = y + text_height // 2
    
    crop_x1 = max(0, crop_center_x - cropped_width // 2)
    crop_y1 = max(0, crop_center_y - cropped_height // 2)
    crop_x2 = min(img.width, crop_center_x + cropped_width // 2)
    crop_y2 = min(img.height, crop_center_y + cropped_height // 2)
    
    # Adjust the cropping box to ensure it's within the image boundaries
    crop_x1 = max(0, min(crop_x1, img.width - cropped_width))
    crop_y1 = max(0, min(crop_y1, img.height - cropped_height))
    crop_x2 = crop_x1 + cropped_width
    crop_y2 = crop_y1 + cropped_height

    # Crop the image to the adjusted bounding box
    cropped_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    # Create the directory for the font if it doesn't exist
    font_dir = os.path.join("output", font_name)
    os.makedirs(font_dir, exist_ok=True)

    # Save the cropped image
    cropped_img.save(os.path.join(font_dir, f"{font_name}_img_{k}_cpy_{j}.jpg"))

# Generate images with text overlays on existing background images
for i in range(len(font_names)):
    font_name = font_names[i]
    for k in range(len(bg_imgs)):  # Iterate over background images
        preIMG = random.choice(bg_imgs)
        for j in range(texts_per_image):
            img = Image.open(preIMG).copy()  # Ensure a fresh copy of the background image
            
            word = random.choice(words)
            color = generate_random_hex_color()
            font_size = 40
            _font = ImageFont.truetype(fonts[font_name], font_size)
            
            draw_text(img, word, _font, color, k, j, font_name)
    
    # Generate images with random colored backgrounds
    for m in range(2000):  # Generate 2000 images
        img = generate_random_color_background()
        word = random.choice(words)
        color = generate_random_hex_color()
        font_size = 40
        _font = ImageFont.truetype(fonts[font_name], font_size)
        
        draw_text(img, word, _font, color, i, m, font_name)
