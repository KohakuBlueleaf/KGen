import math
from PIL import Image, ImageFont, ImageDraw


def create_image_grid(images, rows=None, cols=None):
    """
    Create a grid of images from a list of PIL Image objects.
    
    :param images: List of PIL Image objects
    :param rows: Number of rows in the grid (optional)
    :param cols: Number of columns in the grid (optional)
    :return: A new PIL Image object with the grid
    """
    # Determine the number of rows and columns
    n = len(images)
    if rows is None and cols is None:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
    elif rows is None:
        rows = math.ceil(n / cols)
    elif cols is None:
        cols = math.ceil(n / rows)
    
    # Get the size of the first image
    w, h = images[0].size
    
    # Create a new image with the appropriate size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    # Paste the images into the grid
    for i, img in enumerate(images):
        box = (i % cols * w, i // cols * h)
        grid.paste(img, box)
    
    return grid


def create_image_grid_with_prompts(
    image_prompt_pairs,
    rows=None,
    cols=None,
    font_path=None,
    font_size=20,
    prompt_height=30,
):
    """
    Create a grid of images with prompts from a list of (PIL Image, prompt) pairs.

    :param image_prompt_pairs: List of (PIL Image, prompt) tuples
    :param rows: Number of rows in the grid (optional)
    :param cols: Number of columns in the grid (optional)
    :param font_path: Path to a TTF font file (optional)
    :param font_size: Font size for the prompts (default: 20)
    :param prompt_height: Height of the area for each prompt (default: 30)
    :return: A new PIL Image object with the grid
    """
    # Determine the number of rows and columns
    n = len(image_prompt_pairs)
    if rows is None and cols is None:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
    elif rows is None:
        rows = math.ceil(n / cols)
    elif cols is None:
        cols = math.ceil(n / rows)

    # Get the size of the first image
    w, h = image_prompt_pairs[0][0].size

    # Create a new image with the appropriate size
    grid = Image.new("RGB", size=(cols * w, rows * (h + prompt_height)), color="white")
    draw = ImageDraw.Draw(grid)

    # Load font
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    # Paste the images and draw prompts
    for i, (img, prompt) in enumerate(image_prompt_pairs):
        x = i % cols * w
        y = i // cols * (h + prompt_height)

        # Paste the image
        grid.paste(img, (x, y + prompt_height))

        # Draw the prompt
        prompt = prompt[: int(w / 7)]  # Truncate prompt if too long (rough estimate)
        text_width = draw.textlength(prompt, font=font)
        text_x = x + (w - text_width) / 2  # Center text
        draw.text((text_x, y), prompt, fill="black", font=font)

    return grid