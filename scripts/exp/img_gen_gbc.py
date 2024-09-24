import math
import random

import orjson
import orjsonl
import torch
from objprint import objprint
from PIL import Image, ImageFont, ImageDraw

from kgen_exp.diff import load_model, generate, encode_prompts
from kgen.utils import remove_repeated_suffix


def load_prompts(file):
    datas = []
    for data in orjsonl.load(file):
        org_data = data["entry"]
        index = org_data["index"]
        result1 = data["result1"]
        result2 = data["result2"]
        org_prompt1 = remove_repeated_suffix(org_data["short_caption"].strip())
        org_prompt2 = ".".join(
            remove_repeated_suffix(org_data["detail_caption"].strip()).split(".")[:2]
        )
        gen_prompt1 = result1["generated"]
        gen_prompt2 = result2["extended"]
        datas.append((index, org_prompt1, org_prompt2, gen_prompt1, gen_prompt2))
    return datas


def generate_entry(entry, sdxl_pipe):
    index, *prompts = entry
    (prompt_embeds, neg_prompt_embeds), (pooled_embeds2, neg_pooled_embeds2) = (
        encode_prompts(sdxl_pipe, prompts, "")
    )
    result = generate(
        sdxl_pipe,
        prompt_embeds,
        neg_prompt_embeds,
        pooled_embeds2,
        neg_pooled_embeds2,
        num_inference_steps=16,
        width=1024,
        height=1024,
        guidance_scale=3.0,
    )
    return list(zip(result, ["short", "truncated long", "tipo gen", "tipo extend"]))


def create_image_grid_with_prompts(image_prompt_pairs, rows=None, cols=None, font_path=None, font_size=20, prompt_height=30):
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
    grid = Image.new('RGB', size=(cols*w, rows*(h+prompt_height)), color='white')
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
        prompt = prompt[:int(w/7)]  # Truncate prompt if too long (rough estimate)
        text_width = draw.textlength(prompt, font=font)
        text_x = x + (w - text_width) / 2  # Center text
        draw.text((text_x, y), prompt, fill="black", font=font)
    
    return grid


if __name__ == "__main__":
    pipe = load_model("stabilityai/stable-diffusion-xl-base-1.0")
    datas = load_prompts("./data/gbc-output.jsonl")

    result = generate_entry(random.choice(datas), pipe)
    grid = create_image_grid_with_prompts(result)
    grid.save("grid.png")
