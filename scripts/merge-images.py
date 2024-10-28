import os
import math
from PIL import Image


PATH = "./output/short-long-gen-extend"
images = os.listdir(PATH)
images_short = [
    Image.open(os.path.join(PATH, i)) for i in images if i.endswith("short.webp")
]
images_gen = [
    Image.open(os.path.join(PATH, i)) for i in images if i.endswith("gen.webp")
]
images_long = [
    Image.open(os.path.join(PATH, i)) for i in images if i.endswith("long.webp")
]
images_extend = [
    Image.open(os.path.join(PATH, i)) for i in images if i.endswith("extend.webp")
]


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
    grid = Image.new("RGB", size=(cols * w, rows * h))

    # Paste the images into the grid
    for i, img in enumerate(images):
        box = (i % cols * w, i // cols * h)
        grid.paste(img, box)

    return grid


image_short = create_image_grid(images_short)
image_gen = create_image_grid(images_gen)
image_long = create_image_grid(images_long)
image_extend = create_image_grid(images_extend)

image_short.save("./output/short.jpg", quality=100)
image_gen.save("./output/gen.jpg", quality=100)
image_long.save("./output/long.jpg", quality=100)
image_extend.save("./output/extend.jpg", quality=100)
