import os
import random
from PIL import Image

from kgen_exp.img_utils import create_image_grid


PATH = "./download"
all_images = [
    os.path.join(PATH, i) for i in os.listdir(PATH) if i.endswith(".webp")
]
images = {}
for i in all_images:
    basename = os.path.basename(i)
    cate = basename.rsplit("-", 1)[0]
    if cate not in images:
        images[cate] = []
    images[cate].append(i)


for cate in images.keys():
    img = images[cate]
    random.shuffle(img)
    grid = create_image_grid([Image.open(i) for i in img[:16]])
    grid.save(f"./output/tgts-grid/{cate}.jpg", quality=100)
