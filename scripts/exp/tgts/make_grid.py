import os
import random

import numpy as np
from PIL import Image

from kgen_exp.img_utils import create_image_grid

tgts_sim = np.load("./output/img-tgts-sim.npy", allow_pickle=True).tolist()
model_order = [
    ("conventional", "Nucleus Sampling (Temp 1.0, min-P 0.1)"),
    ("beam_search", "Beam Search"),
    ("stochastic_beam_search", "Stochastic Beam Search"),
    ("div_beam_search", "Diverse Beam Search"),
    ("cg-mcts_exp-0", "cg-MCTS (exp 0.5)"),
    ("cg-mcts_exp-1", "cg-MCTS (exp 1.0)"),
    ("cg-mcts_exp-2", "cg-MCTS (exp 2.0)"),
    ("cg-mcts_exp-3", "cg-MCTS (exp 3.0)"),
    ("rw-mcts_exp-0", "rw-MCTS (exp 0.5)"),
    ("rw-mcts_exp-1", "rw-MCTS (exp 1.0)"),
    ("rw-mcts_exp-2", "rw-MCTS (exp 2.0)"),
    ("rw-mcts_exp-3", "rw-MCTS (exp 3.0)"),
]
print(tgts_sim.keys())


PATH = "./download"
all_images = [os.path.join(PATH, i) for i in os.listdir(PATH) if i.endswith(".webp")]
images = {}
images_simmat = {}
for i in all_images:
    basename = os.path.basename(i)
    cate = basename.rsplit("-", 1)[0]
    if cate not in images:
        images[cate] = []
        images_simmat[cate] = tgts_sim[cate]
    images[cate].append(i)


for cate in images.keys():
    img = images[cate]
    simmat = images_simmat[cate]
    index = np.argsort(np.min(simmat, axis=1), axis=0)
    img = [img[i] for i in index]
    grid = create_image_grid([Image.open(i) for i in img[30 : 30 + 9]])
    grid.save(f"./output/tgts-grid/{cate}.jpg", quality=100)
