import os
import sys
from functools import partial
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_tensor
from vendi_score import vendi
from PIL import Image
from tqdm import trange, tqdm
from transformers import AutoModel

from .base import TextMetricRunner


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class VendiTextRunner(TextMetricRunner):
    multi = True

    def __init__(self, text_model):
        self.text_model = text_model
        self.features = []

    @torch.no_grad()
    def eval(self, texts, ref_texts=None):
        features = self.text_model(texts).float().cpu()
        self.features.append(features)

    @torch.no_grad()
    def eval_multi(self, texts, ref_texts=None, batch_size=32):
        self.features = []
        super().eval_multi(texts, ref_texts, batch_size)
        torch.cuda.empty_cache()
        all_features = torch.cat(self.features, dim=0)
        normed_all_features = F.normalize(all_features)
        # normalize the cosine similarity to [0, 1]
        similarities = normed_all_features @ normed_all_features.T  # * 0.5 + 0.5
        print(
            similarities[similarities != 1].mean(),
            similarities[similarities != 1].std(),
        )
        result = vendi.score_K(similarities.cpu().numpy(), q=1, normalize=True)
        return result, similarities.cpu().numpy()


if __name__ == "__main__":
    model = (
        AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        .bfloat16()
        .eval()
        .requires_grad_(False)
        .cuda()
    )
    runner = VendiTextRunner(lambda x: torch.from_numpy(model.encode(x)))

    PATH = "./test"
    results = {}
    simmats = {}
    for idx, file in enumerate([i for i in os.listdir(PATH)]):
        texts = []
        with open(os.path.join(PATH, file), "r", encoding="utf-8") as f:
            texts = sorted(f.readlines())

        result, sim_mat = runner.eval_multi(texts, batch_size=128)
        results[file] = result
        simmats[file] = sim_mat

    print("=" * 20)
    for file, result in results.items():
        print(f"{file:<10}:", result.item())

    np.save("./output/tgts-sim.npy", simmats)
