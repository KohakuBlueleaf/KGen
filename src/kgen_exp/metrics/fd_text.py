import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .base import TextMetricRunner


class FDTextRunner(TextMetricRunner):
    multi = True

    def __init__(self, text_model):
        self.text_model = text_model
        self.a_features = []
        self.b_features = []
        self.mode = "a"

    @staticmethod
    def compute_frechet_distance(mu1, sigma1, mu2, sigma2):
        a = (mu1 - mu2).square().sum(dim=-1)
        b = sigma1.trace() + sigma2.trace()
        c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)

        return a + b - 2 * c

    def frechet_distance(self):
        a_features = torch.cat(self.a_features, dim=0)
        b_features = torch.cat(self.b_features, dim=0)
        a_features_sum = a_features.sum(dim=0)
        b_features_sum = b_features.sum(dim=0)
        a_features_cov_sum = torch.matmul(a_features.T, a_features)
        b_features_cov_sum = torch.matmul(b_features.T, b_features)

        mean_a = a_features_sum.unsqueeze(0) / len(a_features)
        mean_b = b_features_sum.unsqueeze(0) / len(b_features)
        cov_a_num = a_features_cov_sum - len(a_features) * torch.matmul(
            mean_a.T, mean_a
        )
        cov_b_num = b_features_cov_sum - len(b_features) * torch.matmul(
            mean_b.T, mean_b
        )
        cov_a = cov_a_num / (len(a_features) - 1)
        cov_b = cov_b_num / (len(b_features) - 1)
        return self.compute_frechet_distance(
            mean_a.squeeze(0), cov_a, mean_b.squeeze(0), cov_b
        )

    @torch.no_grad()
    def eval(self, texts, ref_texts=None):
        features = self.text_model(texts)
        if self.mode == "a":
            self.a_features.append(F.normalize(features.cpu()))
        else:
            self.b_features.append(F.normalize(features.cpu()))

    @torch.no_grad()
    def eval_multi(self, texts, ref_texts=None, batch_size=32):
        self.a_features = []
        self.mode = "a"
        super().eval_multi(texts, None, batch_size)
        if ref_texts is not None:
            self.b_features = []
            self.mode = "b"
            super().eval_multi(ref_texts, None, batch_size)
        torch.cuda.empty_cache()
        return self.frechet_distance()


if __name__ == "__main__":
    model = (
        AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        .float()
        .eval()
        .requires_grad_(False)
        .cuda()
    )
    runner = FDTextRunner(lambda x: torch.from_numpy(model.encode(x)))

    PATH = "./test"
    ref = "./test/reference.txt"
    with open(ref, "r", encoding="utf-8") as f:
        ref_texts = f.readlines()

    results = {}
    for idx, file in enumerate([i for i in os.listdir(PATH)]):
        if file == "reference.txt":
            continue
        texts = []
        with open(os.path.join(PATH, file), "r", encoding="utf-8") as f:
            texts = f.readlines()

        if runner.b_features == []:
            result = runner.eval_multi(texts, ref_texts, batch_size=128)
        else:
            result = runner.eval_multi(texts, batch_size=128)
        results[file] = result

    print("=" * 20)
    for file, result in results.items():
        print(f"{file:<10}:", result.item())
