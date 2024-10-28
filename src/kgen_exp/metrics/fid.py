import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_tensor
from torchmetrics.image import FrechetInceptionDistance
from PIL import Image
from tqdm import trange, tqdm

from .base import MetricRunner


def load_img(file, size):
    with Image.open(file) as img:
        return to_tensor(img.convert("RGB").resize(size, Image.BICUBIC))


class DINOv2FeatureExtractor(nn.Module):
    def __init__(self, name="vitl14", device="cuda"):
        super().__init__()
        self.model = (
            torch.hub.load("facebookresearch/dinov2", "dinov2_" + name)
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
        self.size = 224, 224

    @classmethod
    def available_models(cls):
        return ["vits14", "vitb14", "vitl14", "vitg14"]

    def forward(self, x):
        if x.shape[1] == 1:
            x = torch.cat([x] * 3, dim=1)
        x = self.normalize(x)
        with torch.autocast("cuda", dtype=torch.float16):
            x = self.model(x.cuda()).float()
        # With * x.shape[-1] ** 0.5, the result will be much larger and related to the model size(emb dim)
        # in here we only use normalize to get comparable result across different model sizes
        x = F.normalize(x)  # * x.shape[-1] ** 0.5
        return x


class FIDRunner(MetricRunner):
    multi = True

    def __init__(self, feature=2048, img_size=(299, 299)):
        self.model = (
            FrechetInceptionDistance(
                feature=feature,
                reset_real_features=False,
                normalize=True,
                input_img_size=(3, *img_size),
            )
            .eval()
            .requires_grad_(False)
            .cuda()
        )
        self.img_size = img_size
        self.img_load_func = partial(load_img, size=self.img_size)

    @torch.no_grad()
    def eval(self, images, ref_texts=None, is_ref=False):
        images = torch.stack(images).cuda()
        self.model.update(images, is_ref)

    @torch.no_grad()
    def eval_multi(self, images, ref_texts=None, ref_images=None, batch_size=32):
        super().eval_multi(images, ref_texts, ref_images, batch_size)
        result = self.model.compute()
        self.model.reset()
        return result


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "vitb14"
    swinv2 = DINOv2FeatureExtractor(model)
    runner = FIDRunner(swinv2, (224, 224))
    print(model)

    PATH = r"F:\dataset\HakuBooru\out\scenery"
    img_files = [os.path.join(PATH, i) for i in os.listdir(PATH)]
    refs = [i for i in img_files if i.endswith(".webp")]  # [:32768]
    print("ref image count:", len(refs))

    PATH = "./data/scenery-tag"
    results = {}
    for idx, folder in enumerate([i for i in os.listdir(PATH) if i =="Prompt-DB"]):
        img_files = [
            os.path.join(PATH, folder, i)
            for i in os.listdir(os.path.join(PATH, folder))
        ]

        images = [i for i in img_files if i.endswith(".webp")]
        print(folder, len(images))
        result = runner.eval_multi(
            images, ref_images=refs if not idx else [], batch_size=512
        )
        results[folder] = result

    print("=" * 20)
    for folder, result in results.items():
        print(f"{folder:<10}:", result.item())
