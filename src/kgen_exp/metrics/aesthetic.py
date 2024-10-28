import os
import torch
import orjsonl
import orjson

from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from .base import MetricRunner, load


class AestheticRunner(MetricRunner):
    single = True
    multi = True

    def __init__(self):
        # load model and preprocessor
        model, preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float32,
        )
        model = model.cuda().requires_grad_(False)
        self.model = model
        self.preprocessor = preprocessor

    def img_load_func(self, image):
        image = load(image)
        pixel_values = self.preprocessor(images=image, return_tensors="pt").pixel_values
        return pixel_values

    @torch.inference_mode()
    def eval(self, images, ref_texts=None, is_ref=False):
        pixel_values = torch.concat(images).cuda().float()
        with torch.autocast("cuda", dtype=torch.float16):
            result = self.model(pixel_values).logits.squeeze().float().cpu()
        assert torch.all(~torch.isnan(result)), f"nan in result: {torch.sum(torch.isnan(result))}/{result.numel()}"
        return result

    @torch.no_grad()
    def eval_multi(self, images, ref_texts=None, ref_images=None, batch_size=32):
        results, ref_results = super().eval_multi(images, ref_texts, ref_images, batch_size)
        results = torch.concat(results, dim=0).reshape(-1)
        return [i.item() for i in results]


if __name__ == "__main__":
    runner = AestheticRunner()
    images = [None] * 4

    PATH = "./data/coyo-img"
    img_files = [os.path.join(PATH, i) for i in os.listdir(PATH)]
    images[0] = [i for i in img_files if i.endswith("_short.webp")]
    images[1] = [i for i in img_files if i.endswith("_truncate_long.webp")]
    images[2] = [i for i in img_files if i.endswith("_gen.webp")]
    images[3] = [i for i in img_files if i.endswith("_extend.webp")]

    with open("gbc-aesthetic-v25.jsonl", "w") as f:
        for idx, name in enumerate(["short", "truncate", "tipo gen", "tipo extend"]):
            result = runner.eval_multi(images[idx], batch_size=350)
            f.write(str(result) + "\n")
