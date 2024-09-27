import os

import torch
import orjsonl
from transformers import CLIPConfig, CLIPModel, CLIPProcessor
from PIL import Image
from tqdm import trange, tqdm

from .base import MetricRunner, batch_load
from kgen.utils import remove_repeated_suffix


class CLIPMetricRunner(MetricRunner):
    def __init__(self, model_id="zer0int/LongCLIP-GmP-ViT-L-14", dtype=torch.float16):
        self.dtype = torch.float16
        self.model_id = "zer0int/LongCLIP-GmP-ViT-L-14"
        self.config = CLIPConfig.from_pretrained(model_id)
        self.config.text_config.max_position_embeddings = 248
        self.clip_model = CLIPModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            config=self.config,
            device_map="cuda",
            attn_implementation="flash_attention_2",
        ).requires_grad_(False).half()
        self.clip_processor = CLIPProcessor.from_pretrained(
            model_id, padding="max_length", max_length=248
        )

    @torch.no_grad()
    def eval(self, images, ref_texts=None, ref_images=None):
        inputs = self.clip_processor(
            text=ref_texts,
            images=images,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.cuda()
        outputs = self.clip_model(**inputs)
        cosine_sim = outputs.logits_per_image / self.clip_model.logit_scale.exp()
        return cosine_sim[torch.eye(cosine_sim.shape[0]).bool()]

    def eval_multi(self, images, ref_texts=None, ref_images=None, batch_size=32):
        results = super().eval_multi(images, ref_texts, ref_images, batch_size)
        results = torch.concat(results, dim=0)
        return torch.mean(results, dim=0)


if __name__ == "__main__":

    def load_prompts(file):
        datas = []
        for data in orjsonl.load(file):
            org_data = data["entry"]
            index = org_data["key"]
            result1 = data["result1"]
            result2 = data["result2"]
            org_prompt1 = remove_repeated_suffix(
                org_data["caption_llava_short"].strip()
            )
            org_prompt2 = ".".join(
                remove_repeated_suffix(org_data["caption_llava"].strip()).split(".")[:2]
            )
            gen_prompt1 = result1["generated"]
            gen_prompt2 = result2["extended"]
            datas.append((index, org_prompt1, org_prompt2, gen_prompt1, gen_prompt2))
        return datas

    def load_prompts_gbc(file):
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

    runner = CLIPMetricRunner()
    texts = [[], [], [], []]
    images = [[], [], [], []]

    datas = load_prompts_gbc("./data/gbc-output.jsonl")
    img_files = os.listdir("./data/gbc-img")
    for index, org_prompt1, org_prompt2, gen_prompt1, gen_prompt2 in tqdm(
        datas, total=len(datas), desc="load"
    ):
        org_img1 = f"./data/gbc-img/{index}_short.webp"
        org_img2 = f"./data/gbc-img/{index}_truncate_long.webp"
        gen_img1 = f"./data/gbc-img/{index}_tipo_gen.webp"
        gen_img2 = f"./data/gbc-img/{index}_tipo_extend.webp"
        texts[0].append(org_prompt1)
        texts[1].append(org_prompt2)
        texts[2].append(gen_prompt1)
        texts[3].append(gen_prompt2)
        images[0].append(org_img1)
        images[1].append(org_img2)
        images[2].append(gen_img1)
        images[3].append(gen_img2)

    for idx, name in enumerate(["short", "truncate", "tipo gen", "tipo extend"]):
        result = runner.eval_multi(images[idx], texts[idx], batch_size=500)
        print(idx, name, result.shape, torch.mean(result))
