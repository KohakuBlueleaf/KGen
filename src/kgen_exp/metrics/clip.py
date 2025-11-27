import os

import torch
import orjsonl
from transformers import CLIPConfig, AutoModel, AutoProcessor
from PIL import Image
from tqdm import trange, tqdm

from .base import MetricRunner, batch_load
from kgen.utils import remove_repeated_suffix
from kgen.formatter import apply_format


class CLIPMetricRunner(MetricRunner):
    def __init__(self, model_id="zer0int/LongCLIP-GmP-ViT-L-14", dtype=torch.float16):
        self.dtype = torch.float16
        self.clip_model = (
            AutoModel.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="cuda",
                attn_implementation="sdpa",
            )
            .requires_grad_(False)
            .half()
            .cuda()
        )
        self.clip_processor = AutoProcessor.from_pretrained(
            model_id, padding="max_length", use_fast=True
        )

    @torch.no_grad()
    def eval(self, images, ref_texts=None, is_ref=False):
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
        results, _ = super().eval_multi(images, ref_texts, ref_images, batch_size)
        results = torch.concat(results, dim=0)
        return torch.mean(results, dim=0)


if __name__ == "__main__":
    MAX = 1000

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

    runner = CLIPMetricRunner("openai/clip-vit-large-patch14-336")
    # all_results = {}
    # texts = [[], [], [], []]
    # images = [[], [], [], []]

    # datas = load_prompts("./data/coyo-output.jsonl")
    # for index, org_prompt1, org_prompt2, gen_prompt1, gen_prompt2 in tqdm(
    #     datas, total=len(datas), desc="load"
    # ):
    #     org_img1 = f"./data/coyo-img/{index}_short.webp"
    #     org_img2 = f"./data/coyo-img/{index}_truncate_long.webp"
    #     gen_img1 = f"./data/coyo-img/{index}_tipo_gen.webp"
    #     gen_img2 = f"./data/coyo-img/{index}_tipo_extend.webp"
    #     texts[0].append(org_prompt1)
    #     texts[1].append(org_prompt2)
    #     texts[2].append(org_prompt1)
    #     texts[3].append(org_prompt2)
    #     images[0].append(org_img1)
    #     images[1].append(org_img2)
    #     images[2].append(gen_img1)
    #     images[3].append(gen_img2)

    # for idx, name in enumerate(["short", "truncate", "tipo gen", "tipo extend"]):
    #     result = runner.eval_multi(images[idx][:MAX], texts[idx][:MAX], batch_size=500)
    #     all_results[name] = result.item()
    #     # print(idx, name, result.shape, torch.mean(result))

    # def load_prompts(file):
    #     datas = []
    #     for data in orjsonl.load(file):
    #         org_data = data["entry"]
    #         index = org_data["key"]
    #         result1 = data["result1"]
    #         result2 = data["result2"]
    #         org_prompt1 = remove_repeated_suffix(
    #             org_data["caption_llava_short"].strip()
    #         )
    #         org_prompt2 = ".".join(
    #             remove_repeated_suffix(org_data["caption_llava"].strip()).split(".")[:2]
    #         )
    #         gen_prompt1 = result1
    #         gen_prompt2 = result2
    #         datas.append((index, org_prompt1, org_prompt2, gen_prompt1, gen_prompt2))
    #     return datas

    # texts = [[], []]
    # images = [[], []]

    # datas = load_prompts("./data/generated_raw/coyo-output-promptist.jsonl")
    # for index, org_prompt1, org_prompt2, gen_prompt1, gen_prompt2 in tqdm(
    #     datas, total=len(datas), desc="load"
    # ):
    #     gen_img1 = f"./data/short-tlong/Promptist/coyo-{index}-promptist-short.webp"
    #     gen_img2 = f"./data/short-tlong/Promptist/coyo-{index}-promptist-tlong.webp"
    #     texts[0].append(org_prompt1)
    #     texts[1].append(org_prompt2)
    #     images[0].append(gen_img1)
    #     images[1].append(gen_img2)

    # for idx, name in enumerate(["Promptist Short", "Promptist TLong"]):
    #     result = runner.eval_multi(images[idx][:MAX], texts[idx][:MAX], batch_size=500)
    #     all_results[name] = result.item()
    #     # print(idx, name, result.shape, torch.mean(result))

    # texts = [[], []]
    # images = [[], []]

    # datas = load_prompts("./data/generated_raw/coyo-output-gpt2.jsonl")
    # for index, org_prompt1, org_prompt2, gen_prompt1, gen_prompt2 in tqdm(
    #     datas, total=len(datas), desc="load"
    # ):
    #     gen_img1 = f"./data/short-tlong/Prompt-DB/coyo-{index}-gpt2-short.webp"
    #     gen_img2 = f"./data/short-tlong/Prompt-DB/coyo-{index}-gpt2-tlong.webp"
    #     texts[0].append(org_prompt1)
    #     texts[1].append(org_prompt2)
    #     images[0].append(gen_img1)
    #     images[1].append(gen_img2)

    # for idx, name in enumerate(["PromptDB Short", "PromptDB TLong"]):
    #     result = runner.eval_multi(images[idx][:MAX], texts[idx][:MAX], batch_size=500)
    #     all_results[name] = result.item()
    #     # print(idx, name, result.shape, torch.mean(result))

    # texts = [[], []]
    # images = [[], []]

    # datas = load_prompts("./data/generated_raw/coyo-output-oai.jsonl")
    # for index, org_prompt1, org_prompt2, gen_prompt1, gen_prompt2 in tqdm(
    #     datas, total=len(datas), desc="load"
    # ):
    #     gen_img1 = f"./data/short-tlong/GPT4o-mini/coyo-{index}-oai-short.webp"
    #     gen_img2 = f"./data/short-tlong/GPT4o-mini/coyo-{index}-oai-tlong.webp"
    #     texts[0].append(org_prompt1)
    #     texts[1].append(org_prompt2)
    #     images[0].append(gen_img1)
    #     images[1].append(gen_img2)

    # for idx, name in enumerate(["GPT4o-mini Short", "GPT4o-mini TLong"]):
    #     result = runner.eval_multi(images[idx][:MAX], texts[idx][:MAX], batch_size=500)
    #     all_results[name] = result.item()
    #     # print(idx, name, result.shape, torch.mean(result))

    # for name, result in all_results.items():
    #     print(name, result)

    def load_prompts(file):
        datas = []
        for data in orjsonl.load(file):
            org_data = data["entry"]
            index = org_data["index"]
            result = data["result1"]
            org_prompt1 = org_data["caption"]
            datas.append((index, org_prompt1, result))
        return datas

    all_results = {}

    texts = [[], []]
    images = [[], []]

    datas = load_prompts("./data/scenery-output.jsonl")
    for index, org_prompt1, gen_prompt1 in tqdm(datas, total=len(datas), desc="load"):
        org_img1 = f"./data/dan-scenery-webp/{index}_org.webp"
        gen_img1 = f"./data/dan-scenery-webp/{index}_tipo_gen.webp"
        texts[0].append(org_prompt1)
        texts[1].append(org_prompt1)
        images[0].append(org_img1)
        images[1].append(gen_img1)

    for idx, name in enumerate(["Scenery Tag", "TIPO"]):
        result = runner.eval_multi(images[idx][:MAX], texts[idx][:MAX], batch_size=500)
        all_results[name] = result.item()

    def load_prompts(file):
        datas = []
        for data in orjsonl.load(file):
            org_data = data["entry"]
            index = org_data["index"]
            result = data["result"]
            org_prompt1 = org_data["caption"]
            datas.append((index, org_prompt1, result))
        return datas

    texts = []
    images = []

    datas = load_prompts("./data/generated_raw/scenery-output-promptdb.jsonl")
    for index, org_prompt1, gen_prompt1 in tqdm(datas, total=len(datas), desc="load"):
        gen_img1 = f"./data/scenery-tag/Prompt-DB/dan-scenery-{index}-db.webp"
        texts.append(org_prompt1)
        images.append(gen_img1)

    all_results["PromptDB"] = runner.eval_multi(
        images[:MAX], texts[:MAX], batch_size=500
    ).item()

    texts = []
    images = []

    datas = load_prompts("./data/generated_raw/scenery-output-oai.jsonl")
    for index, org_prompt1, gen_prompt1 in tqdm(datas, total=len(datas), desc="load"):
        gen_img1 = f"./data/scenery-tag/GPT4o-mini/dan-scenery-{index}-oai.webp"
        texts.append(org_prompt1)
        images.append(gen_img1)

    all_results["GPT4o-mini"] = runner.eval_multi(
        images[:MAX], texts[:MAX], batch_size=500
    ).item()

    texts = []
    images = []

    datas = load_prompts("./data/generated_raw/scenery-output-promptist.jsonl")
    for index, org_prompt1, gen_prompt1 in tqdm(datas, total=len(datas), desc="load"):
        gen_img1 = f"./data/scenery-tag/Promptist/dan-scenery-{index}-promptist.webp"
        texts.append(org_prompt1)
        images.append(gen_img1)

    all_results["Promptist"] = runner.eval_multi(
        images[:MAX], texts[:MAX], batch_size=500
    ).item()

    for name, result in all_results.items():
        print(name, result)
