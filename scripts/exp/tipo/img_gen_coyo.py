import math
import random

import orjson
import orjsonl
import torch
from tqdm import tqdm
from objprint import objprint
from PIL import Image, ImageFont, ImageDraw

from kgen_exp.diff import load_model, generate, encode_prompts
from kgen.utils import remove_repeated_suffix


def load_prompts(file):
    datas = []
    for data in orjsonl.load(file):
        org_data = data["entry"]
        index = org_data["key"]
        result1 = data["result1"]
        result2 = data["result2"]
        org_prompt1 = remove_repeated_suffix(org_data["caption_llava_short"].strip())
        org_prompt2 = ".".join(
            remove_repeated_suffix(org_data["caption_llava"].strip()).split(".")[:2]
        )
        gen_prompt1 = result1["generated"]
        gen_prompt2 = result2["extended"]
        datas.append((index, org_prompt1, org_prompt2, gen_prompt1, gen_prompt2))
    return datas


def generate_entry(entry, sdxl_pipe):
    index, *prompts = entry
    prompts = list(prompts)[2:]
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
    return list(zip(result, ["short", "truncate_long", "tipo_gen", "tipo_extend"][2:]))


if __name__ == "__main__":
    pipe = load_model(
        "stabilityai/stable-diffusion-xl-base-1.0", "cuda:0", custom_vae=True
    )
    datas = load_prompts("./data/coyo-output-tipo500m.jsonl")

    for entry in tqdm(datas):
        index = entry[0]
        result = generate_entry(entry, pipe)
        for i, (img, prompt) in enumerate(result):
            img.save(f"./output/short-long-gen-extend-500m/{index}_{prompt}.png")
