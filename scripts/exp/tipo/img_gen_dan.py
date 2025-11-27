import math
import random

import orjson
import orjsonl
import torch
from tqdm import tqdm
from objprint import objprint
from PIL import Image, ImageFont, ImageDraw

from kgen_exp.diff import load_model, generate, encode_prompts
from kgen.formatter import apply_format


format = """
<|special|>, 
<|characters|>, <|copyrights|>, 
<|artist|>, 

<|general|>,

<|generated|>.

<|quality|>, <|rating|>, <|meta|>
"""


def load_prompts(file):
    datas = []
    for data in orjsonl.load(file):
        org_data = data["entry"]
        index = org_data["index"]
        datas.append((index, data["result3"]))
    return datas


def generate_entry(entry, sdxl_pipe):
    index, *prompts = entry
    (prompt_embeds, neg_prompt_embeds), (pooled_embeds2, neg_pooled_embeds2) = (
        encode_prompts(
            sdxl_pipe, [apply_format(prompt, format) for prompt in prompts], ""
        )
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
    return list(zip(result, ["tipo_gen"]))


if __name__ == "__main__":
    pipe = load_model("KBlueLeaf/Kohaku-XL-Zeta", "cuda:0")
    datas = load_prompts("./data/danbooru-output-500m.jsonl")

    for entry in tqdm(datas):
        index = entry[0]
        result = generate_entry(entry, pipe)
        for i, (img, prompt) in enumerate(result):
            img.save(f"./output/dan-img-500m/{index}_{prompt}.png")
