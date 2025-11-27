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

import base64
import mimetypes
import os
from google import genai
from google.genai import types


def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)


def generate(prompt, prefix):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-flash-preview-image-generation"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_modalities=[
            "IMAGE",
            "TEXT",
        ],
    )

    file_index = 0
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        if (
            chunk.candidates[0].content.parts[0].inline_data
            and chunk.candidates[0].content.parts[0].inline_data.data
        ):
            file_name = f"output/short-long-gen-extend-g2f/{prefix}_{file_index}"
            file_index += 1
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            save_binary_file(f"{file_name}{file_extension}", data_buffer)
        else:
            pass


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


if __name__ == "__main__":
    datas = load_prompts("./data/coyo-output.jsonl")[100:1000]

    for entry in tqdm(datas):
        index = entry[0]
        org_prompt1 = entry[1]
        org_prompt2 = entry[2]
        gen_prompt1 = entry[3]
        gen_prompt2 = entry[4]

        generate(
            f"Refine this prompt than generate one image with refined prompt: {org_prompt1}",
            f"{index}-short",
        )
        generate(
            f"Refine this prompt than generate one image with refined prompt: {org_prompt2}",
            f"{index}-tlong",
        )
        generate(
            f"Generate one image with provided prompt: {gen_prompt1}", f"{index}-gen"
        )
        generate(
            f"Generate one image with provided prompt: {gen_prompt2}", f"{index}-extend"
        )
