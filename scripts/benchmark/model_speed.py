import re
import os
import random
from time import time
from orjson import loads, dumps

import torch
from transformers import logging
from tqdm import trange, tqdm
from objprint import objprint

import kgen.models as models
import kgen.executor.tipo as tipo
from kgen.formatter import seperate_tags, apply_format
from kgen.generate import generate
from kgen.utils import remove_repeated_suffix


DEFAULT_FORMAT = """<|special|>, <|characters|>, <|copyrights|>, 
<|artist|>, 

<|extended|>.

<|general|>,

<|quality|>, <|meta|>, <|rating|>
"""
tipo.BAN_TAGS = []
tipo.retry_criteria = lambda *args: True


logging.set_verbosity_error()
print(f"threads: {torch.get_num_threads()} {torch.get_num_interop_threads()}")

model_name, gguf_list = models.tipo_model_list[-2]
models.download_gguf(model_name, gguf_list[0])
models.load_model(
    os.path.basename(model_name)+"_"+gguf_list[0],
    gguf=True,
    device="cuda",
    # main_gpu=0,
)
# models.load_model(
#     model_name,
#     gguf=False,
#     device="cuda",
#     # main_gpu=0,
# )
generate(max_new_tokens=4)


def task(tags: list[str], nl_prompt: str):
    width = 832
    height = 1216
    meta, operations, general, nl_prompt = tipo.parse_tipo_request(
        seperate_tags(tags),
        nl_prompt,
        tag_length_target="long",
        generate_extra_nl_prompt=nl_prompt
        and "<|generated|>" in DEFAULT_FORMAT
        or (
            not nl_prompt
            and (
                "<|generated|>" in DEFAULT_FORMAT 
                or "<|extended|>" in DEFAULT_FORMAT
            )
        ),
    )
    print(operations)
    meta["aspect_ratio"] = f"{width / height:.1f}"
    result, timing = tipo.tipo_runner(meta, operations, general, nl_prompt)
    formatted = re.sub(r"([()\[\]])", r"\\\1", apply_format(result, DEFAULT_FORMAT))
    return formatted, timing


if __name__ == "__main__":
    test = 1#00

    with open("./data/danbooru.json", "rb") as f:
        data = loads(f.read())
    entries = []
    for entry in tqdm(data[:test], smoothing=0.01):
        short_caption = remove_repeated_suffix(entry["florence_short"])
        long_caption = remove_repeated_suffix(
            entry.get("phi3v_horny", None) or entry["florence_long"]
        )
        # Some outlier have TOO MANY tags, need to cutoff
        info_tags = (
            entry["artist"][:5]
            + entry["character"][:5]
            + entry["copyright"][:5]
            + entry["meta"][:10]
            + entry["rating"]
            + entry["special"]
            + entry["year"]
        )
        content_tags = entry["general"]
        random.shuffle(content_tags)
        tags = info_tags + content_tags[:5]
        entries.append((tags, short_caption, long_caption))

    results = []
    for i in trange(test):
        tags, short_caption, long_caption = entries[i]
        t0 = time()
        formatted, timing = task(tags, "")
        t1 = time()
        timing["generate_time"] = t1 - t0
        results.append(timing)

    total = {}
    for k in results[0].keys():
        total[k] = sum([x[k] for x in results])

    average = {k: v / test for k, v in total.items()}

    objprint(total)
    objprint(average)