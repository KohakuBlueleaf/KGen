import random

import torch
import tqdm
from transformers import logging
from orjson import loads, dumps

import kgen.models as models
import kgen.executor.tipo as tipo
from kgen.formatter import seperate_tags, apply_format
from kgen.generate import generate

from objprint import objprint


# no retry, ban tag mechanism in benchmark
tipo.BAN_TAGS = []


logging.set_verbosity_error()
print(f"threads: {torch.get_num_threads()} {torch.get_num_interop_threads()}")


def task(tags, nl_prompt, extra_prompt=True):
    meta, operations, general, nl_prompt = tipo.parse_tipo_request(
        seperate_tags(tags.split(",")),
        nl_prompt,
        tag_length_target="long",
        generate_extra_nl_prompt=extra_prompt,
    )
    result, timing = tipo.tipo_runner(
        meta,
        operations,
        general,
        nl_prompt,
        retry_criteria=lambda *args, **kwargs: True,
    )
    return result


def compute_z_array(s):
    n = len(s)
    Z = [0] * n
    l, r = 0, 0  # Initialize the window [l, r]

    for i in range(1, n):
        if i <= r:
            # Inside the window, we can use previously computed values
            Z[i] = min(r - i + 1, Z[i - l])
        # Attempt to extend the Z-box as far as possible
        while i + Z[i] < n and s[Z[i]] == s[i + Z[i]]:
            Z[i] += 1
        # Update the window if we've extended past r
        if i + Z[i] - 1 > r:
            l, r = i, i + Z[i] - 1
    return Z


def remove_repeated_suffix(text):
    # Strip leading and trailing whitespaces
    text = text.strip()
    if not text:
        return text
    rev_text = text[::-1]
    Z = compute_z_array(rev_text)
    for idx, k in enumerate(Z[::-1]):
        if k != 0:
            break
    text = text[: idx + k - 1]
    return text


if __name__ == "__main__":
    models.load_model(
        "TIPO-500M-ft_TIPO-500M-ft-F16.gguf",
        gguf=True,
        device="cuda",
        main_gpu=0,
    )
    with open("./data/danbooru.json", "r", encoding="utf-8") as f:
        data = loads(f.read())
    with open("./data/danbooru-output-500m.jsonl", "ab") as f:
        for entry in tqdm.tqdm(data[156:1000], smoothing=0.01):
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
            # expand tag + generate NL
            result1 = task(", ".join(tags), "", True)
            # expand long to tag
            result2 = task("", long_caption, False)
            # expand both short and tag and generate new prompt
            result3 = task(
                ", ".join(tags), ".".join(short_caption.split(".")[:2]), True
            )
            generated_entry = {
                "entry": entry,
                "result1": result1,
                "result2": result2,
                "result3": result3,
            }
            f.write(dumps(generated_entry) + b"\n")
