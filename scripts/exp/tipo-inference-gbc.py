import torch
import tqdm
from transformers import logging
from orjson import loads, dumps

import kgen.models as models
import kgen.executor.tipo as tipo
from kgen.formatter import seperate_tags, apply_format
from kgen.generate import generate
from kgen.utils import remove_repeated_suffix


# no retry, ban tag mechanism in benchmark
tipo.BAN_TAGS = []


logging.set_verbosity_error()
print(f"threads: {torch.get_num_threads()} {torch.get_num_interop_threads()}")


def task(tags, nl_prompt):
    meta, operations, general, nl_prompt = tipo.parse_tipo_request(
        seperate_tags(tags.split(",")),
        nl_prompt,
        tag_length_target="long",
        generate_extra_nl_prompt=True,
    )
    result, timing = tipo.tipo_runner(
        meta,
        operations,
        general,
        nl_prompt,
        retry_criteria=lambda *args, **kwargs: True,
    )
    return result


if __name__ == "__main__":
    # models.load_model(
    #     "Amber-River/tipo",
    #     device="cuda",
    #     subfolder="500M-epoch3"
    # )
    # models.load_model(
    #     "TIPO-500M_epoch5-F16.gguf",
    #     gguf=True,
    #     device="cuda",
    #     main_gpu=0,
    # )
    models.load_model(
        "TIPO-200M-40Btok-F16.gguf",
        gguf=True,
        device="cuda",
        main_gpu=0,
    )
    with open("./data/gbc.json", "r") as f:
        data = loads(f.read())
    with open("./data/gbc-output.jsonl", "ab") as f:
        for entry in tqdm.tqdm(data[:10000], smoothing=0.01):
            entry.pop("url", None)
            original_caption = entry["original_caption"]
            short_caption = remove_repeated_suffix(entry["short_caption"].strip())
            detail_caption = remove_repeated_suffix(entry["detail_caption"].strip())
            # short to long
            result1 = task("", short_caption)
            # truncated long to long
            result2 = task("", ".".join(detail_caption.split(".")[:2]))
            generated_entry = {
                "entry": entry,
                "result1": result1,
                "result2": result2,
            }
            f.write(dumps(generated_entry) + b"\n")
