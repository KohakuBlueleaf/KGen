import re
import random
from time import time

import torch
from transformers import AutoTokenizer, logging

import kgen.models as models
import kgen.executor.tipo as tipo
from kgen.formatter import seperate_tags, apply_format
from kgen.generate import generate


DEFAULT_FORMAT = """<|special|>, <|characters|>, <|copyrights|>, 
<|artist|>, 

<|extended|>.

<|general|>,

<|generated|>.

<|quality|>, <|meta|>, <|rating|>
"""
tipo.BAN_TAGS = [
    "background",
    "name",
    "text",
    "joke",
    "costume",
    "alternative",
    "speech",
    "stickers",
    "hat",
]


logging.set_verbosity_error()
print(f"threads: {torch.get_num_threads()} {torch.get_num_interop_threads()}")

clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
t5_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pile-t5-large")

# models.load_model("Amber-River/tipo", device="cuda", subfolder="500M-epoch3")
models.load_model(
    "TIPO-500M_epoch5-F16.gguf",
    gguf=True,
    device="cuda",
    main_gpu=0,
)
generate(max_new_tokens=4)

# tracer = VizTracer()
# tracer.start()
# generate(max_new_tokens=16)
# tracer.stop()
# tracer.save()
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#     generate(
#         max_new_tokens=16,
#     )
# prof.export_chrome_trace("tipo-test.json")
# exit()


tags = nl_prompt = ""
tags = """
""".strip()
# nl_prompt = ""
# tags = """
# masterpiece, scenery, absurdres, safe, newest, no humans, cyberpunk
# """
nl_prompt = """
An illustration of a girl
""".strip()


def task(tags, nl_prompt):
    width = 832
    height = 1216
    meta, operations, general, nl_prompt = tipo.parse_tipo_request(
        seperate_tags(tags.split(",")),
        nl_prompt,
        tag_length_target="long",
        generate_extra_nl_prompt=not nl_prompt,
    )
    meta["aspect_ratio"] = f"{width / height:.1f}"
    result, timing = tipo.tipo_runner(meta, operations, general, nl_prompt)
    formatted = re.sub(r"([()\[\]])", r"\\\1", apply_format(result, DEFAULT_FORMAT))
    return formatted, timing


if __name__ == "__main__":
    clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    t5_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pile-t5-large")

    test = 1
    start = time()
    for _ in range(test):
        t0 = time()
        formatted, timing = task(tags, nl_prompt)
        t1 = time()
    finish = time()
    print(f"Total cost {(finish - start)}s | {test} iter")

    print(timing)
    print("=" * 87)
    print("=" * 40, "INPUT", "=" * 40)
    print()
    if tags.strip().strip("\n"):
        print(tags.strip().strip("\n"))
        print()
    if nl_prompt.strip().strip("\n"):
        print(nl_prompt.strip().strip("\n"))
        print()
    print("=" * 40, "OUTPUT", "=" * 39)
    print()
    print(formatted.strip())
    print()
    print("=" * 87)
    print()
    timing["total"] = t1 - t0
    total = timing["total"]
    generate_pass = timing["generate_pass"]

    print(
        f"""Process Time:
    Total    || {total:5.2f} sec / {generate_pass:5} Passes | {generate_pass/total:7.2f} Passes Per Second
    """
    )
    if "generated_tokens" in timing:
        total_generated_tokens = timing["generated_tokens"]
        total_input_tokens = timing["input_tokens"]
        print(
            f"""Processed Tokens:
    {total_input_tokens:} Input Tokens
    {total_generated_tokens:} Output Tokens
        """
        )
    if "generated_tokens" in timing and "total_sampling" in timing:
        sampling_time = timing["total_sampling"] / 1000
        process_time = timing["prompt_process"] / 1000
        model_time = timing["total_eval"] / 1000

        print(
            f"""    Process  || {process_time:5.2f} sec / {total_input_tokens:5} Tokens | {total_input_tokens/process_time:7.2f} Tokens Per Second
    Sampling || {sampling_time:5.2f} sec / {total_generated_tokens:5} Tokens | {total_generated_tokens/sampling_time:7.2f} Tokens Per Second
    Eval     || {model_time:5.2f} sec / {total_generated_tokens:5} Tokens | {total_generated_tokens/model_time:7.2f} Tokens Per Second
        """
        )

    formatted_clip_tokens = len(clip_tokenizer(formatted)["input_ids"])
    formatted_t5_tokens = len(t5_tokenizer(formatted)["input_ids"])
    print(
        f"""Length of Formatted Prompt: 
    {formatted_clip_tokens} CLIP tokens
    {formatted_t5_tokens} T5 tokens
    """
    )
    print("=" * 87)
