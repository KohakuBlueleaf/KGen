import re
import random
from time import time

import torch
from objprint import objprint
from torch.profiler import profile, ProfilerActivity
from transformers import AutoTokenizer, logging
from viztracer import VizTracer

import kgen.models as models
from kgen.generate import generate
from kgen.formatter import (
    seperate_tags,
    apply_format,
    apply_titpop_prompt,
    parse_titpop_result,
    parse_titpop_request,
)
from kgen.utils import shuffle_iterable, same_order_deduplicate


DEFAULT_FORMAT = """<|special|>, <|characters|>, <|copyrights|>, 
<|artist|>, 

<|general|>,

<|extended|>.

<|quality|>, <|meta|>, <|rating|>
"""
BAN_TAGS = [
    "background",
    "name",
    "text",
    "joke",
    "costume",
    "alternative",
    "speech",
    "stickers",
]


logging.set_verbosity_error()
print(f"threads: {torch.get_num_threads()} {torch.get_num_interop_threads()}")

clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
t5_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pile-t5-large")

# models.load_model(
#     "KBlueLeaf/TITPOP-200M-dev", device="cuda", subfolder="dan-cc-coyo_8000-step"
# )
models.load_model(
    "TITPOP-200M-dev_dan-cc-coyo_20000-step-F16.gguf",
    gguf=True,
    device="cuda",
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
# prof.export_chrome_trace("titpop-test.json")
# exit()


def tag_filter(tag):
    if any(b in tag for b in BAN_TAGS):
        return False
    return True


def post_generate_process(parsed, meta, general, nl_prompt, mode, length, expand):
    if "generated" in parsed and nl_prompt and not parsed.get("extended", "").strip():
        parsed["extended"] = parsed.pop("generated")
    input_tags = [tag.strip() for tag in general.split(",")]
    input_prompts = [tag.strip() for tag in nl_prompt.split(".") if tag.strip()]

    input_generals = [tag for tag in parsed.get("general", []) if tag in input_tags]
    output_generals = shuffle_iterable(
        [
            tag
            for tag in parsed.get("general", [])
            if tag_filter(tag) and tag not in input_tags
        ]
    )
    output_nl_prompts = [
        tag.strip()
        for tag in parsed.get("extended", "").split(".")
        if tag_filter(tag.strip()) and tag.strip() not in input_prompts and tag.strip()
    ]
    if output_nl_prompts and input_prompts[-1] in output_nl_prompts[0]:
        input_prompts[-1] = output_nl_prompts[0]
        output_nl_prompts.pop(0)
    if output_nl_prompts:
        output_nl_prompts = shuffle_iterable(output_nl_prompts[:-1]) + [
            output_nl_prompts[-1]
        ]
    if len(input_prompts) + len(output_nl_prompts) > 5:
        output_nl_prompts = output_nl_prompts[: max(5 - len(input_prompts), 0)]

    new_general = input_generals + output_generals
    new_nl_prompt = input_prompts + output_nl_prompts

    parsed["general"] = same_order_deduplicate(new_general)
    parsed["extended"] = ". ".join(same_order_deduplicate(new_nl_prompt))

    return parsed


def retry_criteria(parsed, check_slice=slice(0, -1)):
    checks = [
        len(parsed.get("special", []) + parsed.get("general", [])),
        len(parsed.get("extended", "").split(".")),
        len(parsed.get("generated", "").split(".")),
    ]
    low_thresholds = [36, 4, 4]
    high_thresholds = [1000, 1000, 1000]
    print(checks)
    return all(
        l <= i <= h
        for l, i, h in list(zip(low_thresholds, checks, high_thresholds))[check_slice]
    )


def generate_with_retry(
    meta,
    general,
    nl_prompt,
    mode,
    length,
    expand,
    gen_meta,
    seed=0,
    max_retry=10,
    max_same_output=5,
    retry_criteria=retry_criteria,
    total_timing=None,
    get_timing_detail=True,
):
    iter_count = 0
    prev_output = set()
    same_output_count = 0
    while iter_count <= max_retry and same_output_count < max_same_output:
        target = mode.split("_to_")[-1]
        prompt = apply_titpop_prompt(
            meta, general, nl_prompt, mode, length, expand, gen_meta
        )
        result, input_token_count, token_generated = generate(
            prompt=prompt,
            temperature=0.5,
            min_p=0.1,
            top_p=0.95,
            top_k=60,
            max_new_tokens=512,
            seed=seed + iter_count,
        )
        timing = {}
        timing["generate_pass"] = 1
        timing["generated_tokens"] = token_generated
        timing["input_tokens"] = input_token_count
        if get_timing_detail and hasattr(models.text_model, "export_time"):
            timing.update(models.text_model.export_time())
        if total_timing is not None:
            for key in timing:
                total_timing[key] = total_timing.get(key, 0) + timing[key]
        parsed = parse_titpop_result(result)
        parsed = post_generate_process(
            parsed, meta, general, nl_prompt, mode, length, expand
        )
        if target == "long" and "generated" not in parsed:
            target = "short"

        slices_map = {
            "tag": slice(0, 1),
            "short": slice(1, 2),
            "long": slice(2, 3),
        }
        if retry_criteria(parsed, slices_map.get(target, slice(0, -1))):
            break
        iter_count += 1
        if result in prev_output:
            same_output_count += 1
        else:
            same_output_count = 0
            prev_output.add(result)

        nl_prompt = (
            parsed.get("generated", []) or parsed.get("extended", []) or nl_prompt
        )
        general = ", ".join(parsed.get("special", []) + parsed.get("general", []))
        nl_prompt = nl_prompt.strip()
    return result, parsed


def titpop_runner(meta, operations, general, nl_prompt, gen_meta=False):
    total_timing = {}
    for idx, (mode, length, expand) in enumerate(operations):
        is_last = idx == len(operations) - 1
        prompt = apply_titpop_prompt(
            meta, general, nl_prompt, mode, length, expand, gen_meta and is_last
        )
        if length is None and not expand:
            parsed = parse_titpop_result(prompt)
            break
        result, parsed = generate_with_retry(
            meta,
            general,
            nl_prompt,
            mode,
            length,
            expand,
            gen_meta and is_last,
            seed=random.randint(0, 2**32),
            total_timing=total_timing,
        )
        if not is_last:
            if "generated" in parsed and nl_prompt:
                parsed["extended"] = parsed.pop("generated")
            nl_prompt = (
                parsed.get("generated", []) or parsed.get("extended", []) or nl_prompt
            )
            general = ", ".join(
                parsed.get("special", [])
                + parsed.get("meta", [])
                + parsed.get("general", [])
            )
            nl_prompt = nl_prompt.strip()

    return parsed, total_timing


tags = nl_prompt = ""
tags = """
1girl,
daiichi ruby (umamusume), umamusume,

solo, cherry blossoms, outdoor, kimono, sideboob, expressionless,

masterpiece, newest, absurdres, sensitive
"""
# nl_prompt = ""
# tags = """
# masterpiece, scenery, absurdres, safe, newest, no humans, cyberpunk
# """
nl_prompt = """
An illustration of a girl
"""


def task(tags, nl_prompt):
    width = 1344
    height = 768
    meta, operations, general, nl_prompt = parse_titpop_request(
        seperate_tags(tags.split(",")),
        nl_prompt,
        tag_length_target="long",
        generate_extra_nl_prompt="<|generated|>" in DEFAULT_FORMAT or not nl_prompt,
    )
    # meta["aspect_ratio"] = f"{width / height:.1f}"
    # addon_meta = meta.pop("meta", "")
    result, timing = titpop_runner(meta, operations, general, nl_prompt)
    # result["meta"] = addon_meta
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
