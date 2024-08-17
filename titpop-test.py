import re
import random
from time import time
from contextlib import nullcontext

from objprint import objprint
from transformers import AutoTokenizer, logging

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


logging.set_verbosity_error()


DEFAULT_FORMAT = """<|special|>, <|characters|>, <|copyrights|>, 
<|artist|>, 

<|extended|>

<|general|>,

<|quality|>, <|meta|>, <|rating|>
"""

clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
t5_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pile-t5-large")

total_generated_tokens = 0
total_input_tokens = 0

models.load_model("../TITPOP-200M-5ep-ft", device="cpu")
generate(
    max_new_tokens=16,
)

BAN_TAGS = [
    "school swimsuit",
    "blush",
]


def tag_filter(tag):
    if any(b in tag for b in BAN_TAGS):
        return False
    return True


def post_generate_process(parsed, meta, general, nl_prompt, mode, length, expand):
    if "generated" in parsed and nl_prompt:
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
    if output_nl_prompts:
        output_nl_prompts = shuffle_iterable(output_nl_prompts[:-1]) + [
            output_nl_prompts[-1]
        ]
    if len(input_prompts) + len(output_nl_prompts) > 5:
        output_nl_prompts = output_nl_prompts[:max(5 - len(input_prompts), 0)]

    new_general = input_generals + output_generals
    new_nl_prompt = input_prompts + output_nl_prompts

    parsed["general"] = same_order_deduplicate(new_general)
    parsed["extended"] = ". ".join(same_order_deduplicate(new_nl_prompt)) + "."

    return parsed


def retry_criteria(parsed):
    return (
        len(parsed.get("general", [])) >= 38
        and len(parsed.get("general", [])) <= 48
        and len(parsed.get("extended", "").split(".")) <= 5
        and len(parsed.get("generated", "").split(".")) <= 5
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
):
    iter_count = 0
    prev_output = set()
    same_output_count = 0
    while iter_count <= max_retry and same_output_count < max_same_output:
        prompt = apply_titpop_prompt(
            meta, general, nl_prompt, mode, length, expand, gen_meta
        )
        result = generate(
            prompt=prompt,
            temperature=0.25,
            top_p=0.95,
            top_k=60,
            max_new_tokens=512,
            seed=seed + iter_count,
        )
        parsed = parse_titpop_result(result)
        parsed = post_generate_process(
            parsed, meta, general, nl_prompt, mode, length, expand
        )

        if retry_criteria(parsed):
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
        general = ", ".join(
            parsed.get("special", [])
            + parsed.get("meta", [])
            + parsed.get("general", [])
        )
        nl_prompt = nl_prompt.strip()
    return result, parsed


def titpop_runner(meta, operations, general, nl_prompt, gen_meta=False):
    global total_generated_tokens, total_input_tokens
    for idx, (mode, length, expand) in enumerate(operations):
        is_last = idx == len(operations) - 1
        prompt = apply_titpop_prompt(
            meta, general, nl_prompt, mode, length, expand, gen_meta and is_last
        )
        if length is None and expand is None:
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

    return parsed


def _titpop_runner(meta, operations, general, nl_prompt, gen_meta=False):
    global total_generated_tokens, total_input_tokens
    print("=" * 87)
    for idx, (mode, length, expand) in enumerate(operations):
        is_last = idx == len(operations) - 1
        prompt = apply_titpop_prompt(
            meta, general, nl_prompt, mode, length, expand, gen_meta and is_last
        )
        if length is None and expand is None:
            parsed = parse_titpop_result(prompt)
            break
        input_token_count = len(models.tokenizer(prompt)["input_ids"])
        print("=" * 40, "INPUT", "=" * 40)
        print(prompt)
        print("=" * 40, "OUTPUT", "=" * 39)
        result, parsed = generate_with_retry(
            meta,
            general,
            nl_prompt,
            mode,
            length,
            expand,
            gen_meta and is_last,
            seed=random.randint(0, 2**32),
            retry_criteria=lambda x: True,
        )
        output_token_count = len(models.tokenizer(result)["input_ids"])
        token_generated = output_token_count - input_token_count
        total_generated_tokens += token_generated
        total_input_tokens += input_token_count
        print(result)
        print("=" * 87)
        timing = models.text_model.export_time()
        print(
            f"Prompt process time : {timing['prompt_process']:9.3f} ms "
            f"/ {input_token_count:4} tokens "
            f"({input_token_count/timing['prompt_process'] * 1000: 8.3f} Token Per Second)"
        )
        print(
            f"Sampling time       : {timing['total_sampling']:9.3f} ms "
            f"/ {token_generated:4} tokens "
            f"({token_generated/timing['total_sampling'] * 1000 : 8.3f} Token Per Second)"
        )
        print(
            f"Eval time           : {timing['total_eval']:9.3f} ms "
            f"/ {token_generated-1:4} tokens "
            f"({(token_generated-1)/timing['total_eval'] * 1000 : 8.3f} Token Per Second)"
        )
        print(f"Total time          : {timing['total']:9.3f} ms ")
        print("=" * 87)
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

    print(len(parsed["general"]))
    return parsed


tags = (
    "1girl, masterpiece, absurdres, newest, safe, aqua hair, "
    "loli, purple eyes, dragon wings, closed mouth, medium breasts, "
    "dragon girl, dragon tail, pointy ears, long hair, "
    "scenery, sideboob, close-up, from side"
)
nl_prompt = (
    "A cute dragon girl is sitting on a bench in a cafe with cozy lighting"
)


t0 = time()
meta, operations, general, nl_prompt = parse_titpop_request(
    seperate_tags(tags.split(",")),
    nl_prompt,
    tag_length_target="long",
    generate_extra_nl_prompt="<|generated|>" in DEFAULT_FORMAT,
)
width = 1344
height = 768
meta["aspect_ratio"] = f"{width / height:.1f}"
objprint(meta, operations, general, nl_prompt)
result = titpop_runner(meta, operations, general, nl_prompt)
formatted = re.sub(r"([()\[\]<>])", r"\\\1", apply_format(result, DEFAULT_FORMAT))
t1 = time()

print("=" * 87)
print("=" * 40, "INPUT", "=" * 40)
print()
print(tags)
print()
print(nl_prompt)
print()
print("=" * 40, "OUTPUT", "=" * 39)
print()
print(formatted)
print()
print("=" * 87)
print()

print(
    f"""Total Process Time:
    {t1-t0:.2f} sec
"""
)
print(
    f"""Total Processed Tokens:
    {total_input_tokens:} Input Tokens
    {total_generated_tokens:} Output Tokens
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