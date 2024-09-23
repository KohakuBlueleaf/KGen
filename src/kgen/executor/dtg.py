import re
from random import shuffle
from contextlib import nullcontext

from transformers import set_seed

from .. import models
from ..utils import same_order_deduplicate
from ..generate import generate


def apply_dtg_prompt(tag_map, target="", aspect_ratio=1.0):
    special_tags = ", ".join(tag_map.get("special", []))
    rating = ", ".join(tag_map.get("rating", []))
    artist = ", ".join(tag_map.get("artist", []))
    characters = ", ".join(tag_map.get("characters", []))
    copyrights = ", ".join(tag_map.get("copyrights", []))
    general = ", ".join(tag_map.get("general", []))
    prompt = f"""
rating: {rating or '<|empty|>'}
artist: {artist.strip() or '<|empty|>'}
characters: {characters.strip() or '<|empty|>'}
copyrights: {copyrights.strip() or '<|empty|>'}
aspect ratio: {f"{aspect_ratio:.1f}" or '<|empty|>'}
target: {'<|' + target + '|>' if target else '<|long|>'}
general: {special_tags}, {general.strip().strip(",")}<|input_end|>
""".strip()

    if models.model_have_quality_info.get(models.current_model_name, None):
        quality = ", ".join(tag_map.get("quality", ["masterpiece"]))
        prompt = f"quality: {quality}\n{prompt}"

    return prompt


def black_list_match(tag, black_list):
    for b in black_list:
        if re.match(b, tag):
            return True
    return False


def tag_gen(
    text_model,
    tokenizer,
    prompt,
    prompt_tags,
    len_target,
    black_list,
    temperature=0.5,
    top_p=0.95,
    top_k=100,
    max_new_tokens=256,
    max_retry=20,
    max_same_output=15,
    seed=None,
):
    retry = max_retry
    llm_gen = ""

    set_seed(seed)

    iter_count = 0
    prev_output = set()
    same_output_count = 0
    while retry >= 0 and same_output_count < max_same_output:
        llm_gen = generate(
            model=text_model,
            tokenizer=tokenizer,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=None,
            max_new_tokens=max_new_tokens,
            stream_output=False,
            autocast_gen=nullcontext,
            prompt_lookup_num_tokens=10,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            seed=seed + iter_count,
        )[0]
        iter_count += 1
        llm_gen = llm_gen.replace("</s>", "").replace("<s>", "")
        orig_prompt = llm_gen.split("<|input_end|>")[0]
        extra = llm_gen.split("<|input_end|>")[-1].strip().strip(",")
        extra_tokens = [
            tok.strip()
            for tok in extra.split(",")
            if not black_list_match(tok.strip(), black_list)
        ]
        extra_tokens = same_order_deduplicate(extra_tokens)
        llm_gen = llm_gen.replace(extra, ", ".join(extra_tokens))

        yield llm_gen, extra_tokens, iter_count

        if set(extra_tokens) == prev_output:
            same_output_count += 1
            retry += 1
        else:
            same_output_count = 0
            prev_output = set(extra_tokens)

        if len(prompt_tags) + len(extra_tokens) < len_target:
            retry -= 1
            shuffle(extra_tokens)
            llm_gen = f"{orig_prompt}<|input_end|>{', '.join(extra_tokens)}"
            prompt = llm_gen.strip().replace("  <|", " <|")
        else:
            break
    yield llm_gen, extra_tokens, iter_count
