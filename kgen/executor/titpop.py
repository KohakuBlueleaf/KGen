import re
import random

from .. import models
from ..generate import generate
from ..formatter import seperate_tags
from ..utils import shuffle_iterable, same_order_deduplicate


BAN_TAGS = []


def apply_titpop_prompt(meta, general, nl_prompt, mode, length, expand, gen_meta=False):
    content = {
        "tag": general,
    }
    if nl_prompt and (mode is None or "short" not in mode):
        content["long"] = nl_prompt
    elif nl_prompt:
        content["short"] = nl_prompt

    prompt = ""
    target = ""
    for key, value in meta.items():
        if value:
            prompt += f"{key}: {value}\n"

    if length:
        target += f"<|{length}|>"
    if mode:
        content_order = mode.split("_to_")
        target += f" <|{mode}|>"
    else:
        content_order = ["tag"]
    if gen_meta:
        target += " <|gen_meta|>"
    target = target.strip()
    prompt += f"target: {target}\n"

    for idx, key in enumerate(content_order):
        if key in content and content[key].strip():
            prompt += f"{key}: {content[key]}\n"
        elif idx < len(content_order) - 1:
            prompt += f"{key}: \n"
    if expand:
        prompt = prompt.strip()
    else:
        prompt += f"{key}:"
    return prompt


parse = re.compile(r"\n([^:\n]+):(.*(?:\n(?![^:\n]+:).*)*)")
TYPE_MAP = {
    "short": "extended",
    "long": "generated",
}


def parse_titpop_result(result: str):
    result = "\n" + result.strip("<s>").strip("</s>").strip()
    result_dict = {}
    for type, content in parse.findall(result):
        type = type.strip()
        content = content.strip()
        if type == "tag":
            tags = [i.strip() for i in content.split(",") if i.strip()]
            content = seperate_tags(tags)
            for key, value in content.items():
                value = [i for i in value if i]
                if key in result_dict:
                    if not isinstance(result_dict[key], list):
                        result_dict[key] = [result_dict[key]]
                    result_dict[key].extend(value)
                elif not value:
                    result_dict[key] = []
                else:
                    result_dict[key] = value
        else:
            if content == "<|empty|>" or not content:
                continue
            result_dict[TYPE_MAP.get(type, type)] = content
    return result_dict


def parse_titpop_request(
    tag_map,
    nl_prompt="",
    expand_tags=True,
    expand_prompt=True,
    generate_extra_nl_prompt=True,
    tag_first=True,
    tag_length_target="",
    add_quality=True,
):
    rating = ", ".join(tag_map.get("rating", []))
    artist = ", ".join(tag_map.get("artist", []))
    characters = ", ".join(tag_map.get("characters", []))
    copyrights = ", ".join(tag_map.get("copyrights", []))
    general = ", ".join(tag_map.get("special", []) + tag_map.get("general", []))
    general = general.strip().strip(",")
    meta = {
        "meta": ", ".join(tag_map.get("meta", [])),
        "rating": rating or None,
        "artist": artist.strip() or None,
        "characters": characters.strip() or None,
        "copyrights": copyrights.strip() or None,
    }
    if (
        models.model_have_quality_info.get(models.current_model_name, None)
        or add_quality
    ):
        quality = ", ".join(tag_map.get("quality", []) or ["masterpiece"])
        meta["quality"] = quality
    tag_length = tag_length_target or "long"

    # list of [mode, target_output, output_name, length_target, expand]
    # mode with None means tag only
    operations = []
    op_for_nl_gen = None
    match (general.strip(), nl_prompt.strip(), bool(expand_tags), bool(expand_prompt)):
        case ("", "", _, _):  # no input
            # no input, tag_only -> tag_to_long
            operations = [
                [None, tag_length, True],
            ]
            op_for_nl_gen = ["tag_to_long", tag_length, True]
        case (_, "", expand, _):  # tag only
            if expand:
                operations = [
                    [None, tag_length, True],
                ]
            op_for_nl_gen = ["tag_to_long", tag_length, False]
        case ("", _, _, expand):  # prompt only
            # long_to_tag -> short_to_tag_to_long
            # expand here means "expand 'long'"
            operations = [
                ["long_to_tag", tag_length, expand],
            ]
            op_for_nl_gen = ["short_to_tag_to_long", tag_length, False]
        case (_, _, False, False):
            op_for_nl_gen = ["short_to_tag_to_long", tag_length, False]
        case (_, _, True, False):
            operations = [
                ["short_to_tag", tag_length, True],
                ["short_to_tag_to_long", tag_length, False],
            ]
            op_for_nl_gen = ["short_to_tag_to_long", tag_length, False]
        case (_, _, False, True):
            operations = [
                ["tag_to_long", tag_length, True],
                ["tag_to_short_to_long", tag_length, False],
            ]
            op_for_nl_gen = ["tag_to_short_to_long", tag_length, False]
        case (_, _, True, True):
            if tag_first:
                operations = [
                    ["short_to_tag", tag_length, True],
                    ["tag_to_long", tag_length, True],
                ]
                op_for_nl_gen = ["short_to_tag_to_long", tag_length, False]
            else:
                operations = [
                    ["tag_to_long", tag_length, True],
                    ["long_to_tag", tag_length, True],
                ]
                op_for_nl_gen = ["tag_to_short_to_long", tag_length, False]
    if generate_extra_nl_prompt:
        operations.append(op_for_nl_gen)
    return meta, operations, general, nl_prompt


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

    if len(input_generals) + len(output_generals) > 48:
        output_generals = output_generals[: max(48 - len(input_generals), 0)]
    if len(input_prompts) + len(output_nl_prompts) > 8:
        output_nl_prompts = output_nl_prompts[: max(5 - len(input_prompts), 0)]

    new_general = input_generals + output_generals
    new_nl_prompt = input_prompts + output_nl_prompts

    print(new_general)
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
            general = ", ".join(parsed.get("special", []) + parsed.get("general", []))
            nl_prompt = nl_prompt.strip()

    return parsed, total_timing
