import re
import random

from .. import models
from ..generate import generate
from ..formatter import seperate_tags
from ..utils import shuffle_iterable, same_order_deduplicate
from ..metainfo import (
    TARGET_TIPO,
    TARGET_TIPO_MAX,
    TARGET_TIPO_NL,
    TARGET_TIPO_NL_MAX,
)


BAN_TAGS = []


def apply_tipo_prompt(meta, general, nl_prompt, mode, length, expand, gen_meta=False):
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


def parse_tipo_result(result: str):
    result = "\n" + result.strip("<s>").strip("</s>").strip()
    result_dict = {}
    for type, content in parse.findall(result):
        type = type.strip()
        content = content.strip()
        if TYPE_MAP.get(type, type) in result_dict:
            continue
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
            result_dict["tag"] = tags
        elif type not in {"short", "long"}:
            result_dict[TYPE_MAP.get(type, type)] = [
                i.strip() for i in content.split(",") if i.strip()
            ]
        else:
            if content == "<|empty|>" or not content:
                continue
            result_dict[TYPE_MAP.get(type, type)] = content
    return result_dict


OPERATION_LIST = {
    # None,
    "None",
    "short_to_tag",
    "short_to_tag_to_long",
    "tag_to_long",
    "tag_to_long_to_short",
    "tag_to_short",
    "tag_to_short_to_long",
    "long_to_tag",
}


def tipo_single_request(
    tag_map,
    nl_prompt="",
    tag_length_target="",
    nl_length_target="",
    add_quality=True,
    operation="",
):
    assert operation in OPERATION_LIST
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
        quality = ", ".join(tag_map.get("quality", []))
        meta["quality"] = quality
    tag_length = tag_length_target or "long"
    nl_length = nl_length_target or "long"

    if operation == "None":
        operation = None
    if operation is not None and not operation.endswith("to_tag"):
        length = nl_length
    else:
        length = tag_length

    # list of [mode, target_output, output_name, length_target, expand]
    # mode with None means tag only
    operations = [
        [operation, length, True],
    ]
    return meta, operations, general, nl_prompt


def parse_tipo_request(
    tag_map,
    nl_prompt="",
    expand_tags=True,
    expand_prompt=True,
    generate_extra_nl_prompt=True,
    tag_first=True,
    tag_length_target="",
    nl_length_target="",
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
        quality = ", ".join(tag_map.get("quality", []))
        meta["quality"] = quality
    tag_length = tag_length_target or "long"
    nl_length = nl_length_target or "long"

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
            op_for_nl_gen = ["tag_to_long", nl_length, True]
        case (_, "", expand, _):  # tag only
            if expand:
                operations = [
                    [None, tag_length, True],
                ]
            op_for_nl_gen = ["tag_to_long", nl_length, False]
        case ("", _, _, expand):  # prompt only
            # long_to_tag -> short_to_tag_to_long
            # expand here means "expand 'long'"
            operations = [
                ["long_to_tag", tag_length, expand],
            ]
            op_for_nl_gen = ["short_to_tag_to_long", nl_length, False]
        case (_, _, False, False):
            op_for_nl_gen = ["short_to_tag_to_long", nl_length, False]
        case (_, _, True, False):
            operations = [
                ["short_to_tag", tag_length, True],
                ["short_to_tag_to_long", nl_length, False],
            ]
            op_for_nl_gen = ["short_to_tag_to_long", nl_length, False]
        case (_, _, False, True):
            operations = [
                ["tag_to_long", tag_length, True],
                ["tag_to_short_to_long", nl_length, False],
            ]
            op_for_nl_gen = ["tag_to_short_to_long", nl_length, False]
        case (_, _, True, True):
            if tag_first:
                operations = [
                    ["short_to_tag", tag_length, True],
                    ["tag_to_long", nl_length, True],
                ]
                op_for_nl_gen = ["short_to_tag_to_long", nl_length, False]
            else:
                operations = [
                    ["tag_to_long", nl_length, True],
                    ["long_to_tag", tag_length, True],
                ]
                op_for_nl_gen = ["tag_to_short_to_long", nl_length, False]
    if generate_extra_nl_prompt:
        operations.append(op_for_nl_gen)
    return meta, operations, general, nl_prompt


def tag_filter(tag):
    if any(b in tag or re.match(b, tag) is not None for b in BAN_TAGS):
        return False
    return True


def post_generate_process(parsed, meta, general, nl_prompt, mode, length, expand):
    if mode is None:
        parsed.pop("extended", None)
        parsed.pop("generated", None)
    else:
        if not "long" in mode:
            parsed.pop("generated", None)
        if not "short" in mode:
            parsed.pop("extended", None)
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
        input_prompts[-1] = output_nl_prompts.pop(0)
    if output_nl_prompts:
        output_nl_prompts = shuffle_iterable(output_nl_prompts[:-1]) + [
            output_nl_prompts[-1]
        ]

    max_length_tags = TARGET_TIPO_MAX[length]
    max_length_nl = TARGET_TIPO_NL_MAX[length]

    if len(input_generals) + len(output_generals) > max_length_tags:
        output_generals = output_generals[: max(max_length_nl - len(input_generals), 0)]
    if len(input_prompts) + len(output_nl_prompts) > max_length_nl:
        output_nl_prompts = output_nl_prompts[
            : max(max_length_nl - len(input_prompts), 0)
        ]
    if "generated" in parsed:
        generated_nl = [
            tag.strip()
            for tag in parsed.get("generated", "").split(".")
            if tag_filter(tag.strip())
            and tag.strip() not in input_prompts
            and tag.strip()
        ]
        if len(generated_nl) > max_length_nl:
            generated_nl = generated_nl[:max_length_nl]
    else:
        generated_nl = []

    new_general = input_generals + output_generals
    new_nl_prompt = input_prompts + output_nl_prompts

    parsed["general"] = same_order_deduplicate(new_general)
    parsed["extended"] = ". ".join(same_order_deduplicate(new_nl_prompt))
    if generated_nl:
        parsed["generated"] = ". ".join(same_order_deduplicate(generated_nl))

    return parsed


def retry_criteria(parsed, check_slice=slice(0, -1), length="long"):
    checks = [
        len(parsed.get("special", []) + parsed.get("general", [])),
        len(parsed.get("extended", "").split(".")),
        len(parsed.get("generated", "").split(".")),
    ]
    low_thresholds = [
        TARGET_TIPO[length],
        TARGET_TIPO_NL[length],
        TARGET_TIPO_NL[length],
    ]
    high_thresholds = [1000, 1000, 1000]

    result = all(
        l <= i <= h
        for l, i, h in list(zip(low_thresholds, checks, high_thresholds))[check_slice]
    )

    return result


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
    **kwargs,
):
    iter_count = 0
    prev_output = set()
    same_output_count = 0
    while iter_count <= max_retry and same_output_count < max_same_output:
        if mode is not None:
            target = mode.split("_to_")[-1]
        else:
            target = "tag"
        prompt = apply_tipo_prompt(
            meta, general, nl_prompt, mode, length, expand, gen_meta
        )
        generation_setting = {
            "temperature": 0.35,
            "min_p": 0.1,
            "top_p": 0.95,
            "top_k": 60,
            "max_new_tokens": 512,
            "seed": seed + iter_count,
        }
        generation_setting.update(kwargs)
        result, input_token_count, token_generated = generate(
            prompt=prompt, **generation_setting
        )
        timing = {}
        timing["generate_pass"] = 1
        timing["generated_tokens"] = token_generated
        timing["input_tokens"] = input_token_count
        if get_timing_detail and hasattr(models.text_model, "export_time"):
            timing.update(models.text_model.export_time())
        if total_timing is not None:
            if "initial_input_tokens" not in total_timing:
                total_timing["initial_input_tokens"] = input_token_count
            for key in timing:
                total_timing[key] = total_timing.get(key, 0) + timing[key]
        parsed = parse_tipo_result(result)
        parsed = post_generate_process(
            parsed, meta, general, nl_prompt, mode, length, expand
        )
        yield result, parsed
        if target == "long" and "generated" not in parsed:
            target = "short"

        slices_map = {
            "tag": slice(0, 1),
            "short": slice(1, 2),
            "long": slice(2, 3),
        }
        # print(mode, end=" ")
        if retry_criteria(parsed, slices_map.get(target, slice(0, -1)), length):
            break
        iter_count += 1
        if result in prev_output:
            same_output_count += 1
        else:
            same_output_count = 0
            prev_output.add(result)

        nl_prompt = (
            parsed.get("extended", "") or parsed.get("generated", "") or nl_prompt
        )
        general = ", ".join(parsed.get("special", []) + parsed.get("general", []))
        nl_prompt = nl_prompt.strip()
    yield result, parsed


def tipo_runner_generator(
    meta, operations, general, nl_prompt, gen_meta=False, **kwargs
):
    total_timing = {}
    for idx, (mode, length, expand) in enumerate(operations):
        is_last = idx == len(operations) - 1
        prompt = apply_tipo_prompt(
            meta, general, nl_prompt, mode, length, expand, gen_meta and is_last
        )
        if length is None and not expand:
            parsed = parse_tipo_result(prompt)
            break
        for result, parsed in generate_with_retry(
            meta,
            general,
            nl_prompt,
            mode,
            length,
            expand,
            gen_meta and is_last,
            seed=kwargs.pop("seed", 0) or random.randint(0, 2**32),
            total_timing=total_timing,
            **kwargs,
        ):
            yield parsed, total_timing
        if not is_last:
            if "generated" in parsed and nl_prompt:
                parsed["extended"] = parsed.pop("generated")
            nl_prompt = (
                parsed.get("generated", []) or parsed.get("extended", []) or nl_prompt
            )
            general = ", ".join(parsed.get("special", []) + parsed.get("general", []))
            nl_prompt = nl_prompt.strip()

    yield parsed, total_timing


def tipo_runner(meta, operations, general, nl_prompt, gen_meta=False, **kwargs):
    for parsed, timing in tipo_runner_generator(
        meta, operations, general, nl_prompt, gen_meta, **kwargs
    ):
        pass
    return parsed, timing
