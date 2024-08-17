import os
import re
import pathlib

from . import models
from .metainfo import SPECIAL, POSSIBLE_QUALITY_TAGS, RATING_TAGS


tag_list_folder = pathlib.Path(os.path.dirname(__file__)) / "tag-list"
tag_lists = {
    os.path.splitext(f)[0]: set(open(tag_list_folder / f).read().strip().split("\n"))
    for f in os.listdir(tag_list_folder)
    if f.endswith(".txt")
}
tag_lists["special"] = set(SPECIAL)
tag_lists["quality"] = set(POSSIBLE_QUALITY_TAGS)
tag_lists["rating"] = set(RATING_TAGS)


def seperate_tags(all_tags):
    all_tags = [i.strip() for i in all_tags if i.strip()]
    tag_map = {cate: [] for cate in tag_lists.keys()}
    tag_map["general"] = []
    for tag in all_tags:
        for cate in tag_lists.keys():
            if tag in tag_lists[cate]:
                tag_map[cate].append(tag)
                break
            if tag.replace("_", " ") in tag_lists[cate]:
                tag_map[cate].append(tag.replace("_", " "))
                break
        else:
            if len(tag) < 4:
                tag_map["general"].append(tag)
            else:
                tag_map["general"].append(tag.replace("_", " "))
    return tag_map


redundant_form: list[tuple[re.Pattern, str]] = [
    [re.compile(r"\n +"), "\n"],
    [re.compile(r"\n\n\n+"), "\n\n"],
    [re.compile(r"  +"), " "],
]


def apply_format(tag_map, form):
    for type in tag_map:
        if f"<|{type}|>" in form:
            if not tag_map[type]:
                form = form.replace(f"<|{type}|>,", "")
                form = form.replace(f"<|{type}|>", "")
            else:
                data = tag_map[type] or ""
                if isinstance(data, list):
                    form = form.replace(f"<|{type}|>", ", ".join(data))
                else:
                    form = form.replace(f"<|{type}|>", data)
    for pattern, repl in redundant_form:
        form = pattern.sub(repl, form)
    return form.strip().strip(",")


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
        if key in content:
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
    general = ", ".join(
        tag_map.get("special", [])
        + tag_map.get("meta", [])
        + tag_map.get("general", [])
    )
    general = general.strip().strip(",")
    meta = {
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
    else:
        operations.append(["short_to_tag_to_long", None, None])
    return meta, operations, general, nl_prompt


if __name__ == "__main__":
    from json import dumps

    print(tag_lists.keys())
    print([len(t) for t in tag_lists.values()])
    print(
        dumps(
            tag_map := seperate_tags(
                [
                    "1girl",
                    "fukuro daizi",
                    "kz oji",
                    "henreader",
                    "ask (askzy)",
                    "aki99",
                    "masterpiece",
                    "newest",
                    "absurdres",
                    "loli",
                    "solo",
                    "dragon girl",
                    "dragon horns",
                    "white dress",
                    "long hair",
                    "side up",
                    "river",
                    "tree",
                    "forest",
                    "pointy ears",
                    ":3",
                    "blue hair",
                    "blush",
                    "breasts",
                    "collarbone",
                    "dress",
                    "eyes visible through hair",
                    "fang",
                    "looking at viewer",
                    "nature",
                    "off shoulder",
                    "open mouth",
                    "orange eyes",
                    "tail",
                    "twintails",
                    "wings",
                ]
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    print()
    print(
        apply_format(
            tag_map,
            """<|special|>, 
<|characters|>, <|copyrights|>, 
<|artist|>, 

<|general|>, 

<|quality|>, <|meta|>, <|rating|>""",
        )
    )
    print()
    print()
    print(apply_dtg_prompt(tag_map, 1.0))
