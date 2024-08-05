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


def apply_format(tag_map, form):
    for type in tag_map:
        if f"<|{type}|>" in form:
            if not tag_map[type]:
                form = form.replace(f"<|{type}|>,", "")
                form = form.replace(f"<|{type}|>", "")
            else:
                form = form.replace(f"<|{type}|>", ", ".join(tag_map[type]))
    while "\n " in form:
        form = form.replace("\n ", "\n")
    while "\n\n\n" in form:
        form = form.replace("\n\n\n", "\n\n")
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
                    result_dict[key].extend(value)
                elif not value:
                    result_dict[key] = []
                else:
                    result_dict[key] = value
        else:
            if not isinstance(content, list):
                content = [content.strip()]
            if content[0] == "<|empty|>" or not content[0]:
                continue
            result_dict[TYPE_MAP.get(type, type)] = content
    return result_dict


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
