import os
import re
import pathlib

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
    if "<|extended|>" in form and not tag_map.get("extended", ""):
        form = form.replace("<|extended|>", "<|generated|>")
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
    form = re.sub(r"<\|(?:(?!<\|.*\|>).)*\|>", "", form)
    for pattern, repl in redundant_form:
        form = pattern.sub(repl, form)
    return form.strip().strip(",")
