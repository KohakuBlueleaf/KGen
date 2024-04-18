import time
import pathlib

import kgen.models as models
from kgen.formatter import seperate_tags, apply_format, apply_dtg_prompt
from kgen.metainfo import TARGET
from kgen.generate import tag_gen
from kgen.logging import logger


SEED_MAX = 2**31 - 1
TOTAL_TAG_LENGTH = {
    "VERY_SHORT": "very short",
    "SHORT": "short",
    "LONG": "long",
    "VERY_LONG": "very long",
}
DEFAULT_FORMAT = """<|special|>, 
<|characters|>, <|copyrights|>, 
<|artist|>, 

<|general|>, 

<|quality|>, <|meta|>, <|rating|>"""


def process(
    prompt: str,
    aspect_ratio: float,
    seed: int,
    tag_length: str,
    ban_tags: str,
    format: str,
    temperature: float,
):
    propmt_preview = prompt.replace("\n", " ")[:40]
    logger.info(f"Processing propmt: {propmt_preview}...")
    logger.info(f"Processing with seed: {seed}")
    black_list = [tag.strip() for tag in ban_tags.split(",") if tag.strip()]
    all_tags = [tag.strip() for tag in prompt.strip().split(",") if tag.strip()]

    tag_length = tag_length.replace(" ", "_")
    len_target = TARGET[tag_length]

    tag_map = seperate_tags(all_tags)
    dtg_prompt = apply_dtg_prompt(tag_map, tag_length, aspect_ratio)
    for _, extra_tokens, iter_count in tag_gen(
        models.text_model,
        models.tokenizer,
        dtg_prompt,
        tag_map["special"] + tag_map["general"],
        len_target,
        black_list,
        temperature=temperature,
        top_p=0.95,
        top_k=100,
        max_new_tokens=256,
        max_retry=20,
        max_same_output=15,
        seed=seed % SEED_MAX,
    ):
        pass
    tag_map["general"] += extra_tokens
    prompt_by_dtg = apply_format(tag_map, format)
    logger.info(
        "Prompt processing done. General Tags Count: "
        f"{len(tag_map['general'] + tag_map['special'])}"
        f" | Total iterations: {iter_count}"
    )
    return prompt_by_dtg


if __name__ == "__main__":
    # or whatever path you want to put your model file
    models.model_dir = pathlib.Path(__file__).parent / "models"

    file = models.download_gguf()
    files = models.list_gguf()
    file = files[-1]
    logger.info(f"Use gguf model from local file: {file}")
    models.load_model(file, gguf=True)

    prompt = """
1girl, ask (askzy), masterpiece
"""

    t0 = time.time_ns()
    result = process(
        prompt,
        aspect_ratio=1.0,
        seed=1,
        tag_length=TOTAL_TAG_LENGTH["LONG"],
        ban_tags="",
        format=DEFAULT_FORMAT,
        temperature=1.35,
    )
    t1 = time.time_ns()
    logger.info(f"Result:\n{result}")
    logger.info(f"Time cost: {(t1 - t0) / 10**6:.1f}ms")
