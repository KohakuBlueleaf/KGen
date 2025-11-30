import asyncio
import aiofiles
import torch
import tqdm
from tqdm.asyncio import tqdm_asyncio
from transformers import logging
from orjson import loads, dumps
from asynciolimiter import Limiter

from kgen.utils import remove_repeated_suffix
from kgen.formatter import seperate_tags, apply_format
from oai_prompt_gen import fetch


logging.set_verbosity_error()
print(f"threads: {torch.get_num_threads()} {torch.get_num_interop_threads()}")


limiter = Limiter(50)
semaphore = asyncio.Semaphore(64)


DEFAULT_PROMPT = [
    "A beautiful scenery",
    "Scenery of",
    "An illustration of a scenery",
    "Scenery",
]
DEFAULT_FORMAT = """<|special|>,
<|characters|>, <|copyrights|>,
<|artist|>,
<|general|>,
<|generated|>.
<|quality|>, <|meta|>, <|rating|>
"""


async def task(entry):
    tags = seperate_tags(entry["caption"].split(","))
    tags["general"] = ["scenery"]
    all_tags = (
        tags["artist"][:5]
        + tags["characters"][:5]
        + tags["copyrights"][:5]
        + tags["meta"][:10]
        + tags["quality"]
        + tags["rating"]
        + tags["special"]
        + tags["general"]
    )
    info_str = f"danbooru tags: {', '.join(all_tags)}"
    results = await fetch(["A beautiful scenery of"], info_str, limiter)

    tags["generated"] = results[0]
    tags["artist"] = tags["artist"][:5]
    tags["characters"] = tags["characters"][:5]
    tags["copyrights"] = tags["copyrights"][:5]
    tags["meta"] = tags["meta"][:10]

    generated_entry = {
        "failed": results[0] == "A beautiful scenery of",
        "entry": entry,
        "result": apply_format(tags, DEFAULT_FORMAT),
    }
    return generated_entry


async def main(data):
    tasks = []
    for entry in tqdm.tqdm(data[:32768], smoothing=0.001):
        tasks.append(task(entry["entry"]))
    async with aiofiles.open("./data/scenery-output-oai.jsonl", "ab") as f:
        for result in await tqdm_asyncio.gather(*tasks, smoothing=0.01):
            await f.write(dumps(result) + b"\n")


if __name__ == "__main__":
    with open("./data/scenery-output.jsonl", "rb") as f:
        data = [loads(line) for line in f.readlines()]

    asyncio.run(main(data))
