import asyncio
import torch
import tqdm
from tqdm.asyncio import tqdm_asyncio
from transformers import logging
from orjson import loads, dumps
from asynciolimiter import Limiter

from kgen.utils import remove_repeated_suffix
from oai_prompt_gen import fetch


logging.set_verbosity_error()
print(f"threads: {torch.get_num_threads()} {torch.get_num_interop_threads()}")


limiter = Limiter(50)
semaphore = asyncio.Semaphore(128)

SHORT_KEY = "short_caption"
LONG_KEY = "detail_caption"


async def task(entry, short, detail):
    result1, result2 = await fetch(
        [short, detail], limiter=limiter, semaphore=semaphore
    )
    generated_entry = {
        "failed": short.strip() == result1.strip() or detail.strip() == result2.strip(),
        "entry": entry,
        "result1": result1,
        "result2": result2,
    }
    return generated_entry


async def main():
    with open("./data/gbc.json", "rb") as f:
        data = loads(f.read())

    tasks = []
    for entry in tqdm.tqdm(data[:10000], smoothing=0.001):
        entry.pop("url", None)
        short_caption = remove_repeated_suffix(entry[SHORT_KEY].strip())
        detail_caption = remove_repeated_suffix(
            ".".join(entry[LONG_KEY].strip().split(".")[:2])
        )
        tasks.append(task(entry, short_caption, detail_caption))
    with open("./data/gbc-output-oai.jsonl", "ab") as f:
        for result in await tqdm_asyncio.gather(*tasks, smoothing=0.01):
            f.write(dumps(result) + b"\n")


if __name__ == "__main__":
    asyncio.run(main())
