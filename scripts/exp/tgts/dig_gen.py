import asyncio
import os
from tqdm import tqdm
from httpx import AsyncClient, Limits

# Use DIG to generate images from text distributedly
# https://github.com/KohakuBlueleaf/DIG
import dig_client.config as config

config.SERVER_URL = "http://192.168.1.2:21224"
import dig_client.requestor as requestor
from dig_client.requestor import request_image_generation


requestor.client = AsyncClient(timeout=3600, limits=Limits(max_connections=128))
requestor.semaphore = asyncio.Semaphore(128)


def load_propmts(file):
    with open(file, "r", encoding="utf-8") as f:
        return list(enumerate(f.readlines()))


async def main():
    tasks = []
    prompt_files = os.listdir("./test")
    for prompt_file in prompt_files:
        if "cg-mcts-new" not in prompt_file:
            continue
        category = os.path.basename(prompt_file).split(".")[0]
        propmts = load_propmts(f"./test/{prompt_file}")
        for idx, prompt in propmts:
            prompt = prompt.strip().replace(", ,", ",")
            tasks.append(
                request_image_generation(prompt, f"{category}-{idx}", int(idx))
            )

    for batch in tqdm(
        [tasks[i : i + 512] for i in range(0, len(tasks), 512)],
        total=len(tasks) // 512 + int(len(tasks) % 512 > 0),
        desc="Requesting images",
    ):
        await asyncio.gather(*batch)


if __name__ == "__main__":
    asyncio.run(main())
