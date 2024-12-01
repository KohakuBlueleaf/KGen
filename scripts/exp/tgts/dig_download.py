import os
import asyncio
from tqdm import tqdm
from httpx import AsyncClient, Limits

# Use DIG to generate images from text distributedly
# https://github.com/KohakuBlueleaf/DIG
import dig_client.config as config

config.SERVER_URL = "http://192.168.1.2:21224"
import dig_client.downloader as downloader
from dig_client.downloader import check_image_status


downloader.client = AsyncClient(timeout=3600, limits=Limits(max_connections=128))
downloader.semaphore = asyncio.Semaphore(128)


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
            tasks.append(check_image_status(f"{category}-{idx}"))

    for batch in tqdm(
        [tasks[i : i + 512] for i in range(0, len(tasks), 512)],
        total=len(tasks) // 512 + int(len(tasks) % 512 > 0),
        desc="Downloading images",
    ):
        await asyncio.gather(*batch)


if __name__ == "__main__":
    asyncio.run(main())
