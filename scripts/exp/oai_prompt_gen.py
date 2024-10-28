import json
import asyncio
from contextlib import nullcontext

from tqdm import tqdm, trange
from openai import OpenAI, AsyncOpenAI

client = AsyncOpenAI()


INPUT_FORMAT = "- {prompt}"
OAI_PROMPT_FORMAT = """
Additional info:
{info}

Please follow above additional information (if provided) to generate extended, refined version of the following prompts:
{prompts}

Each prompt should begin with the phrase of each provided prompt and continue with a different, vivid description of a specific scene. Each prompt should be long and detail with different feeling in it.

return all the content in jsonl format without any redundant explanation, directly return jsonl without extra format such as markdown.
Each line of your raw output is one json object, remember to put one empty line between object.
""".strip()


test_prompts = [
    "A beautiful scenery",
    "A girl is sitting in the cafe",
    "Dog running",
    "An anime illustration",
]


async def _fetch(oai_prompt):
    result = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who are good at follow the instruction to provide specific format of content.",
            },
            {"role": "user", "content": oai_prompt},
        ],
        temperature=1.0,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "text"},
    )
    return result


async def _fetch_limit(oai_prompt, limiter):
    while True:
        if limiter:
            await limiter.wait()
        try:
            result = await _fetch(oai_prompt)
            break
        except Exception as e:
            print("Error", e)
            await asyncio.sleep(5)
    return result


async def fetch(
    prompts: list[str], info: str = "", limiter=None, semaphore=None
) -> list[dict]:
    all_prompts = [INPUT_FORMAT.format(prompt=prompt) for prompt in prompts]
    oai_prompt = OAI_PROMPT_FORMAT.format(prompts="\n".join(all_prompts), info=info)
    async with semaphore or nullcontext():
        for retry in range(5):
            result = await _fetch_limit(oai_prompt, limiter)
            results = []
            for i, line in enumerate(result.choices[0].message.content.split("\n")):
                if line.strip() == "":
                    continue
                try:
                    results.append(json.loads(line)["prompt"])
                except:
                    continue
            if len(results) >= len(prompts):
                return results[: len(prompts)]
        return prompts


async def process_all(prompts: list[str], batch_size: int, **kwargs) -> list[dict]:
    results = []
    for i in trange(0, len(prompts), batch_size, **kwargs):
        results.extend(await fetch(prompts[i : i + batch_size]))
    return results


if __name__ == "__main__":
    import asyncio

    test = asyncio.run(fetch(test_prompts))
    # print("\n\n".join(test))
