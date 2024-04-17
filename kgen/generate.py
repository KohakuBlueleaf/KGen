import regex as re
import random
from contextlib import nullcontext
from random import shuffle

import torch

try:
    from llama_cpp import Llama
except ImportError:

    class Llama:
        pass


from transformers import (
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)


def generate(
    model: PreTrainedModel | Llama,
    tokenizer: PreTrainedTokenizerBase,
    prompt="",
    temperature=0.5,
    top_p=0.95,
    top_k=45,
    repetition_penalty=1.17,
    max_new_tokens=128,
    autocast_gen=lambda: torch.autocast("cpu", enabled=False),
    **kwargs,
):
    if isinstance(model, Llama):
        # print(kwargs)
        result = model.create_completion(
            prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            repeat_penalty=repetition_penalty or 1,
            seed=kwargs.get("seed", None),
        )
        # print(prompt, "===", result["choices"][0]["text"])
        return prompt + result["choices"][0]["text"]
    if "seed" in kwargs:
        set_seed(kwargs["seed"])

    torch.cuda.empty_cache()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        **kwargs,
    )
    with torch.no_grad(), autocast_gen():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)

    torch.cuda.empty_cache()
    return output


def black_list_match(tag, black_list):
    for b in black_list:
        if re.match(b, tag):
            return True
    return False


def tag_gen(
    text_model,
    tokenizer,
    prompt,
    prompt_tags,
    len_target,
    black_list,
    temperature=0.5,
    top_p=0.95,
    top_k=100,
    max_new_tokens=256,
    max_retry=20,
    max_same_output=15,
    seed=None,
):
    retry = max_retry
    llm_gen = ""

    set_seed(seed)

    iter_count = 0
    prev_output = set()
    same_output_count = 0
    while retry >= 0 and same_output_count < max_same_output:
        llm_gen = generate(
            model=text_model,
            tokenizer=tokenizer,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=None,
            max_new_tokens=max_new_tokens,
            stream_output=False,
            autocast_gen=nullcontext,
            prompt_lookup_num_tokens=10,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            seed=seed + iter_count,
        )
        iter_count += 1
        llm_gen = llm_gen.replace("</s>", "").replace("<s>", "")
        orig_prompt = llm_gen.split("<|input_end|>")[0]
        extra = llm_gen.split("<|input_end|>")[-1].strip().strip(",")
        extra_tokens = list(
            [
                tok.strip()
                for tok in extra.split(",")
                if not black_list_match(tok.strip(), black_list)
            ]
        )
        llm_gen = llm_gen.replace(extra, ", ".join(extra_tokens))

        yield llm_gen, extra_tokens, iter_count

        if set(extra_tokens) == prev_output:
            same_output_count += 1
            retry += 1
        else:
            same_output_count = 0
            prev_output = set(extra_tokens)

        if len(prompt_tags) + len(extra_tokens) < len_target:
            # print(f"retry: {retry}, same_output_count: {same_output_count}, len: {len(extra_tokens)}")
            retry -= 1
            shuffle(extra_tokens)
            llm_gen = f"{orig_prompt}<|input_end|>{', '.join(extra_tokens)}"
            prompt = llm_gen.strip().replace("  <|", " <|")
        else:
            break
    yield llm_gen, extra_tokens, iter_count
