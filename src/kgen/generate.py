import torch
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)
from . import models

try:
    from llama_cpp import Llama
except ImportError:

    class Llama:
        pass


def generate(
    model: PreTrainedModel | Llama = None,
    tokenizer: PreTrainedTokenizerBase = None,
    prompt="",
    temperature=0.5,
    top_p=0.95,
    top_k=45,
    repetition_penalty=1.17,
    max_new_tokens=128,
    autocast_gen=lambda: torch.autocast("cpu", enabled=False),
    **kwargs,
):
    if model is None:
        model = models.text_model
    if tokenizer is None:
        tokenizer = models.tokenizer
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
        prompt_tokens = result["usage"]["prompt_tokens"]
        completion_tokens = result["usage"]["completion_tokens"]
        return prompt + result["choices"][0]["text"], prompt_tokens, completion_tokens
    if "seed" in kwargs:
        set_seed(kwargs["seed"] % 2**32)

    torch.cuda.empty_cache()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        min_new_tokens=2,
        **kwargs,
    )
    with torch.no_grad(), autocast_gen():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output[0]
    output = tokenizer.decode(s)
    prompt_tokens = len(input_ids[0])
    completion_tokens = len(s) - prompt_tokens

    torch.cuda.empty_cache()
    return output, prompt_tokens, completion_tokens
