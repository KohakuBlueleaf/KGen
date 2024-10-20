import re
import random
from time import time
from typing import Callable
import heapq as hq

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, logging, GenerationConfig
from transformers.generation import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    LogitsProcessor,
    GenerationMode,
    StoppingCriteria,
    StoppingCriteriaList,
)

import kgen.models as models
import kgen.executor.tipo as tipo
from kgen.formatter import seperate_tags, apply_format
from kgen.generate import generate


class LogitsRecorder(LogitsProcessor):
    def __init__(self):
        self.scores = []

    def clean(self):
        self.scores = []

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        self.scores.append(scores.clone())
        return scores


class NodeSplitter(StoppingCriteria):
    def __init__(self, splitters: list[str, Callable], input_length=0):
        self.splitters = splitters
        self.current = 0
        self.input_length = input_length

    def clean(self, input_length=None):
        self.current = 0
        if input_length is not None:
            self.input_length = input_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        current = models.tokenizer.decode(input_ids[0])
        for splitter in self.splitters:
            if splitter(current, self.input_length):
                return True
        return False


meta, operations, general, prompt = tipo.parse_tipo_request(
    seperate_tags("masterpiece, 1girl, dragon girl, safe, absurdres".split(",")),
    "A dragon girl",
)
mode, length, expand = operations[0]
prompt = tipo.apply_tipo_prompt(meta, general, prompt, mode, length, expand)

models.load_model(
    "KBlueLeaf/TIPO-500M",
    device="cuda",
)
print(models.text_model.main_input_name)

splitters = [
    lambda x, i: x[i:].split("tags")[-1].split("long:")[0].split("short:")[0].count(",")
### MOD HERE ###
    > 6
### END MOD HERE ###
]
generation_config = GenerationConfig(
    min_new_tokens=4,
    return_dict_in_generate=True,
    output_scores=True,
    do_sample=True,
)

processors = LogitsProcessorList()
recorder = LogitsRecorder()
processors.append(recorder)

stop_criteria = StoppingCriteriaList()
splitter = NodeSplitter(splitters, input_length=len(prompt))
stop_criteria.append(splitter)


def get_next(prompt, input_ids=None, key_values=None):
    recorder.clean()
    splitter.clean(len(prompt))

    if input_ids is None:
        inputs = models.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(next(models.text_model.parameters()).device)
    input_length = input_ids.shape[-1]
    extra_kwargs = {}
    if key_values is not None:
        extra_kwargs["past_key_values"] = key_values
    with torch.no_grad():
        generation_output = models.text_model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            max_new_tokens=1024,
            logits_processor=processors,
            stopping_criteria=stop_criteria,
            **extra_kwargs,
        )
    output_sequence = generation_output.sequences

### MOD HERE ###
    scores = recorder.scores
    total_score = 1
    total = 0
    for i, (score, choosed) in enumerate(
        zip(scores[:-1], output_sequence[0][input_length:])
    ):
        if choosed == output_sequence[0][-1]:
            continue
        score = torch.softmax(score, dim=-1)[0]
        total_score *= score[choosed]
        total += 1

    avg_score = total_score / total
    # print(avg_score)
### END MOD HERE ###
    
    return (
        output_sequence,
        generation_output.past_key_values,
        models.tokenizer.decode(output_sequence[0]),
        avg_score,
    )

### MOD HERE ###
def get_variants(prompt, target_variants=5):
    queue = [(0, 0, prompt, None, None)]
    results = []
    total_forward = 0
    while len(results) < target_variants:
        if len(queue) == 0:
            break
        score, level, prompt, input_ids, key_values = hq.heappop(queue)
        print(score, level)
        next_level = level - 1
        score = score * level / next_level
        for _ in range(max(2, 4+level)):
            output_sequence, past_key_values, decode, next_score = get_next(
                prompt, input_ids, key_values
            )
            total_forward += 1
            if output_sequence[0][-1] == models.tokenizer.eos_token_id:
                results.append(
                    (
                        score + next_score / next_level,
                        # score - next_score,
                        next_level,
                        decode,
                    )
                )
                continue
            next_q = (
                score + next_score / next_level,
                # score - next_score,
                next_level,
                decode,
                output_sequence,
                past_key_values,
            )
            hq.heappush(queue, next_q)
    print(total_forward)
    return results
### END MOD HERE ###

results = (
    get_variants(prompt, target_variants=7)
    # + get_variants(prompt, target_variants=3)
    # + get_variants(prompt, target_variants=3)
)


for score, level, result in sorted(results, key=lambda x: x[0] / x[1], reverse=False):
    print(f"{score/level}")
    print("-" * 20)
    print(f"{result}")
    print("=" * 50)
