import re
import random
from time import time
from typing import Callable

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
    "KBlueLeaf/TIPO-200M",
    device="cuda",
)

splitters = [
    lambda x, i: x[i:].split("tags")[-1].split("long:")[0].split("short:")[0].count(",") > 4
]

inputs = models.tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(next(models.text_model.parameters()).device)
input_length = input_ids.shape[-1]
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

all_result = []
for idx in range(3):
    recorder.clean()
    splitter.clean()
    with torch.no_grad():
        generation_output = models.text_model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            max_new_tokens=1024,
            logits_processor=processors,
            stopping_criteria=stop_criteria,
        )

    output_sequence = generation_output.sequences[0][input_length:]
    scores = recorder.scores
    total_score = 0
    for i, (score, choosed) in enumerate(zip(scores[:-1], output_sequence[:-1])):
        score = torch.softmax(score, dim=-1)[0]
        total_score += score[choosed]

    avg_score = total_score / len(output_sequence)
    all_result.append([avg_score, output_sequence, idx])


for result in sorted(all_result, key=lambda x: x[0], reverse=True):
    score, output_sequence, idx = result
    decode = models.tokenizer.decode(output_sequence)
    print(f"{idx:04}: {score}")
    print("-" * 20)
    print(f"{prompt}{decode}")
    print("=" * 50)
