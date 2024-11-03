import re
import random
import heapq as hq
from time import time
from typing import Callable
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, logging, GenerationConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import (
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    MinPLogitsWarper,
)
from graphviz import Digraph

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


def get_next(
    prompt,
    input_ids=None,
    key_values=None,
    recorder: LogitsRecorder = None,
    splitter: NodeSplitter = None,
    gen_kwargs={},
):
    if input_ids is None:
        inputs = models.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(next(models.text_model.parameters()).device)
    input_length = input_ids.shape[-1]
    extra_kwargs = {}
    if key_values is not None:
        extra_kwargs["past_key_values"] = key_values

    processors = LogitsProcessorList()
    if recorder:
        recorder.clean()
        processors.append(recorder)
    processors.append(TemperatureLogitsWarper(1.5))
    processors.append(MinPLogitsWarper(0.1))
    # processors.append(TopPLogitsWarper(0.95))
    # processors.append(TopKLogitsWarper(60))

    stop_criteria = StoppingCriteriaList()
    if splitter:
        splitter.clean(len(prompt))
        stop_criteria.append(splitter)

    gen_kwargs["min_new_tokens"] = 4
    gen_kwargs["return_dict_in_generate"] = True
    gen_kwargs["output_scores"] = True
    gen_kwargs["do_sample"] = True
    generation_config = GenerationConfig(max_length=1024, **gen_kwargs)
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

    if recorder is not None:
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
        avg_score = (total_score ** (1 / total)).item()
        print(avg_score)
    else:
        avg_score = 0

    return (
        output_sequence,
        generation_output.past_key_values,
        models.tokenizer.decode(output_sequence[0]),
        avg_score,
    )


class SampleNode:
    def __init__(
        self, prompt=None, inputs=None, past_key_values=None, score=0, parent=None
    ):
        self.prompt: str = prompt
        self._inputs: torch.Tensor = inputs
        self._past_key_values: tuple[tuple[torch.FloatTensor]] = past_key_values
        self.score: float = score

        self.parent: SampleNode = parent
        self.childs: list[SampleNode] = []

        self.have_leaf: bool = False
        if inputs is not None:
            self.is_leaf: bool = self.inputs[0][-1] == models.tokenizer.eos_token_id
        else:
            self.is_leaf: bool = False
        self.is_leaf = bool(self.is_leaf)

    @property
    def inputs(self):
        if self._inputs is None:
            return None
        return self._inputs.clone()

    @property
    def past_key_values(self):
        if self._past_key_values is None:
            return None
        new_cache = []
        for past_key_value in self._past_key_values:
            new_cache.append((past_key_value[0].clone(), past_key_value[1].clone()))
        return tuple(new_cache)

    def gen_new_child(self):
        recorder = LogitsRecorder()

        splitters = [lambda x, i: (x[i:].split("tags")[-1].count(",") > 6)]
        splitter = NodeSplitter(splitters, input_length=len(prompt))
        out_seq, past_key_values, decode, score = get_next(
            self.prompt,
            input_ids=self.inputs,
            key_values=self.past_key_values,
            recorder=recorder,
            splitter=splitter,
        )
        new_child = SampleNode(
            prompt=decode,
            inputs=out_seq,
            past_key_values=past_key_values,
            score=score,
        )
        self.childs.append(new_child)
        if new_child.is_leaf:
            now = self
            while now.parent is not None:
                now.parent.have_leaf = True
                now = now.parent
        return new_child


def greedy_tree_sample(prompt, variations=7):
    total_gen = 0
    root = SampleNode(prompt=prompt)
    for _ in range(variations):
        root.gen_new_child()
        total_gen += 1

    results = []
    for child in root.childs:
        if child.is_leaf:
            results.append(child.prompt)
    while len(results) < variations:
        now = root
        while now.childs:
            next = max(now.childs, key=lambda x: x.score if not x.is_leaf else 0)
            if next.is_leaf:
                break
            now = next
        now = new_child = now.gen_new_child()
        while now.parent:
            now.parent.score = min(now.parent.score, new_child.score)
            now = now.parent
        total_gen += 1
        if new_child.is_leaf:
            results.append(new_child.prompt)
    print(total_gen)
    return results


def conventional_sample(prompt, variations=7):
    recorder = LogitsRecorder()

    splitters = [lambda x, i: (x[i:].split("tags")[-1].count(",") > 4)]
    splitter = NodeSplitter(splitters, input_length=len(prompt))

    total_gen = variations
    datas = []
    for _ in range(variations):
        out_seq, past_key_values, decode, score = get_next(
            prompt,
            input_ids=None,
            key_values=None,
            recorder=recorder,
            splitter=splitter,
        )
        datas.append((out_seq, past_key_values, decode, score))

    results = []
    for out_seq, past_key_values, decode, score in datas:
        is_leaf = bool(out_seq[0][-1] == models.tokenizer.eos_token_id)
        while not is_leaf:
            total_gen += 1
            out_seq, past_key_values, decode, score = get_next(
                decode,
                input_ids=out_seq,
                key_values=past_key_values,
                recorder=recorder,
                splitter=splitter,
            )
            is_leaf = bool(out_seq[0][-1] == models.tokenizer.eos_token_id)
        results.append(decode)
    print("Total generation:", total_gen)

    return results


# Function to draw the tree
def draw_tree(node: SampleNode):
    idx = 0

    def assign_idx(node: SampleNode):
        nonlocal idx
        node.idx = idx
        idx += 1
        for child in node.childs:
            assign_idx(child)

    assign_idx(node)
    dot = Digraph()

    def add_nodes_edges(node: SampleNode):
        for child in node.childs:
            dot.node(str(child.idx))
            dot.edge(str(node.idx), str(child.idx))
            add_nodes_edges(child)

    dot.node(str(node.idx))  # Add root node
    add_nodes_edges(node)
    return dot


if __name__ == "__main__":
    meta, operations, general, prompt = tipo.parse_tipo_request(
        seperate_tags("1girl, fox girl, fox ears, multiple tails".split(",")),
        "",
    )
    mode, length, expand = operations[0]
    prompt = tipo.apply_tipo_prompt(meta, general, prompt, mode, length, expand)

    models.load_model(
        "KBlueLeaf/TIPO-500M",
        device="cuda",
    )
    # results = greedy_tree_sample(prompt)
    results = conventional_sample(prompt, 128)
    # for result in sorted(results):
    #     print("=" * 20)
    #     print(result)
    # print("=" * 20)
