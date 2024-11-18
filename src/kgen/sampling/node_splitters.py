from typing import Callable

import torch
from transformers.generation import StoppingCriteria

import kgen.models as models


class NodeSplitter(StoppingCriteria):
    def __init__(
        self,
        splitters: list[str, Callable] = None,
        ids_splitters: list[str, Callable] = None,
        input_length=0,
    ):
        self.splitters = splitters
        self.ids_splitters = ids_splitters
        self.input_length = 0
        self.input_id_length = 0

    def clean(self):
        self.input_id_length = self.input_length = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        if self.splitters:
            current = models.tokenizer.decode(input_ids[0])
            if self.input_length == 0:
                self.input_length = len(current)
            for splitter in self.splitters:
                if splitter(current, self.input_length):
                    return True
        if self.ids_splitters:
            if self.input_id_length == 0:
                self.input_id_length = len(input_ids[0])
            for splitter in self.ids_splitters:
                if splitter(input_ids, self.input_id_length):
                    return True
        return False


def tag_splitter(start="tags: ", sep=", ", end="\n", tag_count=1):
    def splitter(text, length):
        examine_part = text[length:].split(start, 1)[1].split(end, 1)[0]
        tags = examine_part.count(sep)
        return tags >= tag_count

    return splitter
