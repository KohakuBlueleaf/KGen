import random
from typing import Iterable


def shuffle_iterable(x: Iterable):
    x_cls = type(x)
    x_list = list(x)
    random.shuffle(x_list)
    return x_cls(x_list)


def same_order_deduplicate(x: list):
    new_list = []
    history = set()
    for i in x:
        if i not in history:
            new_list.append(i)
            history.add(i)
    return new_list
