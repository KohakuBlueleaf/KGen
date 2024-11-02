from typing import Optional

import numpy as np

import kgen.models as models
import kgen.executor.tipo as tipo
from kgen.formatter import seperate_tags, apply_format
from kgen.generate import generate
from kgen.sampling import SampleNode, LogitsRecorder, NodeSplitter, get_next


class MCTSNode(SampleNode):
    childs: list["MCTSNode"]
    def __init__(
        self, prompt=None, inputs=None, past_key_values=None, score=0, parent=None
    ):
        super().__init__(prompt, inputs, past_key_values, score, parent)
        # MCTS specific properties
        self.visit_count = 0
        self.total_value = 0
        self.self_visit_count = 0
        self.self_total_value = 0
        self.simulated_result: Optional[str] = None

    def get_uct_score(self, exploration_weight=1.0, self_uct=False):
        if self.visit_count == 0:
            return float("inf")

        if self_uct:
            exploitation = self.self_total_value / self.self_visit_count
            exploration = exploration_weight * np.sqrt(
                np.log(self.visit_count) / self.self_visit_count
            )
        else:
            exploitation = self.total_value / self.visit_count
            exploration = exploration_weight * np.sqrt(
                np.log(self.parent.visit_count) / self.visit_count
            )
        return exploitation + exploration

    def select_child(self, exploration_weight=1) -> "MCTSNode":
        if all(c.is_leaf for c in self.childs):
            return self
        # If there are unvisited children, select one randomly
        # unvisited = [c for c in self.childs if c.visit_count == 0]
        # if unvisited:
        #     return random.choice(unvisited)

        # Calculate scores for visited children
        scores = np.array(
            [child.get_uct_score(exploration_weight) for child in self.childs]
            + [self.get_uct_score(exploration_weight=0.5, self_uct=True)]
        )
        scores = np.exp(scores - np.max(scores))  # Softmax normalization
        scores = scores / scores.sum()

        # Select child based on softmax probabilities
        chosen_idx = np.random.choice(len(self.childs) + 1, p=scores)
        return (self.childs + [self])[chosen_idx]

    def expand(self) -> "MCTSNode":
        print("Expand")
        # Generate new child
        recorder = LogitsRecorder()
        splitters = [lambda x, i: (x[i:].split("tags")[-1].count(",") > 6)]
        splitter = NodeSplitter(splitters, input_length=len(self.prompt))

        # Get the immediate next tokens
        total_gen = 1
        out_seq, past_key_values, decode, score = get_next(
            self.prompt,
            input_ids=self.inputs,
            key_values=self.past_key_values,
            recorder=recorder,
            splitter=splitter,
        )

        # Create new child node
        new_child = MCTSNode(
            prompt=decode,
            inputs=out_seq,
            past_key_values=past_key_values,
            score=score,
            parent=self,
        )
        self.childs.append(new_child)
        self.self_visit_count += 1
        self.self_total_value += new_child.score

        # Simulate a complete path from this child
        is_leaf = new_child.is_leaf
        prompt = new_child.prompt
        inputs = new_child.inputs
        past_key_values = new_child.past_key_values
        while not is_leaf:
            total_gen += 1
            inputs, past_key_values, prompt, score = get_next(
                prompt,
                input_ids=inputs,
                key_values=past_key_values,
                recorder=recorder,
                splitter=splitter,
            )
            is_leaf = bool(inputs[0][-1] == models.tokenizer.eos_token_id)
        print("Simulate end")

        new_child.simulated_result = prompt

        return new_child, total_gen

    def backpropagate(self, value: float):
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent


def mcts_tree_sample(prompt: str, variations: int = 7, num_iterations: int = 100):
    root = MCTSNode(prompt=prompt)
    results = []
    total_iterations = 0
    total_gen = 0

    while len(results) < variations and total_iterations < num_iterations:
        # Selection
        node = root
        depth = 0
        while node.childs:
            depth += 1
            next = node.select_child(exploration_weight=4/depth)
            if next is node:
                depth -= 1
                break
            node = next
        print(f"Select depth: {depth}")

        # Expansion
        if not node.is_leaf:
            new_node, gen = node.expand()
            total_gen += gen
            print(new_node.simulated_result is None)

            # Evaluation using the simulated result
            value = new_node.score

            # Backpropagation
            new_node.backpropagate(value)

            # If we got a complete generation, add it to results
            if new_node.is_leaf or new_node.simulated_result is not None:
                print("Add result")
                results.append(new_node.simulated_result)

        total_iterations += 1

    print(f"Total iterations: {total_iterations}")
    print(f"Total generations: {total_gen}")
    return results


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
    results = mcts_tree_sample(prompt, variations=32, num_iterations=1000000)
    # for result in sorted(results):
    #     print("=" * 20)
    #     print(result)
    # print("=" * 20)
