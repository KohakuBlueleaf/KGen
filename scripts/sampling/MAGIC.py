from typing import Optional

import numpy as np
import torch

import kgen.models as models
import kgen.executor.tipo as tipo
from kgen.formatter import seperate_tags, apply_format
from kgen.generate import generate
from kgen.sampling import SampleNode, LogitsRecorder, NodeSplitter, get_next, draw_tree
from kgen.sampling.node_splitters import tag_splitter


class MCTSNode(SampleNode):
    parent: "MCTSNode"
    childs: list["MCTSNode"]
    
    def __init__(
        self, prompt, inputs=None, past_key_values=None, score=0, parent=None
    ):
        super().__init__(prompt, inputs, past_key_values, score, parent)
        self.active = False
        self.spent = False
        self.visits = 0
        self.simulated_result: Optional[str] = None
        
    def uct1(self, exploration_weight=1.4):
        if self.visits == 0:
            return float("inf")
        
        return self.score / self.visits + exploration_weight * np.sqrt(
            np.log(self.parent.visits) / self.visits
        )
        
    def select(self, exploration=1.4) -> "MCTSNode":
        """
        greedy search for leaf node with max uct1
        stop until no children or active children
        
        ---
        should only ever be called by root in this implementation
        """
        node = self
        while node.childs and any(child.active for child in node.childs):
            active_childs = [c for c in node.childs if c.active and not c.spent]
            
            if not active_childs:
                # NOTE: this part is completely baseless, just last ditch effort to make sure it doesn't get stuck
                # it's basically the shortcoming of greedy method
                print(f'If you\'re reading this, this method sucks') 
                node.spent = True
                node._backpropagate(0)
                node = node.parent
            
            node = max(active_childs, key=lambda c: c.uct1(exploration_weight=exploration))
            
        return node
        
    def expand(self, splitters=None, ids_splitters=None) -> tuple["MCTSNode", int] | tuple[None, int]:
        """
        create childs for this node
        convert inactive childs to active for current node if any exist
                       
        ---
        if created child is terminal, return child with generated length
        """
        splitter = NodeSplitter(
            splitters=splitters,
            ids_splitters=ids_splitters,
            input_length=len(self.prompt),
        )
        recorder = LogitsRecorder()
        
        # progressive widening
        k = max(2, 4-self.depth)
        alpha = 0.5 ** (1 + self.depth)
        num_childs = np.ceil(k * self.visits ** alpha)
            
        total_gen = 0
        while len(self.childs) < num_childs:
            out_seq, past_key_values, decode, score, inp_len, final_len = get_next(
                self.prompt,
                input_ids=self.inputs,
                key_values=self.past_key_values,
                recorder=recorder,
                splitter=splitter,
                # scoring='log' # NOTE: don't use log, it's worse, that or my impl.ed it wrong
            )
            total_gen += final_len - inp_len
            
            child = MCTSNode(
                prompt=decode,
                inputs=out_seq,
                past_key_values=past_key_values,
                score=score,
                parent=self,
            )
            self.childs.append(child)
            
            if child.is_leaf:
                child._backpropagate(score)
                child.simulated_result = (
                    prompt.replace("<s>", "").replace("</s>", "").strip(),
                    total_gen
                )
                return child, total_gen
        
        for c in self.childs:
            c.active = True
                
        return None, total_gen
        
        
    def simulate(self, splitters=None, ids_splitters=None) -> tuple["MCTSNode", int]:
        """
        simulate from this node until reaching terminal
        nodes are created along simulation path but remain inactive until expansion
        
        ---
        return child with generated length
        """
        splitter = NodeSplitter(
            splitters=splitters,
            ids_splitters=ids_splitters,
            input_length=len(self.prompt),
        )
        recorder = LogitsRecorder()
        
        current_node = self
        total_gen = 0
        while True:
            out_seq, past_key_values, decode, score, inp_len, final_len = get_next(
                current_node.prompt,
                input_ids=current_node.inputs,
                key_values=current_node.past_key_values,
                recorder=recorder,
                splitter=splitter,
                # scoring='log'
            )
            total_gen += final_len - inp_len
            
            child = MCTSNode(
                prompt=decode,
                inputs=out_seq,
                past_key_values=past_key_values,
                score=score,
                parent=current_node,
            )
            current_node.childs.append(child)
            
            child._backpropagate(score)
            if child.is_leaf:
                child.simulated_result = (
                    prompt.replace("<s>", "").replace("</s>", "").strip(),
                    total_gen,
                )
                return child, total_gen
            
            current_node = child
        
    def _backpropagate(self, score) -> None:
        """
        update from this node upward
        """
        node = self
        while node is not None:
            node.visits += 1
            node.score += score 
            node = node.parent
        

def MAGIC_sample(
    prompt: str,
    splitters=None,
    ids_splitters=None,
    variations: int = 7,
    exploration=1.0,
):
    results = []
    root = MCTSNode(prompt=prompt)
    total_iterations = 0
    total_gen = 0

    while len(results) < variations:
        # Selection
        node = root.select(exploration=exploration)

        # Expansion / Simulation (Backpropagation included)
        # NOTE: ideally, we don't separate the functions
        # but for clarity's sake, let's keep it this way
        if node.visits == 0:
            node, gen = node.simulate(
                splitters,
                ids_splitters
            )
            # NOTE: put here because we have multiple select/expand per iteration
            if total_iterations % 10 == 0:
                print(f'iteration: {total_iterations} - results: {len(results)}')
            total_iterations += 1 
        else:
            node, gen = node.expand(
                splitters,
                ids_splitters
            )
        
        total_gen += gen
        
        
        # write result (if node is not None, has to be terminal)
        if node: 
            node.spent = True
            results.append(node.simulated_result)
            print("Add result")


    print(f"Total iterations: {total_iterations}")
    print(f"Total output tokens: {total_gen}")
    return results, root


def _count(node: MCTSNode, depth: int = 0, total_childs=None, total_nodes=None):
    if node.is_leaf:
        return
    if depth not in total_childs:
        total_childs[depth] = 0
        total_nodes[depth] = 0
    total_childs[depth] += len(node.childs)
    total_nodes[depth] += 1
    for child in node.childs:
        _count(child, depth + 1, total_childs, total_nodes)


def count(node: MCTSNode):
    total_childs = {}
    total_nodes = {}
    _count(node, total_childs=total_childs, total_nodes=total_nodes)
    return total_childs, total_nodes


if __name__ == "__main__":
    models.load_model(
        "KBlueLeaf/TIPO-100M",
        device="cuda",
    )

    meta, operations, general, prompt = tipo.parse_tipo_request(
        seperate_tags("1girl, fox girl, fox ears, multiple tails".split(",")),
        "",
    )
    mode, length, expand = operations[0]
    prompt = tipo.apply_tipo_prompt(meta, general, prompt, mode, length, expand)

    results, root = MAGIC_sample(
        prompt,
        ids_splitters=[lambda ids, i: torch.sum(ids[0, i:] == 29892) >= 4],
        variations=1024,
        exploration=1.4,
    )

    dot = draw_tree(root)
    dot.attr(dpi="300")
    dot.render("tree16", cleanup=True, format="png")
    total_childs, total_nodes = count(root)

    print(f"Total nodes per depth: {total_nodes}")
    print(
        "Average childs per node per depth: "
        f"{[total_childs[i] / total_nodes[i] for i in range(len(total_childs))]}"
    )
    gen_per_prompt = [x[1] for x in results]
    print(sum(gen_per_prompt) / len(gen_per_prompt))
