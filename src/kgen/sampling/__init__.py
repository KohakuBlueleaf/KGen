import math

import torch
from transformers import GenerationConfig
from transformers.generation import (
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteriaList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    MinPLogitsWarper,
)
from tqdm import tqdm
from graphviz import Digraph

import kgen.models as models
import kgen.executor.tipo as tipo
from kgen.formatter import seperate_tags, apply_format
from kgen.sampling.node_splitters import NodeSplitter


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


class LengthRecorder(LogitsProcessor):
    def __init__(self):
        self.inp_lengths = -1
        self.final_lengths = -1

    def clean(self):
        self.inp_lengths = -1
        self.final_lengths = -1

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if self.inp_lengths == -1:
            self.inp_lengths = cur_len
        self.final_lengths = cur_len + 1
        return scores


def get_next(
    prompt,
    input_ids=None,
    key_values=None,
    recorder: LogitsRecorder = None,
    splitter: NodeSplitter = None,
    gen_kwargs={},
    scoring="default",
    single_token=False,
):
    if input_ids is None:
        inputs = models.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(next(models.text_model.parameters()).device)
    input_length = input_ids.shape[-1]
    extra_kwargs = {}
    if key_values is not None:
        extra_kwargs["past_key_values"] = key_values

    length_recorder = LengthRecorder()
    processors = LogitsProcessorList()
    processors.append(length_recorder)
    if recorder:
        recorder.clean()
        processors.append(recorder)
    processors.append(TemperatureLogitsWarper(1.0))
    processors.append(MinPLogitsWarper(0.1))
    # processors.append(TopPLogitsWarper(0.95))
    # processors.append(TopKLogitsWarper(60))

    stop_criteria = StoppingCriteriaList()
    if splitter:
        splitter.clean()
        stop_criteria.append(splitter)

    gen_kwargs["min_new_tokens"] = 1 if single_token else 4
    gen_kwargs["max_new_tokens"] = 1 if single_token else 1024
    gen_kwargs["return_dict_in_generate"] = True
    gen_kwargs["output_scores"] = True
    gen_kwargs["do_sample"] = True
    generation_config = GenerationConfig(max_length=1024, **gen_kwargs)
    with torch.no_grad():
        generation_output = models.text_model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            logits_processor=processors,
            stopping_criteria=stop_criteria,
            **extra_kwargs,
        )
    output_sequence = generation_output.sequences

    if recorder is not None:
        scores = recorder.scores
        min_score = 1
        max_score = 0
        total_score = 1
        total = 0
        for i, (score, choosed) in enumerate(
            zip(scores, output_sequence[0][input_length:])
        ):
            # seperator usually has a very high score, so we skip it
            if not single_token and choosed == output_sequence[0][-1]:
                continue
            if not single_token and choosed == models.tokenizer.eos_token_id:
                break
            score = torch.softmax(score, dim=-1)[0]
            min_score = min(min_score, score[choosed].item())
            max_score = max(max_score, score[choosed].item())
            total_score *= score[choosed]
            total += 1
        avg_score = (total_score ** (1 / total)).item()
    else:
        min_score = 0
        max_score = 0
        avg_score = 0

    return (
        output_sequence,
        generation_output.past_key_values,
        models.tokenizer.decode(output_sequence[0]),
        # avg_score,
        (min_score + max_score + avg_score) / 3,
        length_recorder.inp_lengths,
        length_recorder.final_lengths,
    )


def clone_kv(past_key_values):
    if past_key_values is None:
        return None
    return tuple(tuple(kv.clone() for kv in layer) for layer in past_key_values)


def move_kv(past_key_values, device="cpu"):
    if past_key_values is None:
        return None
    return tuple(tuple(kv.to(device) for kv in layer) for layer in past_key_values)


class SampleNode:
    def __init__(
        self, prompt=None, inputs=None, past_key_values=None, score=0, parent=None
    ):
        self.prompt: str = prompt.replace("<s>", "").replace("</s>", "").strip()
        self._inputs: torch.Tensor = inputs
        self._past_key_values_device = inputs.device if inputs is not None else "cpu"
        self._past_key_values: tuple[tuple[torch.FloatTensor]] = move_kv(past_key_values)
        self.score: float = score

        self.depth = 0 if parent is None else parent.depth + 1
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
        return move_kv(self._past_key_values, self._past_key_values_device)

    def gen_new_child(self, splitter=None, ids_splitter=None):
        recorder = LogitsRecorder()

        splitter = NodeSplitter(splitter, ids_splitter, input_length=len(prompt))
        out_seq, past_key_values, decode, score, inp_len, final_len = get_next(
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
    splitters = [lambda x, i: (x[i:].split("tags")[-1].count(",") > 4)]
    splitter = NodeSplitter(splitters, input_length=len(prompt))
    pbar = tqdm(total=variations)
    total_gen = 0
    root = SampleNode(prompt=prompt)
    for _ in range(variations):
        root.gen_new_child(splitter=splitter)
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
        now = new_child = now.gen_new_child(splitter=splitter)
        while now.parent:
            now.parent.score = min(now.parent.score, new_child.score)
            now = now.parent
        total_gen += 1
        if new_child.is_leaf:
            pbar.update(1)
            results.append(new_child.prompt)
    print(total_gen)
    return results


def conventional_sample(prompt, variations=7):
    total_gen = 0
    results = []
    for _ in range(variations):
        out_seq, past_key_values, decode, score, inp_len, final_len = get_next(
            prompt,
            input_ids=None,
            key_values=None,
        )
        total_gen += final_len - inp_len
        results.append((decode, score))
    print("Total output tokens:", total_gen)

    return results


def beam_search_sample(prompt, variations=7):
    # recorder
    total_gen = 0

    # Initialize with first tokens
    inputs = models.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(models.text_model.parameters()).device)
    input_length = input_ids.shape[-1]

    beam_width = min(variations * 3, 32)  # Wider beam for more diversity
    active_beams = [(input_ids, None, 1.0)]  # (input_ids, past_kv, cumulative_score)
    completed_sequences = []

    # Setup logits processors for scoring and diversity
    recorder = LogitsRecorder()
    processors = LogitsProcessorList()
    processors.append(recorder)

    while len(completed_sequences) < variations:
        next_beams = []

        # Expand each active beam
        for beam_ids, beam_past_kv, beam_score in active_beams:
            # Generate next token distributions
            recorder.clean()
            gen_kwargs = {}
            gen_kwargs["max_new_tokens"] = 1
            gen_kwargs["return_dict_in_generate"] = True
            gen_kwargs["output_scores"] = True
            gen_kwargs["do_sample"] = True
            generation_config = GenerationConfig(**gen_kwargs)
            with torch.no_grad():
                generation_output = models.text_model.generate(
                    input_ids=beam_ids.clone(),
                    generation_config=generation_config,
                    logits_processor=processors,
                    past_key_values=beam_past_kv,
                )
            total_gen += 1

            # Sample multiple candidates from the distribution
            logits = recorder.scores[-1][0]  # Get logits for last token
            probs = torch.softmax(logits, dim=-1)

            # Sample without replacement for diversity
            num_samples = min(beam_width, 16)  # Limit samples per beam
            candidates = torch.multinomial(probs, num_samples, replacement=False)
            for candidate in candidates:
                new_ids = beam_ids.clone()
                new_ids = torch.cat(
                    [new_ids, candidate.unsqueeze(0).unsqueeze(0)], dim=-1
                )

                # Calculate sequence score
                candidate_prob = probs[candidate].item()
                new_score = beam_score * candidate_prob

                # Check if sequence is complete
                if candidate.item() == models.tokenizer.eos_token_id:
                    decoded = models.tokenizer.decode(new_ids[0])
                    if decoded not in [seq[0] for seq in completed_sequences]:
                        print(
                            f"Completed: {len(completed_sequences)} ({new_score:.4f})"
                        )
                        completed_sequences.append((decoded, new_score))
                else:
                    next_beams.append(
                        (
                            new_ids,
                            clone_kv(generation_output.past_key_values),
                            new_score,
                        )
                    )

        # Prune and select top beams
        next_beams.sort(key=lambda x: x[2], reverse=True)
        active_beams = next_beams[:beam_width]

        # Break if no active beams
        if not active_beams:
            break

        # Avoid infinite loops
        if any(ids.shape[-1] > 1024 for ids, _, _ in active_beams):
            break

    print("Total output tokens:", total_gen)
    print("Completed sequences:", len(completed_sequences))
    # If we don't have enough sequences, return what we have
    return completed_sequences[:variations]


def stochastic_beam_search(prompt, variations=7, temperature=1.0, min_p=0.1):
    # recorder
    total_gen = 0

    # Initialize with first tokens
    inputs = models.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(models.text_model.parameters()).device)
    input_length = input_ids.shape[-1]

    beam_width = min(variations * 3, 32)
    active_beams = [(input_ids, None, 0.0)]  # Using log probabilities
    completed_sequences = []

    # Setup processors
    recorder = LogitsRecorder()
    processors = LogitsProcessorList()
    processors.append(recorder)

    while len(completed_sequences) < variations:
        next_beams = []

        for beam_ids, beam_past_kv, log_score in active_beams:
            recorder.clean()
            gen_kwargs = {
                "max_new_tokens": 1,
                "return_dict_in_generate": True,
                "output_scores": True,
                "do_sample": True,
            }
            generation_config = GenerationConfig(**gen_kwargs)

            with torch.no_grad():
                generation_output = models.text_model.generate(
                    input_ids=beam_ids.clone(),
                    generation_config=generation_config,
                    logits_processor=processors,
                    past_key_values=beam_past_kv,
                )
            total_gen += 1

            # Get logits and apply temperature + min_p
            logits = recorder.scores[-1][0]
            logits = logits / temperature

            # Get log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)

            # Apply min_p filtering
            # mask = probs < min_p * probs.max()
            # log_probs[mask] = float("-inf")

            # Add Gumbel noise for stochastic sampling
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(log_probs) + 1e-10))
            perturbed_logprobs = log_probs + gumbel_noise

            # Get top-k candidates using perturbed scores
            num_samples = min(beam_width, 16)
            topk_values, topk_indices = perturbed_logprobs.topk(num_samples)

            for value, token_id in zip(topk_values, topk_indices):
                new_ids = beam_ids.clone()
                new_ids = torch.cat(
                    [new_ids, token_id.unsqueeze(0).unsqueeze(0)], dim=-1
                )

                # Update sequence log probability
                new_log_score = log_score + log_probs[token_id].item()

                if token_id.item() == models.tokenizer.eos_token_id:
                    decoded = models.tokenizer.decode(new_ids[0])
                    if decoded not in [seq[0] for seq in completed_sequences]:
                        print(
                            f"Completed: {len(completed_sequences)} ({math.exp(new_log_score):.4f})"
                        )
                        completed_sequences.append((decoded, new_log_score))
                else:
                    next_beams.append(
                        (
                            new_ids,
                            clone_kv(generation_output.past_key_values),
                            new_log_score,
                        )
                    )

        # Select top beams based on perturbed scores
        next_beams.sort(key=lambda x: x[2], reverse=True)
        active_beams = next_beams[:beam_width]

        if not active_beams:
            break

        if any(ids.shape[-1] > 1024 for ids, _, _ in active_beams):
            break

    print("Total output tokens:", total_gen)
    print("Completed sequences:", len(completed_sequences))

    # Convert log probabilities back to probabilities for return
    return [(seq, math.exp(score)) for seq, score in completed_sequences[:variations]]


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
        if node.is_leaf:
            dot.node(str(f"leaf#{node.idx}"))
            dot.edge(str(node.idx), str(f"leaf#{node.idx}"))
        elif node.simulated_result is not None:
            dot.node(str(f"sim#{node.idx}"))
            dot.edge(str(node.idx), str(f"sim#{node.idx}"))
        for child in node.childs:
            dot.node(str(child.idx))
            dot.edge(str(node.idx), str(child.idx))
            add_nodes_edges(child)

    dot.node(str(node.idx))  # Add root node
    add_nodes_edges(node)
    return dot


def _count(node: SampleNode, depth: int = 0, total_childs=None, total_nodes=None):
    if node.is_leaf:
        return
    if depth not in total_childs:
        total_childs[depth] = 0
        total_nodes[depth] = 0
    total_childs[depth] += len(node.childs)
    total_nodes[depth] += 1
    for child in node.childs:
        _count(child, depth + 1, total_childs, total_nodes)


def count(node: SampleNode):
    total_childs = {}
    total_nodes = {}
    _count(node, total_childs=total_childs, total_nodes=total_nodes)
    return total_childs, total_nodes


DEFAULT_FORMAT = (
    "<|special|>, <|characters|>, <|copyrights|>, "
    "<|artist|>, <|general|>, <|quality|>, <|meta|>, <|rating|>"
)


if __name__ == "__main__":
    models.load_model(
        "KBlueLeaf/TIPO-100M",
        device="cuda",
    )

    meta, operations, general, prompt = tipo.parse_tipo_request(
        seperate_tags("scenery, wide shot, masterpiece, safe".split(",")),
        "",
    )
    mode, length, expand = operations[0]
    prompt = tipo.apply_tipo_prompt(meta, general, prompt, mode, length, expand)

    results = beam_search_sample(prompt, 1024)
    gen_per_prompt = [x[1] for x in results]
    print(sum(gen_per_prompt) / len(gen_per_prompt))
    with open("./test/beam_search.txt", "w", encoding="utf-8") as f:
        for result, gen in sorted(results):
            result = tipo.parse_tipo_result(result)
            formatted_output = apply_format(result, DEFAULT_FORMAT)
            f.write(formatted_output + "\n")
    # for result in sorted(results):
    #     print("=" * 20)
    #     print(result)
    # print("=" * 20)
