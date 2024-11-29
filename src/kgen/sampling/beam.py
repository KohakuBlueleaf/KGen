import math
from typing import Optional

import numpy as np
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

import kgen.models as models
import kgen.executor.tipo as tipo
from kgen.formatter import seperate_tags, apply_format
from kgen.generate import generate
from kgen.sampling import (
    SampleNode,
    LogitsRecorder,
    NodeSplitter,
    get_next,
    draw_tree,
    count,
    DEFAULT_FORMAT,
    move_kv,
    clone_kv,
)
from kgen.sampling.node_splitters import tag_splitter


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
                new_score = beam_score + math.log(candidate_prob)

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


def diverse_beam_search(
    prompt,
    num_sequences: int = 16,  # Target number of sequences
    num_groups: Optional[int] = None,  # Optional: specify number of groups
    diversity_penalty: float = 0.75,
    temperature: float = 1.5,
):
    """
    Implements diverse beam search with automatic group/beam width calculation.

    Args:
        prompt: Input prompt text
        num_sequences: Target number of sequences to generate
        num_groups: Optional number of groups (if None, will be calculated)
        diversity_penalty: Penalty factor for repeated tokens across groups
        temperature: Temperature for logits scaling
    """
    # Calculate groups and beam width if not specified
    if num_groups is None:
        # Square root gives a balanced split between groups and beams
        num_groups = int(math.sqrt(num_sequences))

    # Calculate beam width needed per group to get desired sequences
    beam_width_per_group = math.ceil(num_sequences / num_groups)

    # Limit beam width to prevent memory issues
    beam_width_per_group = min(beam_width_per_group, 16)

    print(f"Using {num_groups} groups with {beam_width_per_group} beams each")

    # Initialize with first tokens
    inputs = models.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(models.text_model.parameters()).device)

    # Track beams per group
    group_beams = [
        [(input_ids, None, 0.0)] for _ in range(num_groups)
    ]  # (ids, past_kv, score)
    completed_sequences = []
    total_gen = 0

    # Setup recorder and processors
    recorder = LogitsRecorder()
    processors = LogitsProcessorList()
    processors.append(recorder)

    while True:
        next_group_beams = [[] for _ in range(num_groups)]

        # Process each group sequentially
        for group_id, active_beams in enumerate(group_beams):
            # Track tokens selected by previous groups for diversity
            prev_group_tokens = set()
            if group_id > 0:
                for prev_group in range(group_id):
                    for ids, _, _ in group_beams[prev_group]:
                        if ids.shape[-1] > 1:
                            prev_group_tokens.add(ids[0, -1].item())

            # Expand each beam in current group
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

                # Get logits and apply temperature
                logits = recorder.scores[-1][0] / temperature

                # Apply diversity penalty
                if prev_group_tokens:
                    diversity_mask = torch.zeros_like(logits)
                    diversity_mask[list(prev_group_tokens)] = -diversity_penalty
                    logits = logits + diversity_mask

                # Get log probabilities
                log_probs = torch.log_softmax(logits, dim=-1)

                # Sample top candidates for this beam
                topk_values, topk_indices = log_probs.topk(beam_width_per_group)

                for value, token_id in zip(topk_values, topk_indices):
                    new_ids = beam_ids.clone()
                    new_ids = torch.cat(
                        [new_ids, token_id.unsqueeze(0).unsqueeze(0)], dim=-1
                    )
                    new_log_score = log_score + value.item()

                    if token_id.item() == models.tokenizer.eos_token_id:
                        decoded = models.tokenizer.decode(new_ids[0])
                        if decoded not in [seq[0] for seq in completed_sequences]:
                            score = math.exp(new_log_score)
                            print(
                                f"Completed (Group {group_id}): {len(completed_sequences)} ({score:.4f})"
                            )
                            completed_sequences.append(
                                (decoded, new_log_score, group_id)
                            )
                    else:
                        next_group_beams[group_id].append(
                            (
                                new_ids,
                                clone_kv(generation_output.past_key_values),
                                new_log_score,
                            )
                        )

            # Sort and prune beams within group
            next_group_beams[group_id].sort(key=lambda x: x[2], reverse=True)
            next_group_beams[group_id] = next_group_beams[group_id][
                :beam_width_per_group
            ]

        # Update all groups
        group_beams = next_group_beams

        # Check stopping conditions
        if not any(beams for beams in group_beams):
            break

        if any(
            any(ids.shape[-1] > 1024 for ids, _, _ in beams) for beams in group_beams
        ):
            break

        # Break if we have enough sequences
        if len(completed_sequences) >= num_sequences:
            break

    print("Total output tokens:", total_gen)
    print("Completed sequences:", len(completed_sequences))
    print("Groups represented:", len(set(g for _, _, g in completed_sequences)))

    # Convert log probabilities to probabilities and sort
    final_results = []
    for seq, score, group in completed_sequences:
        final_results.append((seq, math.exp(score)))

    # Sort by score and return exactly num_sequences results
    final_results.sort(key=lambda x: x[1], reverse=True)
    return final_results[:num_sequences]


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

    # results = beam_search_sample(prompt, 1024)
    results = diverse_beam_search(
        prompt,
        num_sequences=1024,  # Number of sequences to generate
        diversity_penalty=0.75,  # How strongly to encourage diversity
        temperature=1.5,  # Temperature for sampling
    )
    gen_per_prompt = [x[1] for x in results]
    print(sum(gen_per_prompt) / len(gen_per_prompt))
    with open("./test/div_beam_search.txt", "w", encoding="utf-8") as f:
        for result, gen in sorted(results):
            result = tipo.parse_tipo_result(result)
            formatted_output = apply_format(result, DEFAULT_FORMAT)
            f.write(formatted_output + "\n")
    # for result in sorted(results):
    #     print("=" * 20)
    #     print(result)
    # print("=" * 20)
