import kgen.models as models
import kgen.executor.tipo as tipo
from kgen.formatter import seperate_tags, apply_format
from kgen.sampling import DEFAULT_FORMAT, DEFAULT_SAMPLING_CONFIG
from kgen.sampling.beam import (
    beam_search_sample, diverse_beam_search, stochastic_beam_search
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

    results =  beam_search_sample(
        prompt,
        num_sequences=1024
    )
    gen_per_prompt = [x[1] for x in results]
    print(sum(gen_per_prompt) / len(gen_per_prompt))
    with open("./test/beam_search.txt", "w", encoding="utf-8") as f:
        for result, gen in sorted(results):
            result = tipo.parse_tipo_result(result)
            formatted_output = apply_format(result, DEFAULT_FORMAT)
            f.write(formatted_output + "\n")

    results =  stochastic_beam_search(
        prompt,
        num_sequences=1024
    )
    gen_per_prompt = [x[1] for x in results]
    print(sum(gen_per_prompt) / len(gen_per_prompt))
    with open("./test/stochastic_beam_search.txt", "w", encoding="utf-8") as f:
        for result, gen in sorted(results):
            result = tipo.parse_tipo_result(result)
            formatted_output = apply_format(result, DEFAULT_FORMAT)
            f.write(formatted_output + "\n")

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
