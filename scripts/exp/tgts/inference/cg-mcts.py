import torch

import kgen.models as models
import kgen.executor.tipo as tipo
from kgen.formatter import seperate_tags, apply_format
from kgen.sampling import (
    count,
    DEFAULT_FORMAT,
    DEFAULT_SAMPLING_CONFIG,
)
from kgen.sampling.cg_mcts import cg_mcts_sample


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

    for exploration in [0.5, 1.0, 2.0, 3.0]:
        results, root = cg_mcts_sample(
            prompt,
            ids_splitters=[lambda ids, i: torch.sum(ids[0, i:] == 29892) >= 4],
            variations=1024,
            exploration=exploration,
            **DEFAULT_SAMPLING_CONFIG,
        )
        with open(f"./test/cg-mcts_exp-{exploration}.txt", "w", encoding="utf-8") as f:
            for result, gen in sorted(results):
                result = tipo.parse_tipo_result(result)
                formatted_output = apply_format(result, DEFAULT_FORMAT)
                f.write(formatted_output + "\n")

        total_childs, total_nodes = count(root)

        print(f"Total nodes per depth: {total_nodes}")
        print(
            "Average childs per node per depth: "
            f"{[total_childs[i] / total_nodes[i] for i in range(len(total_childs))]}"
        )
        gen_per_prompt = [x[1] for x in results]
        print(sum(gen_per_prompt) / len(gen_per_prompt))
