import kgen.models as models
import kgen.executor.tipo as tipo
from kgen.formatter import seperate_tags, apply_format
from kgen.sampling import DEFAULT_FORMAT, DEFAULT_SAMPLING_CONFIG, conventional_sample


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

    results = conventional_sample(prompt, 1024, **DEFAULT_SAMPLING_CONFIG)
    gen_per_prompt = [x[1] for x in results]
    print(sum(gen_per_prompt) / len(gen_per_prompt))
    with open("./test/conventional.txt", "w", encoding="utf-8") as f:
        for result, gen in sorted(results):
            result = tipo.parse_tipo_result(result)
            formatted_output = apply_format(result, DEFAULT_FORMAT)
            f.write(formatted_output + "\n")