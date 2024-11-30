import os
import torch
from transformers import AutoModel

from kgen_exp.metrics.fd_text import FDTextRunner


if __name__ == "__main__":
    model = (
        AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        .bfloat16()
        .eval()
        .requires_grad_(False)
        .cuda()
    )
    runner = FDTextRunner(lambda x: torch.from_numpy(model.encode(x)))

    PATH = "./test"
    ref = "./test/reference.txt"
    with open(ref, "r", encoding="utf-8") as f:
        ref_texts = f.readlines()

    results = {}
    for idx, file in enumerate([i for i in os.listdir(PATH)]):
        if file == "reference.txt":
            continue
        texts = []
        with open(os.path.join(PATH, file), "r", encoding="utf-8") as f:
            texts = f.readlines()

        if runner.b_features == []:
            result = runner.eval_multi(texts, ref_texts, batch_size=128)
        else:
            result = runner.eval_multi(texts, batch_size=128)
        results[file] = result

    print("=" * 20)
    for file, result in results.items():
        print(f"{file:<10}:", result.item())
