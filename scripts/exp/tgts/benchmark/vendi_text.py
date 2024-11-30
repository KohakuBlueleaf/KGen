import os
import torch
import numpy as np
from transformers import AutoModel

from kgen_exp.metrics.vendi_text import VendiTextRunner


if __name__ == "__main__":
    model = (
        AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        .bfloat16()
        .eval()
        .requires_grad_(False)
        .cuda()
    )
    runner = VendiTextRunner(lambda x: torch.from_numpy(model.encode(x)))

    PATH = "./test"
    results = {}
    simmats = {}
    for idx, file in enumerate([i for i in os.listdir(PATH)]):
        texts = []
        with open(os.path.join(PATH, file), "r", encoding="utf-8") as f:
            texts = sorted(f.readlines())

        result, sim_mat = runner.eval_multi(texts, batch_size=128)
        results[file] = result
        simmats[file] = sim_mat

    print("=" * 20)
    for file, result in results.items():
        print(f"{file:<10}:", result.item())

    np.save("./output/tgts-sim.npy", simmats)
