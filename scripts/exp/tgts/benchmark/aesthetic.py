import os
import sys

from kgen_exp.metrics.aesthetic import AestheticRunner


if __name__ == "__main__":
    runner = AestheticRunner()

    PATH = "./download"
    all_images = [
        os.path.join(PATH, i) for i in os.listdir(PATH) if i.endswith(".webp")
    ]
    images = {}
    for i in all_images:
        basename = os.path.basename(i)
        cate = basename.rsplit("-", 1)[0]
        if cate not in images:
            images[cate] = []
        images[cate].append(i)

    results = {}
    for idx, folder in enumerate(images.keys()):
        print(folder, len(images[folder]))
        result = runner.eval_multi(images[folder], batch_size=128)
        results[folder] = result
        with open(f"./output/tgts-aesthetic.jsonl", "a") as f:
            f.write(str(result) + "\n")
