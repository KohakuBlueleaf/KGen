import os
import sys

from kgen_exp.metrics.aesthetic import AestheticRunner


if __name__ == "__main__":
    runner = AestheticRunner()

    PATH = "./data/short-tlong"
    results = {}
    for idx, folder in enumerate([i for i in os.listdir(PATH) if i in {"Prompt-DB"}]):
        img_files = [
            os.path.join(PATH, folder, i)
            for i in os.listdir(os.path.join(PATH, folder))
        ]

        images = [i for i in img_files if i.endswith(".webp")]
        print(folder, len(images))
        result = runner.eval_multi(images, batch_size=128)
        results[folder] = result
        print(folder)
        with open(f"./output/short-tlong-aesthetic.jsonl", "a") as f:
            f.write(str(result) + "\n")
