import os
import sys
import numpy as np

from kgen_exp.metrics.aesthetic import AestheticRunner


def main():
    runner = AestheticRunner()

    for folder in [
        # "./data/addon/output/flux-dev-fp8",
        # "./data/addon/output/omnigen2",
        # "./data/addon/output-lumina_hidream/lumina-2",
        # "./data/addon/output-lumina_hidream/hidream_i1_dev",
        "./output/short-long-gen-extend-g2f"
        # r"F:\nn\HDM\data\eval\steps=16"
    ]:
        # img_files = [
        #     os.path.join(folder, i)
        #     for i in os.listdir(folder)
        # ]
        # images = [i for i in img_files if any(i.endswith(ext) for ext in [".png", ".webp"])]
        # print(folder, len(images))
        # result = runner.eval_multi(images, batch_size=128)
        # result = np.asarray(result)
        # mean = result.mean()
        # std = result.std()
        # print(mean, std)
        img_files_short = [
            os.path.join(folder, i) for i in os.listdir(folder) if "short" in i.lower()
        ]
        img_files_gen = [
            os.path.join(folder, i) for i in os.listdir(folder) if "gen" in i.lower()
        ]
        img_files_tlong = [
            os.path.join(folder, i) for i in os.listdir(folder) if "tlong" in i.lower()
        ]
        img_files_extend = [
            os.path.join(folder, i) for i in os.listdir(folder) if "extend" in i.lower()
        ]

        images = [
            i
            for i in img_files_short
            if any(i.endswith(ext) for ext in [".png", ".webp"])
        ]
        print(folder, len(images))
        result = runner.eval_multi(images, batch_size=128)
        result = np.asarray(result)
        smean = result.mean()
        sstd = result.std()

        images = [
            i
            for i in img_files_gen
            if any(i.endswith(ext) for ext in [".png", ".webp"])
        ]
        print(folder, len(images))
        result = runner.eval_multi(images, batch_size=128)
        result = np.asarray(result)
        tmean = result.mean()
        tstd = result.std()

        images = [
            i
            for i in img_files_tlong
            if any(i.endswith(ext) for ext in [".png", ".webp"])
        ]
        print(folder, len(images))
        result = runner.eval_multi(images, batch_size=128)
        result = np.asarray(result)
        tlmean = result.mean()
        tlstd = result.std()

        images = [
            i
            for i in img_files_extend
            if any(i.endswith(ext) for ext in [".png", ".webp"])
        ]
        print(folder, len(images))
        result = runner.eval_multi(images, batch_size=128)
        result = np.asarray(result)
        exmean = result.mean()
        exstd = result.std()

        print(f"Short : {smean:.4f}±{sstd:.4f}")
        print(f"TIPO  : {tmean:.4f}±{tstd:.4f}")
        print(f"TLong : {tlmean:.4f}±{tlstd:.4f}")
        print(f"Extend: {exmean:.4f}±{exstd:.4f}")
        print()
