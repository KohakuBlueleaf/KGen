import os
import sys

from kgen_exp.metrics.fid import DINOv2FeatureExtractor, FIDRunner


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "vitb14"
    swinv2 = DINOv2FeatureExtractor(model)
    runner = FIDRunner(swinv2, (224, 224))
    print(model)

    PATH = r"./data/cc12m"
    ref_files = [os.path.join(PATH, i) for i in os.listdir(PATH)]
    refs = [i for i in ref_files if i.endswith(".jpg")]

    PATH = r"./data/coyo11m"
    ref_files = [os.path.join(PATH, i) for i in os.listdir(PATH)]
    refs += [i for i in ref_files if i.endswith(".jpg")]

    refs = []
    PATH = r"./data/dan-scenery-webp"
    ref_files = [os.path.join(PATH, i) for i in os.listdir(PATH)]
    refs += [i for i in ref_files if i.endswith("_org.webp")]
    print("ref image count:", len(refs))

    need_ref = True

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
        # result = runner.eval_multi(images, ref_images=refs if need_ref else [], batch_size=128)
        # print(result)
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
        sresult = runner.eval_multi(
            images, ref_images=refs if need_ref else [], batch_size=128
        )

        images = [
            i
            for i in img_files_gen
            if any(i.endswith(ext) for ext in [".png", ".webp"])
        ]
        print(folder, len(images))
        tresult = runner.eval_multi(images, ref_images=[], batch_size=128)

        images = [
            i
            for i in img_files_tlong
            if any(i.endswith(ext) for ext in [".png", ".webp"])
        ]
        print(folder, len(images))
        tlresult = runner.eval_multi(images, ref_images=[], batch_size=128)

        images = [
            i
            for i in img_files_extend
            if any(i.endswith(ext) for ext in [".png", ".webp"])
        ]
        print(folder, len(images))
        exresult = runner.eval_multi(images, ref_images=[], batch_size=128)

        print(f"Short : {sresult:.4f}")
        print(f"Gen   : {tresult:.4f}")
        print(f"Tlong : {tlresult:.4f}")
        print(f"Extend: {exresult:.4f}")
        print()
