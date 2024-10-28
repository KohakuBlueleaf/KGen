import os
import sys

from kgen_exp.metrics.fid import DINOv2FeatureExtractor, FIDRunner


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "vitb14"
    swinv2 = DINOv2FeatureExtractor(model)
    runner = FIDRunner(swinv2, (224, 224))
    print(model)

    PATH = r"./data/cc12m/tipo-exp"
    img_files = [os.path.join(PATH, i) for i in os.listdir(PATH)]
    refs = [i for i in img_files if i.endswith(".jpg")]
    PATH = r"./data/tipo-exp"
    img_files = [os.path.join(PATH, i) for i in os.listdir(PATH)]
    refs += [i for i in img_files if i.endswith(".jpg")]
    print("ref image count:", len(refs))

    PATH = "./data/short-tlong"
    results = {}
    for idx, folder in enumerate([i for i in os.listdir(PATH)]):
        img_files = [
            os.path.join(PATH, folder, i)
            for i in os.listdir(os.path.join(PATH, folder))
            if "short" in i
        ]

        images = [i for i in img_files if i.endswith(".webp")]
        print(folder, len(images))
        result = runner.eval_multi(
            images, ref_images=refs if not idx else [], batch_size=500
        )
        results[folder] = result

    print("=" * 20)
    for folder, result in results.items():
        print(f"{folder:<10}:", result.item())
