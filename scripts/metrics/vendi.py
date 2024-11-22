import os
import sys

from kgen_exp.metrics.vendi import DINOv2FeatureExtractor, VendiRunner


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "vitb14"
    swinv2 = DINOv2FeatureExtractor(model)
    runner = VendiRunner(swinv2, (224, 224))
    print(model)

    PATH = "./data/worst"
    results = {}
    for idx, folder in enumerate([i for i in os.listdir(PATH)]):
        img_files = [
            os.path.join(PATH, folder, i)
            for i in os.listdir(os.path.join(PATH, folder))
        ]

        images = [i for i in img_files if i.endswith(".webp")]
        print(folder, len(images))
        result = runner.eval_multi(images, batch_size=512)
        results[folder] = result
        print(folder, result)

    print("=" * 20)
    for folder, result in results.items():
        print(f"{folder:<10}:", result.item())
