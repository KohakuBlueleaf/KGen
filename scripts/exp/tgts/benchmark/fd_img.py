import os
import sys
import numpy as np
from kgen_exp.metrics.fid import DINOv2FeatureExtractor, FIDRunner


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "vitb14"
    swinv2 = DINOv2FeatureExtractor(model)
    runner = FIDRunner(swinv2, (224, 224))
    print(model)

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

    ref = images.pop("reference")
    print("ref image count:", len(ref))
    print("=" * 20)
    results = {}
    for idx, folder in enumerate(images.keys()):
        img_files = images[folder][:1024]
        print(folder, len(img_files))
        result = runner.eval_multi(
            img_files, ref_images=ref if not idx else [], batch_size=512
        )
        results[folder] = result
        print(folder, result)

    print("=" * 20)
    for folder, result in results.items():
        print(f"{folder:<10}:", result.item())
