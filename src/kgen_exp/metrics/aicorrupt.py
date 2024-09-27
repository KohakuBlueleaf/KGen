import os
import numpy
import orjsonl
import orjson
from sdeval.corrupt import AICorruptMetrics

from .base import MetricRunner


class AICorruptRunner(MetricRunner):
    single = True
    multi = True

    def __init__(self):
        # load model and preprocessor
        self.model = AICorruptMetrics(silent=True)

    def eval(self, images, ref_texts=None, ref_images=None):
        return self.model.score(images, mode="seq")

    def eval_multi(self, images, ref_texts=None, ref_images=None, batch_size=32):
        results = super().eval_multi(images, ref_texts, ref_images, batch_size)
        results = numpy.concatenate(results, axis=0)
        return [float(i) for i in results]


if __name__ == "__main__":
    runner = AICorruptRunner()
    images = [None] * 4

    PATH = "./data/coyo-img"
    img_files = [os.path.join(PATH, i) for i in os.listdir(PATH)]
    images[0] = [i for i in img_files if i.endswith("_short.webp")]
    images[1] = [i for i in img_files if i.endswith("_truncate_long.webp")]
    images[2] = [i for i in img_files if i.endswith("_gen.webp")]
    images[3] = [i for i in img_files if i.endswith("_extend.webp")]

    with open("coyo-ai-corrupt.jsonl", "w") as f:
        for idx, name in enumerate(["short", "truncate", "tipo gen", "tipo extend"]):
            result = runner.eval_multi(images[idx], batch_size=500)
            f.write(str(result) + '\n')