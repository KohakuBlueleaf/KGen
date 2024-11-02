import os
import numpy
import orjsonl
import orjson
from PIL import Image
from sdeval.corrupt import AICorruptMetrics
from sdeval.corrupt.aicorrupt import _DEFAULT_MODEL_NAME

from .base import MetricRunner, load


class AICorruptRunner(MetricRunner):
    single = True
    multi = True

    def __init__(self, model_name=_DEFAULT_MODEL_NAME):
        # load model and preprocessor
        self.model = AICorruptMetrics(model_name=model_name, silent=True)

    def img_load_func(self, image):
        image = load(image)
        image = image.resize((384, 384), Image.BICUBIC)
        return image

    def eval(self, images, ref_texts=None, is_ref=False):
        return self.model.score(images, mode="seq")

    def eval_multi(self, images, ref_texts=None, ref_images=None, batch_size=32):
        results, ref_results = super().eval_multi(
            images, ref_texts, ref_images, batch_size
        )
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
            f.write(str(result) + "\n")
