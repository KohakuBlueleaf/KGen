from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from PIL import Image
from tqdm import tqdm, trange


pool = ProcessPoolExecutor(max_workers=16)


def load(image):
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    else:
        return image


def batch_load(images):
    return list(tqdm(pool.map(load, images), total=len(images), leave=False))


class MetricRunner:
    single = False
    multi = False

    def eval(self, images, ref_texts=None, ref_images=None):
        raise NotImplementedError()

    def eval_single(self, image, ref_text=None, ref_image=None):
        if isinstance(image, (Image.Image, str)):
            image = [image]
        if isinstance(ref_image, (Image.Image, str)):
            ref_image = [ref_image]
        if isinstance(ref_text, str):
            ref_text = [ref_text]
        image = batch_load(image)
        ref_image = batch_load(ref_image)

        return self.eval(image, ref_text, ref_image)

    def eval_multi(self, images, ref_texts=None, ref_images=None, batch_size=32):
        if ref_texts is None:
            ref_texts = [None] * len(images)
        if ref_images is None:
            ref_images = [None] * len(images)
        results = []
        for i in trange(0, len(images), batch_size):
            results.append(
                self.eval_single(
                    images[i : i + batch_size],
                    ref_texts[i : i + batch_size],
                    ref_images[i : i + batch_size],
                )
            )

        return results
