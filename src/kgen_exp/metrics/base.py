from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm, trange


# pool = ProcessPoolExecutor(max_workers=24)
pool = ThreadPoolExecutor(max_workers=64)


def load(image):
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    else:
        return image


def batch_load(images, loading_func=load):
    return list(tqdm(pool.map(loading_func, images), total=len(images), leave=False))


class MetricRunner:
    single = False
    multi = False
    img_load_func = load

    def eval(self, images, ref_texts=None, is_ref=False):
        raise NotImplementedError()

    def eval_single(self, image, ref_text=None, is_ref=False):
        if isinstance(image, (Image.Image, str)):
            image = [image]
        if isinstance(ref_text, str):
            ref_text = [ref_text]
        image = batch_load(image, self.img_load_func)

        return self.eval(image, ref_text, is_ref=is_ref)

    def eval_multi(self, images, ref_texts=None, ref_images=None, batch_size=32):
        if ref_texts is None:
            ref_texts = [None] * max(
                len(images), 0 if ref_images is None else len(ref_images)
            )
        results = []
        ref_results = []
        for i in trange(0, len(images), batch_size):
            results.append(
                self.eval_single(
                    images[i : i + batch_size],
                    ref_texts[i : i + batch_size],
                    is_ref=False,
                )
            )
        if ref_images is not None:
            for i in trange(0, len(ref_images), batch_size):
                ref_results.append(
                    self.eval_single(
                        ref_images[i : i + batch_size],
                        ref_texts[i : i + batch_size],
                        is_ref=True,
                    )
                )

        return results, ref_results


class TextMetricRunner:
    single = False
    multi = False

    def eval(self, text, ref_texts=None):
        raise NotImplementedError()

    def eval_single(self, text, ref_text=None):
        if isinstance(text, str):
            text = [text]
        if isinstance(ref_text, str):
            ref_text = [ref_text]
        return self.eval(text, ref_text)

    def eval_multi(self, texts, ref_texts=None, batch_size=32):
        if ref_texts is None:
            ref_texts = [None] * len(texts)
        results = []
        for i in trange(0, len(texts), batch_size):
            results.append(
                self.eval_single(
                    texts[i : i + batch_size],
                    ref_texts[i : i + batch_size],
                )
            )
        return results
