import os
import pathlib

from huggingface_hub import hf_hub_download
from transformers import LlamaForCausalLM, LlamaTokenizer

from .logging import logger


model_dir = pathlib.Path(__file__).parent / "models"
model_list = [
    "KBlueLeaf/DanTagGen-alpha",
    "KBlueLeaf/DanTagGen-beta",
    "KBlueLeaf/DanTagGen-gamma",
]
gguf_name = [
    "ggml-model-f16.gguf",
    "ggml-model-Q8_0.gguf",
    "ggml-model-Q6_K.gguf",
]


try:
    from llama_cpp import Llama
except Exception:
    logger.warning("Llama-cpp-python cannot be imported")
    Llama = None


if not os.path.isdir(model_dir):
    os.makedirs(model_dir, exist_ok=True)


def download_gguf(model_name="KBlueLeaf/DanTagGen-gamma", gguf_name=gguf_name[-1]):
    logger.info(f"Downloading gguf model from {model_name}")
    result = hf_hub_download(
        model_name,
        gguf_name,
        repo_type="model",
        cache_dir=None,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
    )
    new_name = model_dir / f"{model_name.split('/')[-1]}_{gguf_name}"
    if os.path.isfile(new_name):
        os.remove(new_name)
    os.rename(result, new_name)
    logger.info(f"Downloaded gguf model to {new_name}")
    return new_name


def list_gguf():
    files = [
        str(model_dir / file)
        for file in os.listdir(model_dir)
        if file.endswith(".gguf")
    ]
    return files


def load_model(model_name="KBlueLeaf/DanTagGen-gamma", gguf=False):
    global text_model, tokenizer
    if gguf:
        try:
            assert Llama is not None
            model_name = os.path.basename(model_name)
            text_model = Llama(
                str(model_dir / model_name),
                n_ctx=384,
                n_gpu_layers=100,
                verbose=False,
            )
            tokenizer = None
            logger.info(f"Llama-cpp-python/gguf model {model_name} loaded")
            return
        except Exception as e:
            logger.warning(f"Llama-cpp-python/gguf model {model_name} load failed")
            model_name = model_list[-1]
    logger.info(f"Using transformers model {model_name}")
    text_model = LlamaForCausalLM.from_pretrained(model_name).eval().half()
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    logger.info(f"Model {model_name} loaded")


if __name__ == "__main__":
    model_file = download_gguf()
    load_model(model_file, gguf=True)
