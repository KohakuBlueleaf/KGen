import os
import pathlib

import torch
from huggingface_hub import hf_hub_download
from transformers import LlamaForCausalLM, LlamaTokenizer

from .logging import logger
from .patch import patch


text_model = None
tokenizer = None
current_model_name = None


model_dir = pathlib.Path(__file__).parent / "models"
model_list = [
    "KBlueLeaf/DanTagGen-delta-rev2",
    "KBlueLeaf/DanTagGen-delta",
    "KBlueLeaf/DanTagGen-beta",
    "KBlueLeaf/DanTagGen-alpha",
    "KBlueLeaf/DanTagGen-gamma",
]
gguf_name = [
    "ggml-model-Q6_K.gguf",
    "ggml-model-Q8_0.gguf",
    "ggml-model-f16.gguf",
]
model_have_quality_info = {
    "DanTagGen-delta-rev2": True,
    "DanTagGen-delta": True,
    "DanTagGen-beta": False,
    "DanTagGen-alpha": False,
    "DanTagGen-gamma": False,
}

tipo_model_list = [
    ("KBlueLeaf/TIPO-200M-ft", ["TIPO-200M-ft-F16.gguf"]),
    ("KBlueLeaf/TIPO-500M", ["TIPO-500M_epoch5-F16.gguf"]),
    ("KBlueLeaf/TIPO-200M", ["TIPO-200M-40Btok-F16.gguf"]),
    ("KBlueLeaf/TIPO-100M", ["TIPO-100M-F16.gguf"]),
]


try:
    from llama_cpp import Llama
except Exception:
    logger.warning("Llama-cpp-python cannot be imported")
    Llama = None


if not os.path.isdir(model_dir):
    os.makedirs(model_dir, exist_ok=True)


def download_gguf(model_name=model_list[0], gguf_name=gguf_name[-1]):
    if os.path.isfile(model_dir / f"{model_name.split('/')[-1]}_{gguf_name}"):
        logger.info(f"GGUF model {model_name} already exists")
        return model_dir / f"{model_name.split('/')[-1]}_{gguf_name}"
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


def load_model(
    model_name=model_list[0],
    gguf=False,
    device="cpu",
    subfolder=None,
    tokenizer_name=None,
    strict=True,
    **kwargs,
):
    global text_model, tokenizer, current_model_name
    if gguf:
        model_name = os.path.basename(model_name)
        model_repo_name = model_name.split("_")[0]
        current_model_name = model_repo_name
        try:
            assert Llama is not None
            text_model = Llama(
                str(model_dir / model_name),
                n_ctx=1024,
                n_gpu_layers=0 if device == "cpu" else 1000,
                verbose=False,
                **kwargs,
            )
            if tokenizer_name is not None:
                tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
            else:
                tokenizer = None
            logger.info(f"Llama-cpp-python/gguf model {model_name} loaded")
            if device == "cuda":
                logger.warning(
                    "llama.cpp have reproducibility issue on cuda "
                    "(https://github.com/ggerganov/llama.cpp/pull/1346) "
                    "It is suggested to use cpu or "
                    "compile llama-cpp-python by yourself "
                    "and set GGML_CUDA_MAX_STREAMS in the file ggml-cuda.cu to 1."
                )
            return
        except Exception as e:
            if strict:
                raise e
            logger.warning(f"Llama-cpp-python/gguf model {model_name} load failed")
            model_name = f"KBlueLeaf/{model_repo_name}"
    logger.info(f"Using transformers model {model_name}")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if subfolder is not None:
        kwargs["subfolder"] = subfolder
    text_model = patch(
        LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            **kwargs,
        )
        .requires_grad_(False)
        .eval()
        .to(device)
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        tokenizer_name or model_name, subfolder=subfolder
    )
    current_model_name = model_name.split("/")[-1]
    logger.info(f"Model {model_name} loaded")


if __name__ == "__main__":
    model_file = download_gguf()
    load_model(model_file, gguf=True)
