import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import orjsonl
from transformers import AutoModel, AutoTokenizer, T5EncoderModel
from PIL import Image
from tqdm import trange, tqdm
from scipy import linalg

from .base import MetricRunner, batch_load
from kgen.utils import remove_repeated_suffix
from kgen.formatter import apply_format

DEFAULT_FORMAT = """<|special|>, <|characters|>, <|copyrights|>, <|artist|>, <|general|>, <|generated|>. <|quality|>, <|meta|>, <|rating|>"""


def mmd(
    p_samples: torch.Tensor,
    q_samples: torch.Tensor,
    kernel: str = "rbf",
    sigma: float = 1.0,
) -> float:
    """
    Maximum Mean Discrepancy with RBF kernel.

    Args:
        p_samples: [B, dim] tensor
        q_samples: [B, dim] tensor
        kernel: Kernel type ('rbf' or 'linear')
        sigma: Bandwidth for RBF kernel

    Returns:
        MMD value
    """
    # Convert to numpy float64
    p_samples = p_samples.cpu().numpy().astype(np.float64)
    q_samples = q_samples.cpu().numpy().astype(np.float64)

    def rbf_kernel(x, y, sigma):
        # x: [n, dim], y: [m, dim]
        # returns: [n, m]
        x_norm = (x**2).sum(axis=1).reshape(-1, 1)
        y_norm = (y**2).sum(axis=1).reshape(1, -1)
        dists = x_norm + y_norm - 2 * x @ y.T
        return np.exp(-dists / (2 * sigma**2))

    def linear_kernel(x, y):
        return x @ y.T

    if kernel == "rbf":
        k = lambda x, y: rbf_kernel(x, y, sigma)
    else:
        k = linear_kernel

    # MMD^2 = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
    kxx = k(p_samples, p_samples).mean()
    kyy = k(q_samples, q_samples).mean()
    kxy = k(p_samples, q_samples).mean()

    mmd_squared = kxx - 2 * kxy + kyy

    return float(np.sqrt(max(mmd_squared, 0.0)))


def frechet_distance(
    p_samples: torch.Tensor, q_samples: torch.Tensor, eps: float = 1e-6
) -> float:
    """
    Frechet Distance (FID-like) between two distributions.
    Uses scipy for stable matrix square root computation.

    Args:
        p_samples: [B, dim] tensor
        q_samples: [B, dim] tensor
        eps: Small constant for numerical stability

    Returns:
        Frechet distance value
    """
    # Convert to numpy float64 for numerical stability
    p_samples = p_samples.cpu().numpy().astype(np.float64)
    q_samples = q_samples.cpu().numpy().astype(np.float64)

    mu_p = p_samples.mean(axis=0)
    mu_q = q_samples.mean(axis=0)

    sigma_p = np.cov(p_samples.T) + eps * np.eye(p_samples.shape[1])
    sigma_q = np.cov(q_samples.T) + eps * np.eye(q_samples.shape[1])

    # Frechet distance: ||mu_p - mu_q||^2 + Tr(sigma_p + sigma_q - 2*sqrt(sigma_p @ sigma_q))
    diff = mu_p - mu_q

    # Use scipy for matrix square root (most stable method)
    covmean = linalg.sqrtm(sigma_p @ sigma_q)

    # Handle numerical errors in sqrtm
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            print(
                f"Warning: Imaginary component in sqrtm: {np.max(np.abs(covmean.imag))}"
            )
        covmean = covmean.real

    fd = diff.dot(diff) + np.trace(sigma_p + sigma_q - 2 * covmean)

    return float(fd)


class TextEmbedder:
    def __init__(
        self,
        model_name: str = "google/t5-v1_1-xxl",
        # model_name: str = "jinaai/jina-embeddings-v3",
        device: str | None = None,
        normalize: bool = True,
    ):
        """
        Text embedding class using transformers.

        Args:
            model_name: HuggingFace model name
                Default: "google/t5-v1_1-xxl" (best for t2i prompts)

                # Jina embeddings (general purpose, good performance):
                # - "jinaai/jina-embeddings-v3"
                # - "jinaai/jina-embeddings-v2-base-en"

                # E5 models (strong open-source):
                # - "intfloat/e5-large-v2"
                # - "intfloat/e5-mistral-7b-instruct"

                # Sentence transformers (standard baseline):
                # - "sentence-transformers/all-MiniLM-L6-v2"
                # - "sentence-transformers/all-mpnet-base-v2"

                # RoBERTa (academic baseline):
                # - "roberta-base"
                # - "roberta-large"

            device: Device to run on ('cuda', 'cpu', or None for auto)
            normalize: Whether to L2 normalize embeddings
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if "t5" in model_name.lower():
            self.model = (
                T5EncoderModel.from_pretrained(model_name, trust_remote_code=True)
                .requires_grad_(False)
                .eval()
                .bfloat16()
                .to(self.device)
            )
        else:
            self.model = (
                AutoModel.from_pretrained(model_name, trust_remote_code=True)
                .requires_grad_(False)
                .eval()
                .to(self.device)
            )

    @torch.no_grad()
    def encode(
        self, texts: list[str], batch_size: int | None = 32, show_progress: bool = True
    ) -> torch.Tensor:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of strings to encode
            batch_size: Batch size for processing. If None, process all at once
            show_progress: Whether to show progress bar

        Returns:
            Tensor of shape [len(texts), embedding_dim]
        """
        if batch_size is None:
            batch_size = len(texts)

        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding")

        for i in iterator:
            batch_texts = texts[i : i + batch_size]
            try:
                embeddings = self.model.encode(batch_texts)
                if not isinstance(embeddings, torch.Tensor):
                    # jina-embeddings will return np array
                    embeddings = torch.from_numpy(embeddings)
            except:
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)

                # Get model output
                outputs = self.model(**encoded)

                # Mean pooling
                embeddings = self._mean_pooling(
                    outputs.last_hidden_state.float(), encoded["attention_mask"]
                )

            # Normalize if requested
            if self.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu())

        # Concatenate all batches
        torch.cuda.empty_cache()
        return torch.cat(all_embeddings, dim=0)

    def _mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pooling with attention mask.
        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.model.config.hidden_size


import duckdb

connection_table = {}
result_cache = {}


def get_row_by_index(
    parquet_path: str, index_value: int, index_col: str = "index"
) -> dict:
    global connection_table
    """
    Get a single row from parquet file by index value.
    
    Args:
        parquet_path: Path to parquet file
        index_value: The index value to query
        index_col: Name of the index column (default: "index")
    
    Returns:
        Dictionary of the row data (without index in the dict)
    """
    if parquet_path not in connection_table:
        connection_table[parquet_path] = duckdb.connect()
        con = connection_table[parquet_path]
        con.execute(
            f"""
            CREATE TEMP TABLE meta AS
            SELECT * FROM read_parquet('{parquet_path}')
        """
        )
    else:
        con = connection_table[parquet_path]

    if parquet_path not in result_cache:
        result_cache[parquet_path] = {}
    if index_value in result_cache[parquet_path]:
        return result_cache[parquet_path][index_value]

    query = f"""
        SELECT * FROM meta
        WHERE {index_col} = {index_value}
    """

    df = con.execute(query).df()

    if len(df) == 0:
        result_cache[parquet_path][index_value] = None
        return None

    # Return as dictionary (first row)
    result = df.iloc[0].to_dict()
    result_cache[parquet_path][index_value] = result
    return result


if __name__ == "__main__":
    MAX = 1000
    BATCH_SIZE = 8
    embedder = TextEmbedder(device="cuda")

    # def load_prompts(file):
    #     datas = []
    #     for data in orjsonl.load(file):
    #         org_data = data["entry"]
    #         result1 = data["result1"]
    #         result2 = data["result2"]
    #         gt_prompt = remove_repeated_suffix(org_data["caption_llava"].strip())
    #         gen_prompt1 = result1["generated"]
    #         gen_prompt2 = result2["extended"]
    #         datas.append((gt_prompt, gen_prompt1, gen_prompt2))
    #     return datas

    # all_results = {}
    # gt = []
    # texts = [[], []]

    # datas = load_prompts("./data/coyo-output.jsonl")
    # for gt_prompt, gen_prompt1, gen_prompt2 in tqdm(
    #     datas, total=len(datas), desc="load"
    # ):
    #     gt.append(gt_prompt)
    #     texts[0].append(gen_prompt1)
    #     texts[1].append(gen_prompt2)

    # gt_embedding = embedder.encode(gt[:MAX], batch_size=BATCH_SIZE)

    # for idx, name in enumerate(["tipo gen", "tipo extend"]):
    #     emb = embedder.encode(texts[idx][:MAX], batch_size=BATCH_SIZE)
    #     all_results[name] = (
    #         kl_divergence(emb, gt_embedding),
    #         mmd(emb, gt_embedding),
    #         frechet_distance(emb, gt_embedding),
    #     )

    # def load_prompts(file):
    #     datas = []
    #     for data in orjsonl.load(file):
    #         org_data = data["entry"]
    #         index = org_data["key"]
    #         result1 = data["result1"]
    #         result2 = data["result2"]
    #         gt_prompt = remove_repeated_suffix(org_data["caption_llava"].strip())
    #         gen_prompt1 = result1
    #         gen_prompt2 = result2
    #         datas.append((gt_prompt, gen_prompt1, gen_prompt2))
    #     return datas

    # texts = [[], []]

    # datas = load_prompts("./data/generated_raw/coyo-output-promptist.jsonl")
    # for gt_prompt, gen_prompt1, gen_prompt2 in tqdm(
    #     datas, total=len(datas), desc="load"
    # ):
    #     texts[0].append(gen_prompt1)
    #     texts[1].append(gen_prompt2)

    # for idx, name in enumerate(["Promptist Short", "Promptist TLong"]):
    #     result = embedder.encode(texts[idx][:MAX], batch_size=BATCH_SIZE)
    #     all_results[name] = (
    #         kl_divergence(result, gt_embedding),
    #         mmd(result, gt_embedding),
    #         frechet_distance(result, gt_embedding),
    #     )

    # texts = [[], []]

    # datas = load_prompts("./data/generated_raw/coyo-output-gpt2.jsonl")
    # for gt_prompt, gen_prompt1, gen_prompt2 in tqdm(
    #     datas, total=len(datas), desc="load"
    # ):
    #     texts[0].append(gen_prompt1)
    #     texts[1].append(gen_prompt2)

    # for idx, name in enumerate(["GPT2 Short", "GPT2 TLong"]):
    #     result = embedder.encode(texts[idx][:MAX], batch_size=BATCH_SIZE)
    #     all_results[name] = (
    #         kl_divergence(result, gt_embedding),
    #         mmd(result, gt_embedding),
    #         frechet_distance(result, gt_embedding),
    #     )

    # texts = [[], []]

    # datas = load_prompts("./data/generated_raw/coyo-output-oai.jsonl")
    # for gt_prompt, gen_prompt1, gen_prompt2 in tqdm(
    #     datas, total=len(datas), desc="load"
    # ):
    #     texts[0].append(gen_prompt1)
    #     texts[1].append(gen_prompt2)

    # for idx, name in enumerate(["GPT4o Short", "GPT4o TLong"]):
    #     result = embedder.encode(texts[idx][:MAX], batch_size=BATCH_SIZE)
    #     all_results[name] = (
    #         kl_divergence(result, gt_embedding),
    #         mmd(result, gt_embedding),
    #         frechet_distance(result, gt_embedding),
    #     )

    # for name, result in all_results.items():
    #     print(name, result)

    # sys.exit(0)

    danbooru_meta = "./data/danbooru2023-prompt-gen-data.parquet"

    def load_prompts(file):
        datas = []
        for data in tqdm(orjsonl.load(file), total=MAX, desc="load"):
            org_data = data["entry"]
            index = org_data["index"]
            result = data["result1"]

            entry = get_row_by_index(danbooru_meta, int(index))
            short_caption = remove_repeated_suffix(entry["florence_short"])
            long_caption = remove_repeated_suffix(
                entry.get("pixtral_caption", None)
                or entry.get("phi3v_horny", None)
                or entry["florence_long"]
            )
            entry["generated"] = long_caption or short_caption
            data = {}
            for key, val in entry.items():
                data[key] = list(val) if isinstance(val, np.ndarray) else val
            org_prompt1 = apply_format(data, DEFAULT_FORMAT)

            datas.append((index, org_prompt1, apply_format(result, DEFAULT_FORMAT)))
            if len(datas) >= MAX:
                break
        return datas

    all_results = {}

    gt = []
    texts = []

    datas = load_prompts("./data/scenery-output.jsonl")
    for index, org_prompt1, gen_prompt1 in datas:
        gt.append(org_prompt1)
        texts.append(gen_prompt1)

    gt_embedding = embedder.encode(gt[:MAX], batch_size=BATCH_SIZE)
    result = embedder.encode(texts[:MAX], batch_size=BATCH_SIZE)
    all_results["TIPO"] = (
        # kl_divergence(result, gt_embedding),
        mmd(result, gt_embedding),
        frechet_distance(result, gt_embedding),
    )

    def load_prompts(file):
        datas = []
        for data in tqdm(orjsonl.load(file), total=MAX, desc="load"):
            org_data = data["entry"]
            index = org_data["index"]
            result = data["result"]

            entry = get_row_by_index(danbooru_meta, int(index))
            long_caption = remove_repeated_suffix(
                entry.get("phi3v_horny", None) or entry.get("florence_long", None)
            )
            if long_caption:
                entry["extended"] = long_caption
            data = {}
            for key, val in org_data.items():
                data[key] = list(val) if isinstance(val, np.ndarray) else val
            org_prompt1 = apply_format(data, DEFAULT_FORMAT)
            datas.append((index, org_prompt1, result))
            if len(datas) >= MAX:
                break
        return datas

    texts = []

    datas = load_prompts("./data/generated_raw/scenery-output-promptdb.jsonl")
    for index, org_prompt1, gen_prompt1 in datas:
        texts.append(gen_prompt1)

    result = embedder.encode(texts[:MAX], batch_size=BATCH_SIZE)
    all_results["PromptDB"] = (
        # kl_divergence(result, gt_embedding),
        mmd(result, gt_embedding),
        frechet_distance(result, gt_embedding),
    )

    texts = []

    datas = load_prompts("./data/generated_raw/scenery-output-oai.jsonl")
    for index, org_prompt1, gen_prompt1 in tqdm(datas, total=len(datas), desc="load"):
        texts.append(gen_prompt1)

    result = embedder.encode(texts[:MAX], batch_size=BATCH_SIZE)
    all_results["GPT4o-mini"] = (
        # kl_divergence(result, gt_embedding),
        mmd(result, gt_embedding),
        frechet_distance(result, gt_embedding),
    )

    texts = []

    datas = load_prompts("./data/generated_raw/scenery-output-promptist.jsonl")
    for index, org_prompt1, gen_prompt1 in datas:
        texts.append(gen_prompt1)

    result = embedder.encode(texts[:MAX], batch_size=BATCH_SIZE)
    all_results["Promptist"] = (
        # kl_divergence(result, gt_embedding),
        mmd(result, gt_embedding),
        frechet_distance(result, gt_embedding),
    )

    for name, result in all_results.items():
        print(name, result)
