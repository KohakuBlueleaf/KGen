import math
from functools import partial
from time import time
from PIL import Image

import numpy as np
import torch
from tqdm import tqdm, trange
from diffusers import (
    StableDiffusionXLKDiffusionPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
)
from k_diffusion.external import CompVisDenoiser
from k_diffusion.sampling import get_sigmas_polyexponential
from k_diffusion.sampling import (
    sample_dpmpp_2m_sde,
    sample_euler,
    sample_euler_ancestral,
    sample_heun,
)

torch.set_float32_matmul_precision("medium")


def set_timesteps_polyexponential(
    self, orig_sigmas, num_inference_steps, device=None, rho=None
):
    self.num_inference_steps = num_inference_steps

    self.sigmas = get_sigmas_polyexponential(
        num_inference_steps + 1,
        sigma_min=orig_sigmas[-2],
        sigma_max=orig_sigmas[0],
        rho=rho or 0.666666,
        device=device or "cpu",
    )
    self.sigmas = torch.cat([self.sigmas[:-2], self.sigmas.new_zeros([1])])


def set_timesteps_exponential(self, orig_sigmas, num_inference_steps, device=None):
    return set_timesteps_polyexponential(
        self, orig_sigmas, num_inference_steps, device, 1.0
    )


def set_timesteps_linear(self, orig_sigmas, num_inference_steps, device=None):
    self.num_inference_steps = num_inference_steps
    index = (
        torch.linspace(0, orig_sigmas.numel() - 1, num_inference_steps).round().long()
    )
    self.sigmas = torch.cat([orig_sigmas[index], self.sigmas.new_zeros([1])])


def model_forward(k_diffusion_model: torch.nn.Module):
    orig_forward = k_diffusion_model.forward

    def forward(*args, **kwargs):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            result = orig_forward(*args, **kwargs)
        return result.float()

    return forward


def load_model(model_id="KBlueLeaf/Kohaku-XL-Zeta", device="cuda", custom_vae=False):
    if custom_vae:
        vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix"
        ).to(device)
        pipe: StableDiffusionXLKDiffusionPipeline
        pipe = StableDiffusionXLKDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, vae=vae
        ).to(device)
    else:
        pipe: StableDiffusionXLKDiffusionPipeline
        pipe = StableDiffusionXLKDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device)
    pipe.vae.config.force_upcast = False
    pipe.vae.eval().half()
    pipe.text_encoder.eval().half()
    pipe.text_encoder_2.eval().half()
    pipe.k_diffusion_model.eval().half().to(device)
    unet: UNet2DConditionModel = pipe.k_diffusion_model.inner_model.model
    unet.eval().half()
    unet.enable_xformers_memory_efficient_attention()
    pipe.scheduler.set_timesteps = partial(
        set_timesteps_exponential, pipe.scheduler, pipe.scheduler.sigmas
    )
    # pipe.scheduler.set_timesteps = partial(
    #     set_timesteps_linear, pipe.scheduler, pipe.scheduler.sigmas
    # )
    pipe.sampler = partial(sample_dpmpp_2m_sde, eta=0.35, solver_type="heun")
    pipe.k_diffusion_model.forward = model_forward(pipe.k_diffusion_model)

    pipe.vae = torch.compile(pipe.vae, mode="default")
    pipe.text_encoder = torch.compile(pipe.text_encoder, mode="default")
    pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2, mode="default")
    pipe.k_diffusion_model.inner_model.model = torch.compile(
        pipe.k_diffusion_model.inner_model.model, mode="default"
    )

    torch.cuda.empty_cache()
    return pipe


@torch.no_grad()
def encode_prompts(
    pipe: StableDiffusionXLKDiffusionPipeline,
    prompt: str | list[str],
    neg_prompt: str | list[str] = "",
    cutoff_length: int | None = 225,
    padding_to_max_length: bool = True,
    take_all_eos: bool = False,
):
    if not isinstance(prompt, list):
        prompt = [prompt]
    if not isinstance(neg_prompt, list):
        neg_prompt = [neg_prompt]
    if len(prompt) != len(neg_prompt) and (len(prompt) != 1 and len(neg_prompt) != 1):
        raise ValueError("prompt and neg_prompt must have the same length")
    if len(prompt) == 1:
        prompt = prompt * len(neg_prompt)
    if len(neg_prompt) == 1:
        neg_prompt = neg_prompt * len(prompt)
    prompts = prompt + neg_prompt
    max_length = pipe.tokenizer.model_max_length - 2

    input_ids = pipe.tokenizer(prompts, padding=True, return_tensors="pt")
    input_ids2 = pipe.tokenizer_2(prompts, padding=True, return_tensors="pt")
    input_ids_neg = pipe.tokenizer(neg_prompt, padding=True, return_tensors="pt")
    length = max(input_ids.input_ids.size(-1), input_ids2.input_ids.size(-1))
    neg_length = input_ids_neg.input_ids.size(-1)
    target_length = cutoff_length or math.ceil(length / max_length) * max_length + 2
    neg_groups = math.ceil(neg_length / max_length)

    input_ids = pipe.tokenizer(
        prompts,
        padding="max_length",
        max_length=target_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    input_ids = (
        input_ids[:, 0:1],
        input_ids[:, 1:-1],
        input_ids[:, -1:],
    )
    input_ids2 = pipe.tokenizer_2(
        prompts,
        padding="max_length",
        max_length=target_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    input_ids2 = (
        input_ids2[:, 0:1],
        input_ids2[:, 1:-1],
        input_ids2[:, -1:],
    )

    concat_embeds = []
    for i in range(0, input_ids[1].shape[-1], max_length):
        input_id1 = torch.concat(
            (input_ids[0], input_ids[1][:, i : i + max_length], input_ids[2]), dim=-1
        ).to(pipe.device)
        result = pipe.text_encoder(input_id1, output_hidden_states=True).hidden_states[
            -2
        ]
        if take_all_eos:
            concat_embeds.append(result)
            continue
        if i == 0:
            concat_embeds.append(result[:, :-1])
        elif i == input_ids[1].shape[-1] - max_length:
            concat_embeds.append(result[:, 1:])
        else:
            concat_embeds.append(result[:, 1:-1])

    concat_embeds2 = []
    pooled_embeds2 = []
    for i in range(0, input_ids2[1].shape[-1], max_length):
        input_id2 = torch.concat(
            (input_ids2[0], input_ids2[1][:, i : i + max_length], input_ids2[2]), dim=-1
        ).to(pipe.device)
        hidden_states = pipe.text_encoder_2(input_id2, output_hidden_states=True)
        pooled_embeds2.append(hidden_states[0])
        if take_all_eos:
            concat_embeds2.append(hidden_states.hidden_states[-2])
            continue
        if i == 0:
            concat_embeds2.append(hidden_states.hidden_states[-2][:, :-1])
        elif i == input_ids2[1].shape[-1] - max_length:
            concat_embeds2.append(hidden_states.hidden_states[-2][:, 1:])
        else:
            concat_embeds2.append(hidden_states.hidden_states[-2][:, 1:-1])

    prompt_embeds = torch.cat(concat_embeds, dim=1)
    prompt_embeds2 = torch.cat(concat_embeds2, dim=1)
    prompt_embeds = torch.cat([prompt_embeds, prompt_embeds2], dim=-1)

    pooled = torch.mean(torch.stack(pooled_embeds2, dim=0), dim=0)

    embed, neg_embed = prompt_embeds.chunk(2)
    pooled, neg_pooled = pooled.chunk(2)
    if not padding_to_max_length:
        if take_all_eos:
            neg_embed = torch.cat([concat_embeds[0][1:], concat_embeds2[0][1:]], dim=-1)
        else:
            neg_embed = torch.cat(
                [neg_embed[:, : neg_groups * max_length + 1, :], neg_embed[:, -1:, :]],
                dim=1,
            )
        neg_pooled_embeds = torch.stack(
            [emb.chunk(2)[-1] for emb in pooled_embeds2][:neg_groups]
        )
        neg_pooled = torch.mean(neg_pooled_embeds, dim=0)

    return (embed, neg_embed), (pooled, neg_pooled)


def vae_image_postprocess(image_tensor: torch.Tensor) -> Image.Image:
    image = Image.fromarray(
        ((image_tensor * 0.5 + 0.5) * 255)
        .cpu()
        .clamp(0, 255)
        .numpy()
        .astype(np.uint8)
        .transpose(1, 2, 0)
    )
    return image


class DDIMInversionCallback:
    """
    Automatic DDIMInversionCallback
    for Z-Sampling
    it will inverse back to previous sigma than sampling back to current sigma
    """

    def __init__(self, model, sigmas: list, inverse_cfg: float, extra_args=None):
        self.model = model
        self.sigmas = sigmas
        self.inverse_cfg = inverse_cfg
        self.extra_args = {} if extra_args is None else extra_args

    def __call__(self, infos):
        xt = infos["x"]
        denoised = infos["denoised"]
        sigma = infos["sigma"]
        current_sigma_index = self.sigmas.index(sigma)
        if current_sigma_index in {0, len(self.sigmas) - 1}:
            return

        noise = (xt - denoised) / sigma
        x_inverse = denoised + self.sigmas[current_sigma_index - 1] * noise
        new_denoised = self.model(
            x_inverse,
            self.sigmas[current_sigma_index - 1],
            cfg_override=self.inverse_cfg,
            **self.extra_args,
        )


@torch.no_grad()
def sample_ddim(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, s_in * sigmas[i], **extra_args)
        noise = (x - denoised) / sigmas[i]
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "denoised": denoised})
        # go back to xt
        x = denoised + sigmas[i + 1] * noise
    return x


@torch.no_grad()
def generate(
    pipe: StableDiffusionXLKDiffusionPipeline,
    prompt_embeds,
    negative_prompt_embeds,
    pooled_prompt_embeds,
    negative_pooled_prompt_embeds,
    num_inference_steps=24,
    width=1024,
    height=1024,
    guidance_scale=6.0,
    z_sample=False,
):
    pipe.scheduler.set_timesteps(num_inference_steps)
    unet: CompVisDenoiser = pipe.k_diffusion_model

    if prompt_embeds.shape == negative_prompt_embeds.shape:
        time_ids = (
            torch.tensor([height, width, 0, 0, height, width])
            .repeat(2 * prompt_embeds.size(0), 1)
            .to(prompt_embeds)
        )
        added_cond = {
            "time_ids": time_ids,
            "text_embeds": torch.concat(
                [pooled_prompt_embeds, negative_pooled_prompt_embeds]
            ),
        }
        text_ctx = torch.cat([prompt_embeds, negative_prompt_embeds])

        def cfg_wrapper(x, sigma, cfg_override=None):
            cond, uncond = unet(
                torch.cat([x] * 2),
                torch.cat([sigma] * 2),
                cond=text_ctx,
                added_cond_kwargs=added_cond,
            ).chunk(2)
            if cfg_override is not None:
                if isinstance(cfg_override, list):
                    cfg = cfg_override.pop(0)
                else:
                    cfg = float(cfg_override)
            else:
                cfg = guidance_scale
            cfg_output = uncond + cfg * (cond - uncond)
            return cfg_output

    else:
        time_ids = torch.tensor([height, width, 0, 0, height, width]).to(prompt_embeds)
        added_cond_pos = {"time_ids": time_ids, "text_embeds": pooled_prompt_embeds}
        added_cond_neg = {
            "time_ids": time_ids,
            "text_embeds": negative_pooled_prompt_embeds,
        }

        def cfg_wrapper(x, sigma, cfg_override=None):
            cond = unet(x, sigma, cond=prompt_embeds, added_cond_kwargs=added_cond_pos)
            uncond = unet(
                x, sigma, cond=negative_prompt_embeds, added_cond_kwargs=added_cond_neg
            )
            if cfg_override is not None:
                if isinstance(cfg_override, list):
                    cfg = cfg_override.pop(0)
                else:
                    cfg = float(cfg_override)
            else:
                cfg = guidance_scale
            cfg_output = uncond + cfg * (cond - uncond)
            return cfg_output

    sigmas = list(pipe.scheduler.sigmas)
    cfg_lists = [guidance_scale] * len(sigmas)
    x0 = (
        torch.randn(
            (1, 4, height // 8, width // 8),
        )
        .to(prompt_embeds)
        .repeat(prompt_embeds.size(0), 1, 1, 1)
        * sigmas[0]
    )

    # 5, 4, 3, 2, 1, 0
    # 5, 4, *5*, 3, *4*, 2, *3*, 1, 0
    # 5, 4, *5*, **4**, 3, *4*, **3**, 2, *3*, **2**, 1, 0
    # sigmas[:1] + interleave(sigmas[1:], sigmas[:-1]) + sigmas[-1:]
    def interleave(*lists):
        return sum((list(x) for x in zip(*lists)), [])

    if z_sample:
        sigmas = (
            sigmas[:2]
            + interleave(sigmas[:-3], sigmas[1:-2], sigmas[2:-1])
            + sigmas[-2:]
        )
        cfg_lists = (
            cfg_lists[:1]
            + interleave(
                [max(guidance_scale * 0.2, 1.5) for _ in cfg_lists[:-3]],
                cfg_lists[:-3],
                cfg_lists[:-3],
            )
            + cfg_lists[-3:]
        )
    sigmas = torch.tensor(sigmas)

    result = sample_euler(
        cfg_wrapper,
        x0,
        sigmas.to(prompt_embeds.device),
        extra_args={"cfg_override": cfg_lists},
    )
    result /= pipe.vae.config.scaling_factor
    image_tensors = []
    for latent in result:
        image_tensors.append(pipe.vae.decode(latent.unsqueeze(0).half()).sample)
    image_tensors = torch.concat(image_tensors)
    images = []
    for image_tensor in image_tensors:
        images.append(vae_image_postprocess(image_tensor))
    return images


if __name__ == "__main__":
    from lightning.pytorch import seed_everything

    prompt = """
1girl,
king halo (umamusume), umamusume,

fujisaki hikari, ninjin nouka, quasarcake, welchino, hazuki natsu, sho (sho lwlw), 
yuzuki gao, naga u, usashiro mani, azumi kazuki, 
masterpiece, newest, best quality, absurdres, safe,

solo, leaning forward, cleavage, sky, outdoors, black bikini, stomach, swimsuit, navel, 
collarbone, beach, brown eyes, horse ears, cloud, cloudy sky, medium breasts, water, 
bikini under clothes, horizon, bikini, long sleeves, day, looking at viewer, breasts, 
animal ears, jacket, ear covers, horse girl, smile, brown hair, blue sky, open mouth, 
green skirt, frills, skirt, ocean, 

masterpiece, newest, absurdres, sensitive
""".strip()

    neg_prompt = """
low quality, worst quality, text, signature, jpeg artifacts, bad anatomy, 
old, early, copyright name, watermark, artist name, signature
"""

    steps = 32
    sdxl_pipe = load_model()
    (prompt_embeds, neg_prompt_embeds), (pooled_embeds2, neg_pooled_embeds2) = (
        encode_prompts(
            sdxl_pipe,
            [prompt] * 3,
            [neg_prompt] * 3,
            padding_to_max_length=True,
            take_all_eos=True,
        )
    )

    seed = torch.randint(0, 2**31 - 1, []).item()
    print(seed)

    with torch.autocast("cuda"):
        seed_everything(seed)
        result = generate(
            sdxl_pipe,
            prompt_embeds,
            neg_prompt_embeds,
            pooled_embeds2,
            neg_pooled_embeds2,
            num_inference_steps=steps,
            width=1024,
            height=1024,
            guidance_scale=7.0,
        )[0]

    result.save("output/test.png")
    with torch.autocast("cuda"):
        seed_everything(seed)
        result = generate(
            sdxl_pipe,
            prompt_embeds,
            neg_prompt_embeds,
            pooled_embeds2,
            neg_pooled_embeds2,
            num_inference_steps=(steps - 3) // 3 + 2 + bool(steps % 3),
            width=1024,
            height=1024,
            guidance_scale=7.0,
            z_sample=True,
        )[0]

    result.save("output/test-z.png")
