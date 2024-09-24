import math
from functools import partial
from time import time
from PIL import Image

import numpy as np
import torch
from diffusers import StableDiffusionXLKDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from k_diffusion.external import CompVisDenoiser
from k_diffusion.sampling import get_sigmas_polyexponential
from k_diffusion.sampling import sample_dpmpp_2m_sde

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


def load_model(model_id="KBlueLeaf/Kohaku-XL-Zeta", device="cuda"):
    vae: AutoencoderKL = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix").to(device)
    pipe: StableDiffusionXLKDiffusionPipeline
    pipe = StableDiffusionXLKDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, vae=vae
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
    return pipe


@torch.no_grad()
def encode_prompts(
    pipe: StableDiffusionXLKDiffusionPipeline,
    prompt: str | list[str],
    neg_prompt: str | list[str] = "",
    cutoff_length: int | None = 225,
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
    length = max(input_ids.input_ids.size(-1), input_ids2.input_ids.size(-1))
    target_length = cutoff_length or math.ceil(length / max_length) * max_length + 2

    input_ids = pipe.tokenizer(
        prompts, padding="max_length", max_length=target_length, return_tensors="pt"
    ).input_ids
    input_ids = (
        input_ids[:, 0:1],
        input_ids[:, 1:-1],
        input_ids[:, -1:],
    )
    input_ids2 = pipe.tokenizer_2(
        prompts, padding="max_length", max_length=target_length, return_tensors="pt"
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
        if i == 0:
            concat_embeds2.append(hidden_states.hidden_states[-2][:, :-1])
        elif i == input_ids2[1].shape[-1] - max_length:
            concat_embeds2.append(hidden_states.hidden_states[-2][:, 1:])
        else:
            concat_embeds2.append(hidden_states.hidden_states[-2][:, 1:-1])

    prompt_embeds = torch.cat(concat_embeds, dim=1)
    prompt_embeds2 = torch.cat(concat_embeds2, dim=1)
    prompt_embeds = torch.cat([prompt_embeds, prompt_embeds2], dim=-1)

    pooled_embeds2 = torch.mean(torch.stack(pooled_embeds2, dim=0), dim=0)

    return prompt_embeds.chunk(2), pooled_embeds2.chunk(2)


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
):
    pipe.scheduler.set_timesteps(num_inference_steps)
    unet: CompVisDenoiser = pipe.k_diffusion_model
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

    def cfg_wrapper(x, sigma, sigma_cond=None):
        if sigma_cond is not None:
            sigma_cond = torch.cat([sigma_cond] * 2)
        cond, uncond = unet(
            torch.cat([x] * 2),
            torch.cat([sigma] * 2),
            cond=text_ctx,
            added_cond_kwargs=added_cond,
        ).chunk(2)
        cfg_output = uncond + guidance_scale * (cond - uncond)
        return cfg_output

    sigmas = pipe.scheduler.sigmas
    x0 = (
        torch.randn(
            (1, 4, height // 8, width // 8),
        ).to(prompt_embeds).repeat(prompt_embeds.size(0), 1, 1, 1)
        * sigmas[0]
    )
    result = sample_dpmpp_2m_sde(cfg_wrapper, x0, sigmas, eta=0.35)
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
    prompt = """
1girl,
king halo (umamusume), umamusume,

ogipote, misu kasumi, fuzichoco, ningen mame, ask (askzy), maccha (mochancc),

solo, leaning forward, cleavage, sky, outdoors, black bikini, stomach, swimsuit, navel, 

masterpiece, newest, absurdres, sensitive
""".strip()
    # sdxl_pipe = load_model("KBlueLeaf/xxx")
    sdxl_pipe = load_model()
    t0 = time()
    for _ in range(10):
        with torch.autocast("cuda"):
            (prompt_embeds, neg_prompt_embeds), (pooled_embeds2, neg_pooled_embeds2) = (
                encode_prompts(sdxl_pipe, [prompt] * 3, "")
            )
            result = generate(
                sdxl_pipe,
                prompt_embeds,
                neg_prompt_embeds,
                pooled_embeds2,
                neg_pooled_embeds2,
                num_inference_steps=24,
                width=1024,
                height=1024,
                guidance_scale=6.0,
            )[0]
    t1 = time()
    print((t1 - t0) / 100)

    result.save("output/test.png")
