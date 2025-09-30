"""FramePack inference with first/last frame anchoring.

Example:
    python framepack_flf_infer.py \
        --input-image path/to/seed.jpg \
        --prompt "A playful cat runs toward the camera" \
        --start-frame path/to/seed.jpg \
        --end-frame path/to/final.jpg \
        --endpoint-frames 12 \
        --endpoint-strength 0.7

The script mirrors the data flow of ``batch-f1-test.py`` but exposes
inference-time controls to keep the animation glued to explicit start and
end frames without any fine-tuning.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from diffusers import AutoencoderKLHunyuanVideo
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    LlamaModel,
    LlamaTokenizerFast,
    SiglipImageProcessor,
    SiglipVisionModel,
)

from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.utils import crop_or_pad_yield_mask, resize_and_center_crop, save_bcthw_as_mp4

HAS_CUDA = torch.cuda.is_available()

if HAS_CUDA:
    from diffusers_helper.memory import (
        DynamicSwapInstaller,
        cpu,
        fake_diffusers_current_device,
        get_cuda_free_memory_gb,
        gpu,
        load_model_as_complete,
        move_model_to_device_with_memory_preservation,
        offload_model_from_device_for_memory_preservation,
        unload_complete_models,
    )
else:
    cpu = torch.device("cpu")
    gpu = torch.device("cpu")

    class _NoOpDynamicSwapInstaller:
        @staticmethod
        def install_model(*args, **kwargs):
            return None

        @staticmethod
        def uninstall_model(*args, **kwargs):
            return None

    DynamicSwapInstaller = _NoOpDynamicSwapInstaller

    def fake_diffusers_current_device(*args, **kwargs):
        return None

    def get_cuda_free_memory_gb(*args, **kwargs):
        return 0.0

    def move_model_to_device_with_memory_preservation(*args, **kwargs):
        raise RuntimeError("CUDA is required for memory preservation moves.")

    def offload_model_from_device_for_memory_preservation(*args, **kwargs):
        raise RuntimeError("CUDA is required for memory preservation moves.")

    def unload_complete_models(*args, **kwargs):
        return None

    def load_model_as_complete(*args, **kwargs):
        raise RuntimeError("CUDA is required for memory preservation moves.")

from framepack import flf_helpers
from diffusers_helper.k_diffusion.uni_pc_fm import maybe_prepare_endpoints


@dataclass
class EndpointPipeline:
    vae: AutoencoderKLHunyuanVideo
    scheduler: Optional[object] = None


@dataclass
class RuntimeConfig:
    use_cuda: bool
    high_vram: bool
    device: torch.device
    cpu_device: torch.device = cpu
    gpu_memory_preservation: int = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FramePack inference with endpoint anchoring")
    parser.add_argument("--input-image", required=True, help="Path to the guiding input image")
    parser.add_argument("--prompt", required=True, help="Positive prompt")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt")
    parser.add_argument("--start-frame", dest="start_frame", help="Optional image that locks/blends the first frames")
    parser.add_argument("--end-frame", dest="end_frame", help="Optional image that guides the last frames")
    parser.add_argument("--endpoint-frames", type=int, default=8, help="How many frames are affected by the endpoints")
    parser.add_argument("--endpoint-strength", type=float, default=0.65, help="Blend strength for endpoint anchoring")
    parser.add_argument(
        "--endpoint-schedule",
        choices=("linear", "cosine"),
        default="cosine",
        help="How the blend ramps across the endpoint window",
    )
    parser.add_argument(
        "--endpoint-mode",
        choices=("blend", "lock"),
        default="blend",
        help="Lock replaces the first frames before denoising; blend mixes latents",
    )
    parser.add_argument("--two-pass-meet", action="store_true", help="Optionally run an end-anchored backward pass")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames to synthesise")
    parser.add_argument("--steps", type=int, default=8, help="Denoising steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--cfg", type=float, default=1.0, help="Real guidance scale")
    parser.add_argument("--gs", type=float, default=10.0, help="Distilled guidance scale")
    parser.add_argument("--rs", type=float, default=0.0, help="Guidance rescale factor")
    parser.add_argument(
        "--section-pause",
        type=float,
        default=0.0,
        help="Optional pause (in seconds) to insert between section samples to avoid GPU overload",
    )
    parser.add_argument("--output", default="./outputs/flf.mp4", help="Where to save the resulting MP4")
    return parser.parse_args()


def setup_environment():
    root = Path(__file__).resolve().parent
    os.environ.setdefault("HF_HOME", str(root / "hf_download"))


def load_models(runtime: RuntimeConfig):
    text_encoder = LlamaModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", subfolder="text_encoder", torch_dtype=torch.float16
    ).cpu()
    text_encoder_2 = CLIPTextModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", subfolder="text_encoder_2", torch_dtype=torch.float16
    ).cpu()
    tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="tokenizer_2")

    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", subfolder="vae", torch_dtype=torch.float16
    ).cpu()

    feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder="feature_extractor")
    image_encoder = SiglipVisionModel.from_pretrained(
        "lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=torch.float16
    ).cpu()

    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
        "lllyasviel/FramePack_F1_I2V_HY_20250503", torch_dtype=torch.bfloat16
    ).cpu()
    transformer.high_quality_fp32_output_for_inference = True

    transformer.to(dtype=torch.bfloat16)
    vae.to(dtype=torch.float16)
    image_encoder.to(dtype=torch.float16)
    text_encoder.to(dtype=torch.float16)
    text_encoder_2.to(dtype=torch.float16)

    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    image_encoder.eval()
    transformer.eval()

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    transformer.requires_grad_(False)

    if runtime.use_cuda and runtime.high_vram:
        text_encoder.to(runtime.device)
        text_encoder_2.to(runtime.device)
        image_encoder.to(runtime.device)
        vae.to(runtime.device)
        transformer.to(runtime.device)
    elif runtime.use_cuda:
        vae.enable_slicing()
        vae.enable_tiling()
        DynamicSwapInstaller.install_model(transformer, device=runtime.device)
        DynamicSwapInstaller.install_model(text_encoder, device=runtime.device)

    return {
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2,
        "vae": vae,
        "feature_extractor": feature_extractor,
        "image_encoder": image_encoder,
        "transformer": transformer,
    }


def encode_text(models, runtime: RuntimeConfig, prompt: str, negative_prompt: str):
    if runtime.use_cuda and not runtime.high_vram:
        fake_diffusers_current_device(models["text_encoder"], runtime.device)
        load_model_as_complete(models["text_encoder_2"], target_device=runtime.device)

    llama_vec, clip_pooler = encode_prompt_conds(
        prompt,
        models["text_encoder"],
        models["text_encoder_2"],
        models["tokenizer"],
        models["tokenizer_2"],
    )
    if negative_prompt:
        llama_neg, clip_neg = encode_prompt_conds(
            negative_prompt,
            models["text_encoder"],
            models["text_encoder_2"],
            models["tokenizer"],
            models["tokenizer_2"],
        )
    else:
        llama_neg, clip_neg = torch.zeros_like(llama_vec), torch.zeros_like(clip_pooler)

    llama_vec, llama_mask = crop_or_pad_yield_mask(llama_vec, length=512)
    llama_neg, llama_mask_neg = crop_or_pad_yield_mask(llama_neg, length=512)

    if runtime.use_cuda and not runtime.high_vram:
        unload_complete_models(models["text_encoder_2"])

    return (
        llama_vec.to(models["transformer"].dtype),
        llama_mask,
        clip_pooler.to(models["transformer"].dtype),
        llama_neg.to(models["transformer"].dtype),
        llama_mask_neg,
        clip_neg.to(models["transformer"].dtype),
    )


def prepare_image_latent(models, runtime: RuntimeConfig, input_path: str, height: int, width: int):
    image = np.array(Image.open(input_path).convert("RGB"))
    prepared = resize_and_center_crop(image, target_width=width, target_height=height)
    tensor = torch.from_numpy(prepared).float() / 127.5 - 1.0
    tensor = tensor.permute(2, 0, 1)[None, :, None]
    if runtime.use_cuda and not runtime.high_vram:
        load_model_as_complete(models["vae"], target_device=runtime.device)
        tensor = tensor.to(runtime.device)
    else:
        tensor = tensor.to(models["vae"].device)

    latent = vae_encode(tensor, models["vae"])

    if runtime.use_cuda and not runtime.high_vram:
        latent = latent.to(runtime.cpu_device)
        unload_complete_models(models["vae"])

    return latent, prepared


def prepare_clip_embedding(models, runtime: RuntimeConfig, prepared_np):
    if runtime.use_cuda and not runtime.high_vram:
        load_model_as_complete(models["image_encoder"], target_device=runtime.device)

    output = hf_clip_vision_encode(prepared_np, models["feature_extractor"], models["image_encoder"])
    hidden_state = output.last_hidden_state.to(models["transformer"].dtype)

    if runtime.use_cuda and not runtime.high_vram:
        hidden_state = hidden_state.to(runtime.cpu_device)
        unload_complete_models(models["image_encoder"])

    return hidden_state


def run_sampler(
    models,
    runtime: RuntimeConfig,
    args,
    llama_vec,
    llama_mask,
    llama_neg,
    llama_mask_neg,
    clip_pooler,
    clip_neg,
    clip_vision,
    start_latent,
    endpoint_context,
    height,
    width,
):
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    transformer = models["transformer"]

    latent_window_size = (args.frames + 3) // 4
    latent_window_size = max(latent_window_size, 4)
    frames = latent_window_size * 4 - 3
    context_length = sum([16, 2, 1])
    target_total_frames = max(args.frames, 1)
    total_latent_sections = max(math.ceil(max(args.frames - 1, 0) / latent_window_size), 1)

    if runtime.use_cuda and not runtime.high_vram:
        move_model_to_device_with_memory_preservation(
            transformer, target_device=runtime.device, preserved_memory_gb=runtime.gpu_memory_preservation
        )

    device = runtime.device if runtime.use_cuda else runtime.cpu_device

    llama_vec = llama_vec.to(device)
    llama_mask = llama_mask.to(device) if isinstance(llama_mask, torch.Tensor) else llama_mask
    llama_neg = llama_neg.to(device)
    llama_mask_neg = llama_mask_neg.to(device) if isinstance(llama_mask_neg, torch.Tensor) else llama_mask_neg
    clip_pooler = clip_pooler.to(device)
    clip_neg = clip_neg.to(device)
    clip_vision = clip_vision.to(device)
    start_latent = start_latent.to(device=device, dtype=torch.float32)

    indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size]), device=device).unsqueeze(0)
    (
        clean_latent_indices_start,
        clean_latent_4x_indices,
        clean_latent_2x_indices,
        clean_latent_1x_indices,
        latent_indices,
    ) = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
    clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

    history_latents = torch.zeros(
        start_latent.shape[0],
        start_latent.shape[1],
        context_length,
        height // 8,
        width // 8,
        dtype=torch.float32,
        device=device,
    )
    history_latents = torch.cat([history_latents, start_latent], dim=2)
    total_generated_latent_frames = 1
    start_anchor = start_latent[:, :, :1]

    endpoint_pipeline = EndpointPipeline(models["vae"])

    for section_index in range(total_latent_sections):
        if endpoint_context is not None:
            endpoint_context["frame_idx"] = min(total_generated_latent_frames - 1, target_total_frames - 1)
            endpoint_context["total_frames"] = target_total_frames

        clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -context_length:, :, :].split(
            [16, 2, 1], dim=2
        )
        clean_latents = torch.cat([start_anchor, clean_latents_1x], dim=2)

        generated_latents = sample_hunyuan(
            transformer=transformer,
            sampler="unipc",
            width=width,
            height=height,
            frames=frames,
            real_guidance_scale=args.cfg,
            distilled_guidance_scale=args.gs,
            guidance_rescale=args.rs,
            num_inference_steps=args.steps,
            generator=generator,
            prompt_embeds=llama_vec,
            prompt_embeds_mask=llama_mask,
            prompt_poolers=clip_pooler,
            negative_prompt_embeds=llama_neg,
            negative_prompt_embeds_mask=llama_mask_neg,
            negative_prompt_poolers=clip_neg,
            device=transformer.device,
            dtype=torch.bfloat16,
            image_embeddings=clip_vision,
            latent_indices=latent_indices,
            clean_latents=clean_latents,
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=clean_latents_2x,
            clean_latent_2x_indices=clean_latent_2x_indices,
            clean_latents_4x=clean_latents_4x,
            clean_latent_4x_indices=clean_latent_4x_indices,
            endpoint_context=endpoint_context,
            endpoint_pipeline=endpoint_pipeline,
        )

        history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)
        total_generated_latent_frames += int(generated_latents.shape[2])

        if runtime.use_cuda and args.section_pause > 0:
            torch.cuda.synchronize(device)
            time.sleep(args.section_pause)

        if total_generated_latent_frames >= target_total_frames:
            break

    latents = history_latents[:, :, -total_generated_latent_frames:, :, :]
    if target_total_frames > 0:
        latents = latents[:, :, :target_total_frames, :, :]

    if runtime.use_cuda and not runtime.high_vram:
        latents = latents.to(runtime.cpu_device)
        offload_model_from_device_for_memory_preservation(transformer, target_device=runtime.device, preserved_memory_gb=8)
        unload_complete_models()

    return latents, int(latents.shape[2])


def _format_duration(num_frames: int, fps: int) -> str:
    seconds = num_frames / float(fps)
    if num_frames % fps == 0:
        return f"{int(seconds)}s"
    return f"{seconds:.2f}s"

def decode_and_save(models, runtime: RuntimeConfig, latents, output_path: str, fps: int = 30):
    duration_label = _format_duration(latents.shape[2], fps)
    if runtime.use_cuda and not runtime.high_vram:
        load_model_as_complete(models["vae"], target_device=runtime.device)
        latents = latents.to(runtime.device)
    else:
        latents = latents.to(models["vae"].device)

    pixels = vae_decode(latents, models["vae"]).cpu()

    if runtime.use_cuda and not runtime.high_vram:
        unload_complete_models(models["vae"])

    save_bcthw_as_mp4(pixels, output_path, fps=fps, crf=16)
    print(f"Saved {duration_label} video to {output_path}")


def main():
    setup_environment()
    args = parse_args()

    use_cuda = HAS_CUDA
    device = gpu if use_cuda else cpu

    high_vram = True
    if use_cuda:
        free_mem_gb = get_cuda_free_memory_gb(device)
        high_vram = free_mem_gb > 60
        print(f"Free VRAM {free_mem_gb:.2f} GB")
        print(f"High-VRAM Mode: {high_vram}")

    runtime = RuntimeConfig(use_cuda=use_cuda, high_vram=high_vram, device=device)

    models = load_models(runtime)

    llama_vec, llama_mask, clip_pooler, llama_neg, llama_mask_neg, clip_neg = encode_text(
        models, runtime, args.prompt, args.negative_prompt
    )

    # Align spatial resolution to the model buckets.
    input_image = Image.open(args.input_image).convert("RGB")
    bucket_h, bucket_w = find_nearest_bucket(*input_image.size[::-1], resolution=640)
    start_latent, prepared_np = prepare_image_latent(models, runtime, args.input_image, bucket_h, bucket_w)
    clip_vision = prepare_clip_embedding(models, runtime, prepared_np)

    endpoint_cfg = {
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "endpoint_frames": args.endpoint_frames,
        "endpoint_strength": args.endpoint_strength,
        "endpoint_schedule": args.endpoint_schedule,
        "endpoint_mode": args.endpoint_mode,
        "two_pass_meet": args.two_pass_meet,
        "frame_idx": 0,
        "total_frames": args.frames,
    }
    scale = getattr(models["vae"].config, "scaling_factor", 0.18215)
    endpoint_context = None
    if args.start_frame or args.end_frame:
        endpoint_context = maybe_prepare_endpoints(
            EndpointPipeline(models["vae"]), endpoint_cfg, device=models["vae"].device, scale=scale, size_hw=(bucket_h, bucket_w)
        )

    forward_latents, _ = run_sampler(
        models,
        runtime,
        args,
        llama_vec,
        llama_mask,
        llama_neg,
        llama_mask_neg,
        clip_pooler,
        clip_neg,
        clip_vision,
        start_latent,
        endpoint_context,
        bucket_h,
        bucket_w,
    )

    latents = forward_latents

    if args.two_pass_meet and args.end_frame:
        backward_cfg = endpoint_cfg.copy()
        backward_cfg["start_frame"] = args.end_frame
        backward_cfg["end_frame"] = args.start_frame
        backward_context = maybe_prepare_endpoints(
            EndpointPipeline(models["vae"]), backward_cfg, device=models["vae"].device, scale=scale, size_hw=(bucket_h, bucket_w)
        )
        backward_latents, _ = run_sampler(
            models,
            runtime,
            args,
            llama_vec,
            llama_mask,
            llama_neg,
            llama_mask_neg,
            clip_pooler,
            clip_neg,
            clip_vision,
            start_latent,
            backward_context,
            bucket_h,
            bucket_w,
        )
        backward_latents = torch.flip(backward_latents, dims=[2])
        overlap = min(args.endpoint_frames, latents.shape[2])
        if overlap > 0:
            schedule = flf_helpers.make_schedule(overlap, args.endpoint_schedule).to(latents.device, latents.dtype)
            for i in range(overlap):
                alpha = args.endpoint_strength * schedule[i]
                latents[:, :, -overlap + i] = flf_helpers.anchor_x0(
                    latents[:, :, -overlap + i], backward_latents[:, :, -overlap + i], alpha
                )

    fps = 30
    duration_label = _format_duration(latents.shape[2], fps)
    print(f"Generated a {duration_label} clip with endpoint controls.")
    decode_and_save(models, runtime, latents, args.output, fps=fps)



if __name__ == "__main__":
    main()
