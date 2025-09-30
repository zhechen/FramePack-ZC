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
import os
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

from framepack import flf_helpers
from diffusers_helper.k_diffusion.uni_pc_fm import maybe_prepare_endpoints


@dataclass
class EndpointPipeline:
    vae: AutoencoderKLHunyuanVideo
    scheduler: Optional[object] = None


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
    parser.add_argument("--output", default="./outputs/flf.mp4", help="Where to save the resulting MP4")
    return parser.parse_args()


def setup_environment():
    root = Path(__file__).resolve().parent
    os.environ.setdefault("HF_HOME", str(root / "hf_download"))


def load_models(device: torch.device):
    text_encoder = LlamaModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)
    text_encoder_2 = CLIPTextModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", subfolder="text_encoder_2", torch_dtype=torch.float16
    ).to(device)
    tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="tokenizer_2")

    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", subfolder="vae", torch_dtype=torch.float16
    ).to(device)
    vae.eval()
    vae.requires_grad_(False)

    feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder="feature_extractor")
    image_encoder = SiglipVisionModel.from_pretrained(
        "lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=torch.float16
    ).to(device)
    image_encoder.eval()
    image_encoder.requires_grad_(False)

    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
        "lllyasviel/FramePack_F1_I2V_HY_20250503", torch_dtype=torch.bfloat16
    ).to(device)
    transformer.eval()
    transformer.requires_grad_(False)
    transformer.high_quality_fp32_output_for_inference = True

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


def encode_text(models, prompt: str, negative_prompt: str):
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

    return (
        llama_vec.to(models["transformer"].dtype),
        llama_mask,
        clip_pooler.to(models["transformer"].dtype),
        llama_neg.to(models["transformer"].dtype),
        llama_mask_neg,
        clip_neg.to(models["transformer"].dtype),
    )


def prepare_image_latent(models, input_path: str, height: int, width: int):
    image = np.array(Image.open(input_path).convert("RGB"))
    prepared = resize_and_center_crop(image, target_width=width, target_height=height)
    tensor = torch.from_numpy(prepared).float() / 127.5 - 1.0
    tensor = tensor.permute(2, 0, 1)[None, :, None]
    latent = vae_encode(tensor, models["vae"])
    return latent, prepared


def prepare_clip_embedding(models, prepared_np):
    output = hf_clip_vision_encode(prepared_np, models["feature_extractor"], models["image_encoder"])
    return output.last_hidden_state.to(models["transformer"].dtype)


def run_sampler(
    models,
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

    indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
    _, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split(
        [1, 16, 2, 1, latent_window_size], dim=1
    )
    clean_latent_indices = torch.cat([torch.zeros_like(clean_latent_1x_indices[:, :1]), clean_latent_1x_indices], dim=1)

    clean_latents_4x = start_latent[:, :, :16]
    clean_latents_2x = start_latent[:, :, 16:18]
    clean_latents_1x = start_latent[:, :, 18:19]

    concat_latents = torch.cat([start_latent[:, :, :1], clean_latents_1x], dim=2)

    endpoint_pipeline = EndpointPipeline(models["vae"])
    if endpoint_context is not None:
        endpoint_context["total_frames"] = frames

    result = sample_hunyuan(
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
        clean_latents=concat_latents,
        clean_latent_indices=clean_latent_indices,
        clean_latents_2x=clean_latents_2x,
        clean_latent_2x_indices=clean_latent_2x_indices,
        clean_latents_4x=clean_latents_4x,
        clean_latent_4x_indices=clean_latent_4x_indices,
        endpoint_context=endpoint_context,
        endpoint_pipeline=endpoint_pipeline,
    )

    return result, frames


def _format_duration(num_frames: int, fps: int) -> str:
    seconds = num_frames / float(fps)
    if num_frames % fps == 0:
        return f"{int(seconds)}s"
    return f"{seconds:.2f}s"

def decode_and_save(models, latents, output_path: str, fps: int = 30):
    duration_label = _format_duration(latents.shape[2], fps)
    pixels = vae_decode(latents, models["vae"]).cpu()
    save_bcthw_as_mp4(pixels, output_path, fps=fps, crf=16)
    print(f"Saved {duration_label} video to {output_path}")


def main():
    setup_environment()
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(device)

    llama_vec, llama_mask, clip_pooler, llama_neg, llama_mask_neg, clip_neg = encode_text(
        models, args.prompt, args.negative_prompt
    )

    # Align spatial resolution to the model buckets.
    input_image = Image.open(args.input_image).convert("RGB")
    bucket_h, bucket_w = find_nearest_bucket(*input_image.size[::-1], resolution=640)
    start_latent, prepared_np = prepare_image_latent(models, args.input_image, bucket_h, bucket_w)
    clip_vision = prepare_clip_embedding(models, prepared_np)

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
    decode_and_save(models, latents, args.output, fps=fps)



if __name__ == "__main__":
    main()
