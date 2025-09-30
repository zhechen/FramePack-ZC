from diffusers_helper.hf_login import login
import os

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math

import cv2
import glob
import shutil
import PIL

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

import time

def extract_last_frame(video_path):
    """Extract the last frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)  # Move to last frame
    ret, frame = cap.read()
    cap.release()
    cv2.imwrite("./outputs/last.jpg", frame)
    return frame[:,:,::-1]

parser = argparse.ArgumentParser()
args = parser.parse_args()

generated_dir = './outputs/'
#stitched_output = './outputs/zv00.mp4' #156
#if os.path.exists(stitched_output):
#    os.remove(stitched_output)

im_path = '16-00112.jpg'
start_id = 0

#extract_last_frame("./outputs/v050.mp4")
#im_path = './outputs/last.jpg'

import random
seed=random.randint(0, 1000)

prompts = ['Pretty thin Chinese teen girl, grins, shakes, lifts legs, sways gently, clear movements, stays centered.']   

durlens = [18, 10] #* 2 #, 15] #, 15] #8, 15, 9] #

#print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

#transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePack_F1_I2V_HY_20250503', torch_dtype=torch.bfloat16).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

stream = AsyncStream()

@torch.no_grad()
def worker(input_image, prompt, n_prompt, durlen=20):
    total_second_length=durlen
    latent_window_size=9 
    cfg=1.0
    gs=10.0 
    rs=0.0
    gpu_memory_preservation=6
    #"""
    steps=12
    use_teacache=False  
    """
    steps=8
    use_teacache=True  
    #"""
    mp4_crf=16

    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4) # 30
    total_latent_sections = int(max(round(total_latent_sections), 1))

    print('Starting ...')

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding

        #stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))
        print('Text encoding ...')

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image

        #stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))
        print('Image processing ...')

        input_image = np.array(input_image)
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        Image.fromarray(input_image_np).save(os.path.join(generated_dir, f'test.jpg'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        #stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))
        print('VAE encoding ...')

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision

        #stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
        print('CLIP Vision encoding ...')

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling

        #stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))
        print('Start sampling ...')

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))

        #if total_latent_sections > 4:
        #    # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
        #    # items looks better than expanding it when total_latent_sections > 4
        #    # One can try to remove below trick and just
        #    # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
        #    latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
        total_generated_latent_frames = 1

        for section_index in range(total_latent_sections):
            time.sleep(0.5)

            print(f'section_index = {section_index}, total_latent_sections = {total_latent_sections}')

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
            clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=latent_window_size * 4 - 3,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                # shift=3.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                #callback=callback,
            )

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = latent_window_size * 2
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], vae).cpu()
                history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(generated_dir, f'{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf) #30

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            #stream.output_queue.push(('file', output_filename))

            #if is_last_section:
            #    break
    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    #stream.output_queue.push(('end', None))
    #return


def resize_image(image, max_size = -1, pad=False):
    # Get original dimensions
    original_width, original_height = image.size

    if max_size < 0:
        max_size = max(original_width, original_height)

    # Calculate the scaling factor while preserving the aspect ratio
    scaling_factor = min(max_size / original_width, max_size / original_height)

    new_width = int(original_width * scaling_factor / 8) * 8
    new_height = int(original_height * scaling_factor / 8) * 8
    # Resize the image
    resized_image = image.resize((new_width, new_height), PIL.Image.LANCZOS)

    if not pad:
        return new_width, new_height, resized_image
    else:
        max_dim = max(new_width, new_height)
        resized_image = ImageOps.pad(resized_image, (max_dim, max_dim), color='white')
        return max_dim, max_dim, resized_image

def get_latest_video_file(directory, extension='mp4'):
    """Find the latest generated video file in the directory."""
    list_of_files = glob.glob(os.path.join(directory, f'*.{extension}'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def rename_video_file(src_path, dst_directory, clip_number):
    """Rename the latest generated video in a sequential order."""
    dst_path = os.path.join(dst_directory, f"clip_{clip_number:03d}.mp4")
    shutil.move(src_path, dst_path)
    print(f"Renamed {src_path} to {dst_path}")
    return dst_path

def smart_sharpen(image, sharp=0.1, threshold=5, sat=1.0, denoise=True, alpha=1.0, beta=0, lab_a=0, lab_b=0):
    """
    Applies a smart sharpen filter to an image.
    
    Args:
        image_path (str): Path to the input image.
        sharpness (float): Amount of sharpening to apply. Higher values = more sharpening.
        threshold (int): Threshold for edge enhancement. Lower values = more areas sharpened.

    Returns:
        np.ndarray: The sharpened image.
    """
    image = np.array(image)
    if sharp > 0.:
        # Split the image into individual color channels
        channels = cv2.split(image)
        sharpened_channels = []
        # Process each channel separately
        for channel in channels:
            channel = channel.astype(np.float32)

            # Step 1: Denoise using Bilateral Filter
            if denoise:
                denoised_channel = cv2.bilateralFilter(channel, d=5, sigmaColor=5, sigmaSpace=5)
            else:
                denoised_channel = channel
            
            # Step 2: Extract Details for Sharpening (Unsharp Mask)
            blurred = cv2.GaussianBlur(denoised_channel, (0, 0), sigmaX=1.2, sigmaY=1.2)

            # Difference of Gaussian (DoG)
            details = denoised_channel - blurred
            
            # Thresholding the details to remove noise
            details[np.abs(details) < threshold] = 0
            
            # Enhance edges by adding scaled details back
            sharpened_channel = cv2.addWeighted(denoised_channel, 0.99, details, sharp, 0)
            
            # Clip values to valid range
            sharpened_channel = np.clip(sharpened_channel, 0, 255).astype(np.uint8)
            
            sharpened_channels.append(sharpened_channel)
        
        # Merge the processed channels back into a color image
        img = cv2.merge(sharpened_channels)
    else:
        img = image.copy()

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Split HSV channels
    h, s, v = cv2.split(hsv)
    
    # Scale the saturation channel
    s = np.clip(s * sat, 0, 255).astype(np.uint8)
    
    # Merge back the HSV channels
    hsv_adjusted = cv2.merge([h, s, v])
    
    # Convert back to BGR color space
    img_adjusted = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)

    ###
    # Increase brightness
    brightened = cv2.convertScaleAbs(img_adjusted, alpha=alpha, beta=beta)

    if abs(lab_a) > 0 or abs(lab_b) > 0:
        # Convert to LAB color space to enhance redness
        lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Increase the A-channel to enhance red tones
        a = cv2.add(a, lab_a)  # Increase red component (tune value as needed)
        b = cv2.add(b, lab_b)  # Increase red component (tune value as needed)

        # Merge and convert back to BGR
        lab_enhanced = cv2.merge((l, a, b))
        final_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    else:
        final_image = brightened
    
    return Image.fromarray(final_image)

def reverse_video(input_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'

    # Read all frames into a list
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    # Write frames in reverse order
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in reversed(frames):
        out.write(frame)

    out.release()
    print(f"Reversed video saved to {output_path}")

#os.system('rm ./outputs/final_video.mp4')
n_prompt = 'Still video, blurrs, fingers, public hair'

im_dir = './out-r1'
counter = 0
for filename in os.listdir(im_dir):
    if not filename.lower().endswith('.jpg'):
        continue

    im_path = os.path.join(im_dir, filename)
    input_image = Image.open(im_path).convert('RGB')
    _, _, input_image = resize_image(input_image, max_size=700)

    #input_image = extract_last_frame("./outputs/clip_000.mp4")
    #cv2.imwrite(save_frame_path, input_image)

    for pi in range(start_id, len(prompts)):
        cur_prompt = prompts[pi] + ' Preserve face identity. '
        print(cur_prompt)

        input_image = smart_sharpen(input_image, sharp=0., sat=0.99, alpha=0.98, beta=1, lab_a=-1, lab_b=0)
        worker(input_image, cur_prompt, n_prompt, durlen=durlens[pi])

        # After EACH video generation:
        latest_video = get_latest_video_file(generated_dir)

        if latest_video:
            ## 1. Rename the latest video
            #if pi == 0:
            #    reverse_video(latest_video, latest_video[:-4] + '-o.mp4')
            #    video_path = latest_video[:-4] + '-o.mp4'
            #else:
            video_path = latest_video
        
            renamed_video = rename_video_file(video_path, generated_dir, counter)
            counter = counter + 1

            ## 2. Extract last frame
            #if "stroll" in prompts[pi]:
            #    input_image = extract_last_frame(renamed_video)
        else:
            print("No new video file found.")

    time.sleep(5)
