import cv2
import os
import torch
import argparse
import numpy as np

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

from flat_dataset.preprocess import load_raw, compute_gradient_confidence
from utils import iq2depth


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1"
    )
    parser.add_argument(
        "--depthcad_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--noise_IQ_file",
        type=str,
        default=None
    )
    parser.add_argument(
        "--noise_depth_file",
        type=str,
        default=None
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default=None
    )

    args = parser.parse_args()
    return args


def inference(pipe, noise, conf, scale):
    pred_IQs = np.zeros(noise.shape)

    for i in range(6):
        guidance = np.stack([noise[i], conf], axis=0)        
        guidance = torch.from_numpy(guidance).unsqueeze(0)

        prompt = ""

        # generate image
        generator = torch.manual_seed(42)
        pred_IQ = pipe(
            prompt, 
            num_inference_steps=20, 
            generator=generator, 
            image=guidance
        ).images[0]

        pred_IQ = np.nan_to_num(pred_IQ, nan=0, neginf=0, posinf=0)
        pred_IQ = np.mean(np.array(pred_IQ), axis=2) / 255.0    # convert to (0, 1)
        pred_IQ = 2 * pred_IQ - 1   # (-1, 1)
        pred_IQs[i] = pred_IQ * scale

    target_shape = [6, 424, 512]
    reshaped_IQs = np.zeros(target_shape, dtype=np.float32)
    for i in range(6):
        reshaped_IQs[i, :, :] = cv2.resize(pred_IQs[i, :, :], (512, 424), interpolation=cv2.INTER_LINEAR)
    
    return reshaped_IQs


if __name__ == '__main__':
    args = parse_args()
    base_model_path = args.pretrained_model_name_or_path
    depthcad_path = args.depthcad_path
    noise_file = args.noise_IQ_file
    noise_depth_file = args.noise_depth_file
    out_file = args.out_file
    
    # load data
    noise = load_raw(noise_file)
    scale = max(noise.max(), abs(noise.min()))
    noise /= scale

    noise_depth = np.load(noise_depth_file)
    noise_depth = cv2.resize(noise_depth, (512, 512))
    confidence = compute_gradient_confidence(noise_depth)

    amplitudes = np.stack(
        np.fromfile(noise_file, dtype=np.float32).reshape([424, 512, 9])[:, :, [2, 5, 8]],
        axis=2
    )

    # load pipe
    depthcad = ControlNetModel.from_pretrained(depthcad_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, controlnet=depthcad, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    # Inference
    pred_IQs = inference(pipe, noise, confidence, scale)
    depth = iq2depth(pred_IQs, amplitudes)
    np.save(out_file, depth)
