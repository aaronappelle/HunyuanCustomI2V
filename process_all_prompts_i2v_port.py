import os
import glob
from pathlib import Path
import re
from loguru import logger
import torch
from einops import rearrange
import torch.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import argparse
# from hymm_sp.config import parse_args, add_extra_args, sanity_check_args
# from hymm_sp.sample_inference import HunyuanVideoSampler
import sys
# Add path /hpc/home/aba41/diffusion/HunyuanVideo-I2V to sources
from hymm_sp.sample_inference_i2v import HunyuanVideoSampler
sys.path.append("/hpc/home/aba41/diffusion/HunyuanVideo-I2V")
# from hyvideo.inference import HunyuanVideoSampler
# from hyvideo.config import parse_args, add_extra_models_args, sanity_check_args, add_network_args, add_denoise_schedule_args, add_i2v_args, add_inference_args, add_training_args, add_optimizer_args, add_deepspeed_args, add_data_args, add_train_denoise_schedule_args
# add_lora_args  add_parallel_args, 

from hymm_sp.data_kits.video_dataset import DataPreprocess
from hymm_sp.data_kits.data_tools import save_videos_grid
from hymm_sp.modules.parallel_states import (
    initialize_distributed,
    nccl_info,
)


# def parse_custom_args(namespace=None):
#     """Custom argument parser that adds parent_dir argument to the existing args"""
#     parser = argparse.ArgumentParser(description="Hunyuan Multimodal batch processing script for start_frame.png files")
#     parser = add_extra_args(parser)
    
#     # Add our custom argument for parent directory
#     parser.add_argument("--parent-dir", type=str, required=True,
#                        help="Parent directory containing subdirectories with start_frame.png files")
    
#     args = parser.parse_args(namespace=namespace)
#     args = sanity_check_args(args)
#     return args

def add_denoise_schedule_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Denoise schedule args")

    group.add_argument(
        "--denoise-type",
        type=str,
        default="flow",
        help="Denoise type for noised inputs.",
    )

    # Flow Matching
    group.add_argument(
        "--flow-shift",
        type=float,
        default=17.0,
        help="Shift factor for flow matching schedulers.",
    )
    group.add_argument(
        "--flow-reverse",
        action="store_true",
        help="If reverse, learning/sampling from t=1 -> t=0.",
    )
    group.add_argument(
        "--flow-solver",
        type=str,
        default="euler",
        help="Solver for flow matching.",
    )
    group.add_argument(
        "--use-linear-quadratic-schedule",
        action="store_true",
        help="Use linear quadratic schedule for flow matching."
        "Following MovieGen (https://ai.meta.com/static-resource/movie-gen-research-paper)",
    )
    group.add_argument(
        "--linear-schedule-end",
        type=int,
        default=25,
        help="End step for linear quadratic schedule for flow matching.",
    )

    return parser

PRECISIONS = {"fp32", "fp16", "bf16"}
HUNYUAN_VIDEO_CONFIG = {
    "HYVideo-T/2": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
    },
    "HYVideo-T/2-cfgdistill": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "guidance_embed": True,
    },
    "HYVideo-S/2": {
        "mm_double_blocks_depth": 6,
        "mm_single_blocks_depth": 12,
        "rope_dim_list": [12, 42, 42],
        "hidden_size": 480,
        "heads_num": 5,
        "mlp_width_ratio": 4,
    },
}
TEXT_PROJECTION = {
    "linear",                               # Default, an nn.Linear() layer
    "single_refiner",                       # Single TokenRefiner. Refer to LI-DiT
}

# When using decoder-only models, we must provide a prompt template to instruct the text encoder
# on how to generate the text.
# --------------------------------------------------------------------
PROMPT_TEMPLATE_ENCODE = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
) 
PROMPT_TEMPLATE_ENCODE_VIDEO = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)  

PROMPT_TEMPLATE_ENCODE_I2V = (
    "<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

PROMPT_TEMPLATE_ENCODE_VIDEO_I2V = (
    "<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the video by detailing the following aspects according to the reference image: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>\n\n"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

PROMPT_TEMPLATE = {
    "dit-llm-encode": {
        "template": PROMPT_TEMPLATE_ENCODE,
        "crop_start": 36,
    },
    "dit-llm-encode-video": {
        "template": PROMPT_TEMPLATE_ENCODE_VIDEO,
        "crop_start": 95,
    },
    "dit-llm-encode-i2v": {
        "template": PROMPT_TEMPLATE_ENCODE_I2V,
        "crop_start": 36,
        "image_emb_start": 5,
        "image_emb_end": 581,
        "image_emb_len": 576,
        "double_return_token_id": 271
    },
    "dit-llm-encode-video-i2v": {
        "template": PROMPT_TEMPLATE_ENCODE_VIDEO_I2V,
        "crop_start": 103,
        "image_emb_start": 5,
        "image_emb_end": 581,
        "image_emb_len": 576,
        "double_return_token_id": 271
    },
}

# 3D VAE
MODEL_BASE = os.getenv("MODEL_BASE", "./ckpts")
VAE_PATH = {"884-16c-hy": f"{MODEL_BASE}/hunyuan-video-i2v-720p/vae"}
print(f"VAE_PATH: {VAE_PATH}")

# Text Encoder
TEXT_ENCODER_PATH = {
    "clipL": f"{MODEL_BASE}/text_encoder_2",
    "llm": f"{MODEL_BASE}/text_encoder",
    "llm-i2v": f"{MODEL_BASE}/text_encoder_i2v",
}
print(f"TEXT_ENCODER_PATH: {TEXT_ENCODER_PATH}")

# Tokenizer
TOKENIZER_PATH = {
    "clipL": f"{MODEL_BASE}/text_encoder_2",
    "llm": f"{MODEL_BASE}/text_encoder",
    "llm-i2v": f"{MODEL_BASE}/text_encoder_i2v",
}
print(f"TOKENIZER_PATH: {TOKENIZER_PATH}")

def add_network_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="HunyuanVideo network args")

    # Main model
    group.add_argument(
        "--model",
        type=str,
        choices=list(HUNYUAN_VIDEO_CONFIG.keys()),
        default="HYVideo-T/2-cfgdistill",
    )
    group.add_argument(
        "--latent-channels",
        type=str,
        default=16,
        help="Number of latent channels of DiT. If None, it will be determined by `vae`. If provided, "
        "it still needs to match the latent channels of the VAE model.",
    )
    group.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=PRECISIONS,
        help="Precision mode. Options: fp32, fp16, bf16. Applied to the backbone model and optimizer.",
    )

    # RoPE
    group.add_argument(
        "--rope-theta", type=int, default=256, help="Theta used in RoPE."
    )

    group.add_argument("--gradient-checkpoint", action="store_true",
                       help="Enable gradient checkpointing to reduce memory usage.")

    group.add_argument("--gradient-checkpoint-layers", type=int, default=-1,
                       help="Number of layers to checkpoint. -1 for all layers. `n` for the first n layers.")

    return parser


def add_extra_models_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Extra models args, including vae, text encoders and tokenizers)"
    )

    # - VAE
    group.add_argument(
        "--vae",
        type=str,
        default="884-16c-hy",
        choices=list(VAE_PATH),
        help="Name of the VAE model.",
    )
    group.add_argument(
        "--vae-precision",
        type=str,
        default="fp16",
        choices=PRECISIONS,
        help="Precision mode for the VAE model.",
    )
    group.add_argument(
        "--vae-tiling",
        action="store_true",
        help="Enable tiling for the VAE model to save GPU memory.",
    )
    group.set_defaults(vae_tiling=True)
    group.add_argument(
        "--vae-weight",
        type=str,
        default=VAE_PATH["884-16c-hy"],
        help="Path to the VAE model.",
    )

    group.add_argument(
        "--text-encoder",
        type=str,
        default="llm-i2v",
        choices=list(TEXT_ENCODER_PATH),
        help="Name of the text encoder model.",
    )
    group.add_argument(
        "--text-encoder-path",
        type=str,
        default=TEXT_ENCODER_PATH["llm-i2v"],
        help="Path to the text encoder model.",
    )
    group.add_argument(
        "--text-encoder-precision",
        type=str,
        default="fp16",
        choices=PRECISIONS,
        help="Precision mode for the text encoder model.",
    )
    group.add_argument(
        "--text-states-dim",
        type=int,
        default=4096,
        help="Dimension of the text encoder hidden states.",
    )
    group.add_argument(
        "--text-len", type=int, default=256, help="Maximum length of the text input."
    )
    group.add_argument(
        "--tokenizer",
        type=str,
        default="llm-i2v",
        choices=list(TOKENIZER_PATH),
        help="Name of the tokenizer model.",
    )
    group.add_argument(
        "--tokenizer-path",
        type=str,
        default=TOKENIZER_PATH["llm-i2v"],
        help="Path to the tokenizer model.",
    )
    group.add_argument(
        "--prompt-template",
        type=str,
        default="dit-llm-encode-i2v",
        choices=PROMPT_TEMPLATE,
        help="Image prompt template for the decoder-only text encoder model.",
    )
    group.add_argument(
        "--prompt-template-video",
        type=str,
        default="dit-llm-encode-video-i2v",
        choices=PROMPT_TEMPLATE,
        help="Video prompt template for the decoder-only text encoder model.",
    )
    group.add_argument(
        "--hidden-state-skip-layer",
        type=int,
        default=2,
        help="Skip layer for hidden states.",
    )
    group.add_argument(
        "--apply-final-norm",
        action="store_true",
        help="Apply final normalization to the used text encoder hidden states.",
    )

    # - CLIP
    group.add_argument(
        "--text-encoder-2",
        type=str,
        default="clipL",
        choices=list(TEXT_ENCODER_PATH),
        help="Name of the second text encoder model.",
    )
    group.add_argument(
        "--text-encoder-2-path",
        type=str,
        default=TEXT_ENCODER_PATH["clipL"],
        help="Path to the second text encoder model.",
    )
    group.add_argument(
        "--text-encoder-precision-2",
        type=str,
        default="fp16",
        choices=PRECISIONS,
        help="Precision mode for the second text encoder model.",
    )
    group.add_argument(
        "--text-states-dim-2",
        type=int,
        default=768,
        help="Dimension of the second text encoder hidden states.",
    )
    group.add_argument(
        "--tokenizer-2",
        type=str,
        default="clipL",
        choices=list(TOKENIZER_PATH),
        help="Name of the second tokenizer model.",
    )
    group.add_argument(
        "--tokenizer-2-path",
        type=str,
        default=TOKENIZER_PATH["clipL"],
        help="Path to the second tokenizer model.",
    )
    group.add_argument(
        "--text-len-2",
        type=int,
        default=77,
        help="Maximum length of the second text input.",
    )
    group.add_argument("--text-projection", type=str, default="single_refiner", choices=TEXT_PROJECTION,
                       help="A projection layer for bridging the text encoder hidden states and the diffusion model "
                            "conditions.")
    group.set_defaults(use_attention_mask=True)

    return parser

def add_i2v_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="I2V args")

    group.add_argument(
        "--i2v-mode",
        action="store_true",
        help="Whether to open i2v mode."
    )

    group.add_argument(
        "--i2v-resolution",
        type=str,
        default="720p",
        choices=["720p", "540p", "360p"],
        help="Resolution for i2v inference."
    )

    group.add_argument(
        "--i2v-image-path",
        type=str,
        default="./assets/demo/i2v/imgs/0.png",
        help="Image path for i2v inference."
    )

    group.add_argument(
        "--i2v-condition-type",
        type=str,
        default="token_replace",
        choices=["token_replace", "latent_concat"],
        help="Condition type for i2v model."
    )

    group.add_argument(
        "--i2v-stability", action="store_true", help="Whether to use i2v stability mode."
    )

    return parser

def add_inference_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Inference args")

    # ======================== Model loads ========================
    group.add_argument(
        "--model-base",
        type=str,
        default="ckpts",
        help="Root path of all the models, including t2v models and extra models.",
    )
    group.add_argument(
        "--dit-weight",
        type=str,
        default="ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
        help="Path to the HunyuanVideo model. If None, search the model in the args.model_root."
        "1. If it is a file, load the model directly."
        "2. If it is a directory, search the model in the directory. Support two types of models: "
        "1) named `pytorch_model_*.pt`"
        "2) named `*_model_states.pt`, where * can be `mp_rank_00`.",
    )
    group.add_argument(
        "--i2v-dit-weight",
        type=str,
        default="ckpts/hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt",
        help="Path to the HunyuanVideo model. If None, search the model in the args.model_root."
        "1. If it is a file, load the model directly."
        "2. If it is a directory, search the model in the directory. Support two types of models: "
        "1) named `pytorch_model_*.pt`"
        "2) named `*_model_states.pt`, where * can be `mp_rank_00`.",
    )
    group.add_argument(
        "--model-resolution",
        type=str,
        default="540p",
        choices=["540p", "720p"],
        help="Root path of all the models, including t2v models and extra models.",
    )
    group.add_argument(
        "--load-key",
        type=str,
        default="module",
        help="Key to load the model states. 'module' for the main model, 'ema' for the EMA model.",
    )
    group.add_argument(
        "--use-cpu-offload",
        action="store_true",
        help="Use CPU offload for the model load.",
    )
    group.add_argument("--cpu-offload", action="store_true", help="Use CPU offload for the model load.")

    # ======================== Inference general setting ========================
    group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference and evaluation.",
    )
    group.add_argument(
        "--infer-steps",
        type=int,
        default=50,
        help="Number of denoising steps for inference.",
    )
    group.add_argument(
        "--disable-autocast",
        action="store_true",
        help="Disable autocast for denoising loop and vae decoding in pipeline sampling.",
    )
    group.add_argument(
        "--save-path",
        type=str,
        default="./results",
        help="Path to save the generated samples.",
    )
    group.add_argument(
        "--save-path-suffix",
        type=str,
        default="",
        help="Suffix for the directory of saved samples.",
    )
    group.add_argument(
        "--name-suffix",
        type=str,
        default="",
        help="Suffix for the names of saved samples.",
    )
    group.add_argument(
        "--num-videos",
        type=int,
        default=1,
        help="Number of videos to generate for each prompt.",
    )
    # ---sample size---
    group.add_argument(
        "--video-size",
        type=int,
        nargs="+",
        default=(720, 1280),
        help="Video size for training. If a single value is provided, it will be used for both height "
        "and width. If two values are provided, they will be used for height and width "
        "respectively.",
    )
    group.add_argument(
        "--video-length",
        type=int,
        default=129,
        help="How many frames to sample from a video. if using 3d vae, the number should be 4n+1",
    )
    # --- prompt ---
    group.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt for sampling during evaluation.",
    )
    group.add_argument(
        "--seed-type",
        type=str,
        default="fixed", #"auto",
        choices=["file", "random", "fixed", "auto"],
        help="Seed type for evaluation. If file, use the seed from the CSV file. If random, generate a "
        "random seed. If fixed, use the fixed seed given by `--seed`. If auto, `csv` will use the "
        "seed column if available, otherwise use the fixed `seed` value. `prompt` will use the "
        "fixed `seed` value.",
    )
    group.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")

    # Classifier-Free Guidance
    group.add_argument(
        "--neg-prompt", type=str, default=None, help="Negative prompt for sampling."
    )
    group.add_argument(
        "--cfg-scale", type=float, default=1.0, help="Classifier free guidance scale."
    )
    group.add_argument(
        "--embedded-cfg-scale",
        type=float,
        default=None,
        help="Embeded classifier free guidance scale.",
    )

    group.add_argument(
        "--use-fp8",
        action="store_true",
        help="Enable use fp8 for inference acceleration."
    )

    group.add_argument(
        "--reproduce",
        action="store_true",
        help="Enable reproducibility by setting random seeds and deterministic algorithms.",
    )

    return parser

def sanity_check_args(args):
    # VAE channels
    vae_pattern = r"\d{2,3}-\d{1,2}c-\w+"
    if not re.match(vae_pattern, args.vae):
        raise ValueError(
            f"Invalid VAE model: {args.vae}. Must be in the format of '{vae_pattern}'."
        )
    vae_channels = int(args.vae.split("-")[1][:-1])
    if args.latent_channels is None:
        args.latent_channels = vae_channels
    if vae_channels != args.latent_channels:
        raise ValueError(
            f"Latent channels ({args.latent_channels}) must match the VAE channels ({vae_channels})."
        )
    return args

def parse_custom_args(namespace=None):
    parser = argparse.ArgumentParser(description="HunyuanVideo inference/lora training script")

    parser = add_network_args(parser)
    parser = add_extra_models_args(parser)
    parser = add_denoise_schedule_args(parser)
    parser = add_i2v_args(parser)
    # parser = add_lora_args(parser)
    parser = add_inference_args(parser)
    # parser = add_parallel_args(parser)
    # if mode == "train":
    #     parser = add_training_args(parser)
    #     parser = add_optimizer_args(parser)
    #     parser = add_deepspeed_args(parser)
    #     parser = add_data_args(parser)
    #     parser = add_train_denoise_schedule_args(parser)
    parser.add_argument("--parent-dir", type=str, required=True,
                        help="Parent directory containing subdirectories with start_frame.png files")
    # parser.add_argument("--ckpt", type=str, required=True,
    #                     help="Checkpoint path")
    args = parser.parse_args(namespace=namespace)
    args = sanity_check_args(args)

    return args


def find_start_frame_images(parent_dir):
    """Find all start_frame.png files in subdirectories of parent_dir"""
    parent_path = Path(parent_dir)
    if not parent_path.exists():
        raise ValueError(f"Parent directory does not exist: {parent_dir}")
    
    start_frame_files = []
    for subdir in parent_path.iterdir():
        if subdir.is_dir():
            start_frame_path = subdir / "start_frame.png"
            if start_frame_path.exists():
                start_frame_files.append({
                    'image_path': str(start_frame_path),
                    'subdir_name': subdir.name,
                    'full_subdir_path': str(subdir)
                })
    
    # Sort by subdirectory name for consistent processing order
    start_frame_files.sort(key=lambda x: x['subdir_name'])
    
    logger.info(f"Found {len(start_frame_files)} start_frame.png files in subdirectories of {parent_dir}")
    for item in start_frame_files:
        logger.info(f"  - {item['subdir_name']}: {item['image_path']}")
    
    return start_frame_files


def main():
    args = parse_custom_args()
    args.flow_shift_eval_video = args.flow_shift # for compatibility with HunyuanVideo-I2V
    print(args.i2v_dit_weight)
    # models_root_path = Path(args.ckpt)
    models_root_path = Path(args.model_base)
    print("*"*20) 
    initialize_distributed(args.seed)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    print("+"*20)
    
    # Find all start_frame.png files
    start_frame_files = find_start_frame_images(args.parent_dir)
    if not start_frame_files:
        logger.warning(f"No start_frame.png files found in subdirectories of {args.parent_dir}")
        return
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Determine the scene name from parent_dir for use in filenames
    parent_dir_basename = os.path.basename(args.parent_dir)
    if parent_dir_basename.endswith("_frames"):
        parsed_scene_name = parent_dir_basename[:-len("_frames")]
    else:
        parsed_scene_name = parent_dir_basename # Use basename if no _frames suffix

    # Load models
    rank = 0
    vae_dtype = torch.float16
    device = torch.device("cuda")
    if nccl_info.sp_size > 1:
        device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        rank = torch.distributed.get_rank()

    # hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(args.ckpt, args=args, device=device)
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args, device=device)
    # Get the updated args
    # args = hunyuan_video_sampler.args

    # Initialize data preprocessor
    data_preprocessor = DataPreprocess()
    
    # Process each start_frame.png file
    for i, frame_info in enumerate(start_frame_files):
        logger.info(f"Processing {i+1}/{len(start_frame_files)}: {frame_info['subdir_name']}")
        
        # Load and preprocess the image
        batch = data_preprocessor.get_batch(frame_info['image_path'], args.video_size)
        
        pixel_value_llava = batch['pixel_value_llava'].to(device)
        pixel_value_ref = batch['pixel_value_ref'].to(device)
        uncond_pixel_value_llava = batch['uncond_pixel_value_llava'].to(device)
        
        # Use the provided prompts from command line args
        # prompt = args.pos_prompt
        prompt = args.prompt
        negative_prompt = args.neg_prompt
        name = f"start_frame_{frame_info['subdir_name']}"
        # seed = args.seed
        
        pixel_value_ref = pixel_value_ref * 2 - 1.
        pixel_value_ref_for_vae = rearrange(pixel_value_ref,"b c h w -> b c 1 h w")
        
        with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
            ref_latents = hunyuan_video_sampler.vae.encode(pixel_value_ref_for_vae.clone()).latent_dist.sample()
            uncond_ref_latents = hunyuan_video_sampler.vae.encode(torch.ones_like(pixel_value_ref_for_vae)).latent_dist.sample()
            ref_latents.mul_(hunyuan_video_sampler.vae.config.scaling_factor)
            uncond_ref_latents.mul_(hunyuan_video_sampler.vae.config.scaling_factor)

        # Handle prompts properly - build positive prompt
        # prompt = args.add_pos_prompt + prompt
        
        # Handle negative prompt properly - only use if actually specified
        # if negative_prompt.strip() or args.add_neg_prompt.strip():
        #     negative_prompt = args.add_neg_prompt + negative_prompt
        # else:
        #     negative_prompt = None
        
        # outputs = hunyuan_video_sampler.predict(
        #         prompt=prompt,
        #         name=name,
        #         size=args.video_size,
        #         seed=seed,
        #         pixel_value_llava=pixel_value_llava,
        #         uncond_pixel_value_llava=uncond_pixel_value_llava,
        #         ref_latents=ref_latents,
        #         uncond_ref_latents=uncond_ref_latents,
        #         video_length=args.sample_n_frames,
        #         guidance_scale=args.cfg_scale,
        #         num_images_per_prompt=args.num_images,
        #         negative_prompt=negative_prompt,
        #         infer_steps=args.infer_steps,
        #         flow_shift=args.flow_shift_eval_video,
        #         use_linear_quadratic_schedule=args.use_linear_quadratic_schedule,
        #         linear_schedule_end=args.linear_schedule_end,
        #         use_deepcache=args.use_deepcache,
        # )
        print(f"Inference on image {frame_info['image_path']}")
        outputs = hunyuan_video_sampler.predict(
            prompt=prompt,
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=args.seed,
            negative_prompt=negative_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale,
            i2v_mode=args.i2v_mode,
            i2v_resolution=args.i2v_resolution,
            i2v_image_path=frame_info['image_path'], #args.i2v_image_path,
            i2v_condition_type=args.i2v_condition_type,
            i2v_stability=args.i2v_stability,
            name=name
            # ulysses_degree=args.ulysses_degree,
            # ring_degree=args.ring_degree,
        )

        if rank == 0:
            samples = outputs['samples']
            for j, sample_tensor in enumerate(samples):
                current_sample_to_save = sample_tensor.unsqueeze(0)

                # Construct the new descriptive filename
                scene_for_filename = parsed_scene_name 
                frame_identifier = "start" # Identifies the input as "start_frame.png"
                resolution_str = f"{args.video_size[0]}x{args.video_size[1]}"
                # Using subdir_name as the unique [filename] part, as it identifies the input source
                input_file_identifier = frame_info['subdir_name'] 
                seed_str = f"seed{args.seed}"

                output_filename = f"i2v-hunyuan-{scene_for_filename}-{frame_identifier}-{resolution_str}-{input_file_identifier}-{seed_str}.mp4"
                
                # All videos saved directly into save_path (which is OUTPUT_BASEPATH)
                out_path = os.path.join(save_path, output_filename)
                
                save_videos_grid(current_sample_to_save, out_path, fps=25)
                logger.info(f'Sample saved to: {out_path}')

    logger.info(f"Finished processing all {len(start_frame_files)} start_frame.png files")

    
if __name__ == "__main__":
    main()
