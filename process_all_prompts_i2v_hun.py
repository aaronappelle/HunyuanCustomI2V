import os
import glob
from pathlib import Path
from loguru import logger
import torch
from einops import rearrange
import torch.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import argparse
from hymm_sp.config import parse_args, add_extra_args, sanity_check_args
from hymm_sp.sample_inference import HunyuanVideoSampler
from hymm_sp.data_kits.video_dataset import DataPreprocess
from hymm_sp.data_kits.data_tools import save_videos_grid
from hymm_sp.modules.parallel_states import (
    initialize_distributed,
    nccl_info,
)


def parse_custom_args(namespace=None):
    """Custom argument parser that adds parent_dir argument to the existing args"""
    parser = argparse.ArgumentParser(description="Hunyuan Multimodal batch processing script for start_frame.png files")
    parser = add_extra_args(parser)
    
    # Add our custom argument for parent directory
    parser.add_argument("--parent-dir", type=str, required=True,
                       help="Parent directory containing subdirectories with start_frame.png files")
    
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
    models_root_path = Path(args.ckpt)
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

    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(args.ckpt, args=args, device=device)
    # Get the updated args
    args = hunyuan_video_sampler.args

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
        prompt = args.pos_prompt
        negative_prompt = args.neg_prompt
        name = f"start_frame_{frame_info['subdir_name']}"
        seed = args.seed
        
        pixel_value_ref = pixel_value_ref * 2 - 1.
        pixel_value_ref_for_vae = rearrange(pixel_value_ref,"b c h w -> b c 1 h w")
        
        with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
            ref_latents = hunyuan_video_sampler.vae.encode(pixel_value_ref_for_vae.clone()).latent_dist.sample()
            uncond_ref_latents = hunyuan_video_sampler.vae.encode(torch.ones_like(pixel_value_ref_for_vae)).latent_dist.sample()
            ref_latents.mul_(hunyuan_video_sampler.vae.config.scaling_factor)
            uncond_ref_latents.mul_(hunyuan_video_sampler.vae.config.scaling_factor)

        # Handle prompts properly - build positive prompt
        prompt = args.add_pos_prompt + prompt
        
        # Handle negative prompt properly - only use if actually specified
        if negative_prompt.strip() or args.add_neg_prompt.strip():
            negative_prompt = args.add_neg_prompt + negative_prompt
        else:
            negative_prompt = None
        
        outputs = hunyuan_video_sampler.predict(
                prompt=prompt,
                name=name,
                size=args.video_size,
                seed=seed,
                pixel_value_llava=pixel_value_llava,
                uncond_pixel_value_llava=uncond_pixel_value_llava,
                ref_latents=ref_latents,
                uncond_ref_latents=uncond_ref_latents,
                video_length=args.sample_n_frames,
                guidance_scale=args.cfg_scale,
                num_images_per_prompt=args.num_images,
                negative_prompt=negative_prompt,
                infer_steps=args.infer_steps,
                flow_shift=args.flow_shift_eval_video,
                use_linear_quadratic_schedule=args.use_linear_quadratic_schedule,
                linear_schedule_end=args.linear_schedule_end,
                use_deepcache=args.use_deepcache,
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
