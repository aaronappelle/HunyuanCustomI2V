import os
import torch

__all__ = [
    "PROMPT_TEMPLATE", "MODEL_BASE", "PRECISION_TO_TYPE",
    "PRECISIONS", "VAE_PATH", "TEXT_ENCODER_PATH", "TOKENIZER_PATH",
    "TEXT_PROJECTION",
]

# =================== Constant Values =====================

PRECISION_TO_TYPE = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

PROMPT_TEMPLATE_ENCODE_VIDEO = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)

PROMPT_TEMPLATE_ENCODE = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|eot_id|>"
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

# ORIGINAL:
# PROMPT_TEMPLATE = {
#     "li-dit-encode-video": {"template": PROMPT_TEMPLATE_ENCODE_VIDEO, "crop_start": 95},
# }

# NEW FROM HunyuanVideo-I2V
# PROMPT_TEMPLATE = {
#     "li-dit-encode-video": {"template": PROMPT_TEMPLATE_ENCODE_VIDEO, "crop_start": 95},
# }

PROMPT_TEMPLATE = {
    "li-dit-encode-video": {"template": PROMPT_TEMPLATE_ENCODE_VIDEO, "crop_start": 95},
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

# ======================= Model ======================
PRECISIONS = {"fp32", "fp16", "bf16"}

# =================== Model Path =====================
MODEL_BASE = os.getenv("MODEL_BASE")

# 3D VAE
VAE_PATH = {
    "884-16c-hy0801": f"{MODEL_BASE}/vae_3d/hyvae_v1_0801",
}

# Text Encoder

# ORIGINAL
# TEXT_ENCODER_PATH = {
#     "clipL": f"{MODEL_BASE}/openai_clip-vit-large-patch14",
#     "llava-llama-3-8b": f"{MODEL_BASE}/llava-llama-3-8b-v1_1",
# }
TEXT_ENCODER_PATH = {
    "clipL": f"{MODEL_BASE}/openai_clip-vit-large-patch14",
    "llava-llama-3-8b": f"{MODEL_BASE}/llava-llama-3-8b-v1_1",
    "llm-i2v": f"{MODEL_BASE}/text_encoder_i2v",
}

# Tokenizer
# ORIGINAL
# TOKENIZER_PATH = {
#     "clipL": f"{MODEL_BASE}/openai_clip-vit-large-patch14",
#     "llava-llama-3-8b": f"{MODEL_BASE}/llava-llama-3-8b-v1_1",
# }
TOKENIZER_PATH = {
    "clipL": f"{MODEL_BASE}/openai_clip-vit-large-patch14",
    "llava-llama-3-8b": f"{MODEL_BASE}/llava-llama-3-8b-v1_1",
    "llm-i2v": f"{MODEL_BASE}/text_encoder_i2v",
}

TEXT_PROJECTION = {
    "linear",                               # Default, an nn.Linear() layer
    "single_refiner",                       # Single TokenRefiner. Refer to LI-DiT
}

# NEW FROM HunyuanVideo-I2V

# Flow Matching path type
FLOW_PATH_TYPE = {
    "linear",               # Linear trajectory between noise and data
    "gvp",                  # Generalized variance-preserving SDE
    "vp",                   # Variance-preserving SDE
}

# Flow Matching predict type
FLOW_PREDICT_TYPE = {
    "velocity",             # Predict velocity
    "score",                # Predict score
    "noise",                # Predict noise
}

# Flow Matching loss weight
FLOW_LOSS_WEIGHT = {
    "velocity",             # Weight loss by velocity
    "likelihood",           # Weight loss by likelihood
}

# Flow Matching SNR type
FLOW_SNR_TYPE = {
    "lognorm",              # Log-normal SNR
    "uniform",              # Uniform SNR
}

# Flow Matching solvers
FLOW_SOLVER = {
    "euler",                # Euler solver
}