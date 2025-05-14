<!-- ## **HunyuanCustom** -->

<p align="center">
  <img src="assets/material/logo.png"  height=100>
</p>

# **HunyuanCustom** 🌅
 
<div align="center">
  <a href="https://github.com/Tencent/HunyuanCustom"><img src="https://img.shields.io/static/v1?label=HunyuanCustom%20Code&message=Github&color=blue"></a> &ensp;
  <a href="https://hunyuancustom.github.io/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green"></a> &ensp;
  <a href="https://hunyuan.tencent.com/modelSquare/home/play?modelId=192"><img src="https://img.shields.io/static/v1?label=Playground&message=Web&color=green"></a>
</div>
<div align="center">
  <a href="https://arxiv.org/pdf/2505.04512"><img src="https://img.shields.io/static/v1?label=Tech Report&message=Arxiv&color=red"></a> &ensp;
</div>
<div align="center">
  <a href="https://huggingface.co/tencent/HunyuanCustom"><img src="https://img.shields.io/static/v1?label=HunyuanVideo&message=HuggingFace&color=yellow"></a> &ensp;
</div>

> [**HunyuanCustom: A Multimodal-Driven Architecture for Customized Video Generation**](https://arxiv.org/pdf/2505.04512) <be>



## 🔥🔥🔥 News!!
* May 13, 2025: 🎉 HunyuanCustom has been integrated into [ComfyUI-HunyuanVideoWrapper](https://github.com/kijai/ComfyUI-HunyuanVideoWrapper/blob/develop/example_workflows/hyvideo_custom_testing_01.json) by [Kijai](https://github.com/kijai).
* May 12, 2025: 🔥 HunyuanCustom is available in Cloud-Native-Build (CNB) [HunyuanCustom](https://cnb.cool/tencent/hunyuan/HunyuanCustom).
* May  8, 2025: 👋 We release the inference code and model weights of HunyuanCustom. [Download](models/README.md).


## 📑 Open-source Plan

- HunyuanCustom
  - Single-Subject Video Customization
    - [x] Inference 
    - [x] Checkpoints
    - [x] ComfyUI
  - Audio-Driven Video Customization
  - Video-Driven Video Customization
  - Multi-Subject Video Customization

## Contents
- [**HunyuanCustom** 🌅](#hunyuancustom-)
  - [🔥🔥🔥 News!!](#-news)
  - [📑 Open-source Plan](#-open-source-plan)
  - [Contents](#contents)
  - [**Abstract**](#abstract)
  - [**HunyuanCustom Overall Architecture**](#hunyuancustom-overall-architecture)
  - [🎉 **HunyuanCustom Key Features**](#-hunyuancustom-key-features)
    - [**Multimodal Video customization**](#multimodal-video-customization)
    - [**Various Applications**](#various-applications)
  - [📈 Comparisons](#-comparisons)
  - [📜 Requirements](#-requirements)
  - [🛠️ Dependencies and Installation](#️-dependencies-and-installation)
    - [Installation Guide for Linux](#installation-guide-for-linux)
  - [🧱 Download Pretrained Models](#-download-pretrained-models)
  - [🚀 Parallel Inference on Multiple GPUs](#-parallel-inference-on-multiple-gpus)
  - [🔑 Single-gpu Inference](#-single-gpu-inference)
    - [Run with very low VRAM](#run-with-very-low-vram)
  - [Run a Gradio Server](#run-a-gradio-server)
  - [🔗 BibTeX](#-bibtex)
  - [Acknowledgements](#acknowledgements)
---

## **Abstract**

Customized video generation aims to produce videos featuring specific subjects under flexible user-defined conditions, yet existing methods often struggle with identity consistency and limited input modalities. In this paper, we propose HunyuanCustom, a multi-modal customized video generation framework that emphasizes subject consistency while supporting image, audio, video, and text conditions. Built upon HunyuanVideo, our model first addresses the image-text conditioned generation task by introducing a text-image fusion module based on LLaVA for enhanced multi-modal understanding, along with an image ID enhancement module that leverages temporal concatenation to reinforce identity features across frames. To enable audio- and video-conditioned generation, we further propose modality-specific condition injection mechanisms: an AudioNet module that achieves hierarchical alignment via spatial cross-attention, and a video-driven injection module that integrates latent-compressed conditional video through a patchify-based feature-alignment network. Extensive experiments on single- and multi-subject scenarios demonstrate that HunyuanCustom significantly outperforms state-of-the-art open- and closed-source methods in terms of ID consistency, realism, and text-video alignment. Moreover, we validate its robustness across downstream tasks, including audio and video-driven customized video generation. Our results highlight the effectiveness of multi-modal conditioning and identity-preserving strategies in advancing controllable video generation.

## **HunyuanCustom Overall Architecture**

![image](assets/material/method.png)

We propose **HunyuanCustom, a multi-modal, conditional, and controllable generation model centered on subject consistency**, built upon the Hunyuan Video generation framework. It enables the generation of subject-consistent videos conditioned on text, images, audio, and video inputs. 

## 🎉 **HunyuanCustom Key Features**

### **Multimodal Video customization**

HunyuanCustom supports inputs in the form of **text, images, audio, and video**. 
Specifically, it can handle single or multiple image inputs to enable customized video generation for one or more subjects. 
Additionally, it can incorporate extra audio inputs to drive the subject to speak the corresponding audio. 
Lastly, HunyuanCustom supports video input, allowing for the replacement of specified objects in the video with subjects from a given image.
![image](assets/material/teaser.png)

### **Various Applications**

With the multi-modal capabilities of HunyuanCustom, numerous downstream tasks can be accomplished. 
For instance, by taking multiple images as input, HunyuanCustom can facilitate **virtual human advertisements** and **virtual try-on**. Additionally, 
with image and audio inputs, it can create **singing avatars**. Furthermore, by using an image and a video as inputs, 
HunyuanCustom supports **video editing** by replacing subjects in the video with those in the provided image. 
More applications await your exploration!
![image](assets/material/application.png)


## 📈 Comparisons

To evaluate the performance of HunyuanCustom, we compared it with state-of-the-art video customization methods, 
including VACE, Skyreels, Pika, Vidu, Keling, and Hailuo. The comparison focused on face/subject consistency, 
video-text alignment, and overall video quality.

| Models            | Face-Sim | CLIP-B-T | DINO-Sim | Temp-Consis | DD   |
|-------------------|----------|----------|----------|-------------|------|
| VACE-1.3B         | 0.204    | _0.308_  | 0.569    | **0.967**   | 0.53 |
| Skyreels          | 0.402    | 0.295    | 0.579    | 0.942       | 0.72 |
| Pika              | 0.363    | 0.305    | 0.485    | 0.928       | _0.89_ |
| Vidu2.0           | 0.424    | 0.300    | 0.537    | _0.961_     | 0.43 |
| Keling1.6         | 0.505    | 0.285    | _0.580_  | 0.914       | 0.78 |
| Hailuo            | _0.526_  | **0.314**| 0.433    | 0.937       | **0.94** |
| **HunyuanCustom (Ours)** | **0.627**| 0.306    | **0.593**| 0.958       | 0.71 |

## 📜 Requirements

The following table shows the requirements for running HunyuanCustom model (batch size = 1) to generate videos:

|     Model    |  Setting<br/>(height/width/frame) | GPU Peak Memory  |
|:------------:|:--------------------------------:|:----------------:|
| HunyuanCustom   |        720px1280px129f          |       80GB        |
| HunyuanCustom   |        512px896px129f           |       60GB        |

* An NVIDIA GPU with CUDA support is required. 
  * The model is tested on a machine with 8GPUs.
  * **Minimum**: The minimum GPU memory required is 24GB for 720px1280px129f but very slow.
  * **Recommended**: We recommend using a GPU with 80GB of memory for better generation quality.
* Tested operating system: Linux


## 🛠️ Dependencies and Installation

Begin by cloning the repository:
```shell
git clone https://github.com/Tencent/HunyuanCustom.git
cd HunyuanCustom
```

### Installation Guide for Linux

We recommend CUDA versions 12.4 or 11.8 for the manual installation.

Conda's installation instructions are available [here](https://docs.anaconda.com/free/miniconda/index.html).

```shell
# 1. Create conda environment
conda create -n HunyuanCustom python==3.10.9

# 2. Activate the environment
conda activate HunyuanCustom

# 3. Install PyTorch and other dependencies using conda
# For CUDA 11.8
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# For CUDA 12.4
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# 4. Install pip dependencies
python -m pip install -r requirements.txt
# 5. Install flash attention v2 for acceleration (requires CUDA 11.8 or above)
python -m pip install ninja
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3
```

In case of running into float point exception(core dump) on the specific GPU type, you may try the following solutions:

```shell
# Option 1: Making sure you have installed CUDA 12.4, CUBLAS>=12.4.5.8, and CUDNN>=9.00 (or simply using our CUDA 12 docker image).
pip install nvidia-cublas-cu12==12.4.5.8
export LD_LIBRARY_PATH=/opt/conda/lib/python3.8/site-packages/nvidia/cublas/lib/

# Option 2: Forcing to explicitly use the CUDA 11.8 compiled version of Pytorch and all the other packages
pip uninstall -r requirements.txt  # uninstall all packages
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install ninja
pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3
```

Additionally, you can also use HunyuanVideo Docker image. Use the following command to pull and run the docker image.

```shell
# For CUDA 12.4 (updated to avoid float point exception)
docker pull hunyuanvideo/hunyuanvideo:cuda_12
docker run -itd --gpus all --init --net=host --uts=host --ipc=host --name hunyuanvideo --security-opt=seccomp=unconfined --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged hunyuanvideo/hunyuanvideo:cuda_12
pip install gradio==3.39.0 diffusers==0.33.0 transformers==4.41.2

# For CUDA 11.8
docker pull hunyuanvideo/hunyuanvideo:cuda_11
docker run -itd --gpus all --init --net=host --uts=host --ipc=host --name hunyuanvideo --security-opt=seccomp=unconfined --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged hunyuanvideo/hunyuanvideo:cuda_11
pip install gradio==3.39.0 diffusers==0.33.0 transformers==4.41.2
```


## 🧱 Download Pretrained Models

The details of download pretrained models are shown [here](models/README.md).

## 🚀 Parallel Inference on Multiple GPUs

For example, to generate a video with 8 GPUs, you can use the following command:

```bash
cd HunyuanCustom

export MODEL_BASE="./models"
export PYTHONPATH=./
torchrun --nnodes=1 --nproc_per_node=8 --master_port 29605 hymm_sp/sample_batch.py \
    --input './assets/images/seg_woman_01.png' \
    --pos-prompt "Realistic, High-quality. A woman is drinking coffee at a café." \
    --neg-prompt "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border." \
    --ckpt ${MODEL_BASE}"/hunyuancustom_720P/mp_rank_00_model_states.pt" \
    --video-size 720 1280 \
    --seed 1024 \
    --sample-n-frames 129 \
    --infer-steps 30 \
    --flow-shift-eval-video 13.0 \
    --save-path './results/sp_720p'
```

## 🔑 Single-gpu Inference

For example, to generate a video with 1 GPU, you can use the following command:

```bash
cd HunyuanCustom

export MODEL_BASE="./models"
export DISABLE_SP=1
export PYTHONPATH=./
python hymm_sp/sample_gpu_poor.py \
    --input './assets/images/seg_woman_01.png' \
    --pos-prompt "Realistic, High-quality. A woman is drinking coffee at a café." \
    --neg-prompt "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border." \
    --ckpt ${MODEL_BASE}"/hunyuancustom_720P/mp_rank_00_model_states_fp8.pt" \
    --video-size 512 896 \
    --seed 1024 \
    --sample-n-frames 129 \
    --infer-steps 30 \
    --flow-shift-eval-video 13.0 \
    --save-path './results/1gpu_540p' \
    --use-fp8
```

### Run with very low VRAM

```bash
cd HunyuanCustom

export MODEL_BASE="./models"
export CPU_OFFLOAD=1
export PYTHONPATH=./
python hymm_sp/sample_gpu_poor.py \
    --input './assets/images/seg_woman_01.png' \
    --pos-prompt "Realistic, High-quality. A woman is drinking coffee at a café." \
    --neg-prompt "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border." \
    --ckpt ${MODEL_BASE}"/hunyuancustom_720P/mp_rank_00_model_states_fp8.pt" \
    --video-size 720 1280 \
    --seed 1024 \
    --sample-n-frames 129 \
    --infer-steps 30 \
    --flow-shift-eval-video 13.0 \
    --save-path './results/cpu_720p' \
    --use-fp8 \
    --cpu-offload 
```


## Run a Gradio Server
```bash
cd HunyuanCustom

bash ./scripts/run_gradio.sh

```

## 🔗 BibTeX

If you find [HunyuanCustom](https://arxiv.org/abs/2505.04512) useful for your research and applications, please cite using this BibTeX:

```BibTeX
@misc{hu2025hunyuancustom,
      title={HunyuanCustom: A Multimodal-Driven Architecture for Customized Video Generation}, 
      author={Teng Hu and Zhentao Yu and Zhengguang Zhou and Sen Liang and Yuan Zhou and Qin Lin and Qinglin Lu},
      year={2025},
      eprint={2505.04512},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.04512}, 
}
```

## Acknowledgements

We would like to thank the contributors to the [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Llama](https://github.com/meta-llama/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [Xtuner](https://github.com/InternLM/xtuner), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research and exploration. 
