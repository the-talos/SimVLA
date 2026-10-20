# SimVLA: A Simple VLA Baseline for Robotic Manipulation

| **Paper** | **Website** | **Model & Data** |
| :------------------: | :-----------------------: | :---------------------: |
| [![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.18224) | [![Website](https://img.shields.io/badge/Project%20Page-181717?style=for-the-badge&logo=githubpages&logoColor=white)](https://frontierrobo.github.io/SimVLA/) | [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFBA00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/collections/YuankaiLuo/simvla) |

A simple and efficient Vision-Language-Action (VLA) model for robot manipulation tasks.

<img width="506" height="796" alt="image" src="https://github.com/user-attachments/assets/7ffb8969-aa4f-4bcc-8c38-33d5e7da4b25" />

## Installation

```bash
conda create -n simvla python=3.10 -y
conda activate simvla

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.57.0
pip install peft accelerate fastapi tensorboard uvicorn json_numpy safetensors scipy einops timm mmengine pyarrow h5py mediapy num2words av wandb websockets msgpack_numpy
pip install flash-attn==2.5.6 --no-build-isolation
pip install tensorflow tensorflow-datasets
```

> Important: Use `transformers>=4.57.0`.

### How to run the same on Intel's XPU

Inference (the LIBERO policy server) runs on an Intel GPU with the XPU build of PyTorch. Skip the `cu124` wheel and `flash-attn` above and install:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
pip install "transformers>=4.57.0" num2words fastapi uvicorn json_numpy websockets msgpack_numpy safetensors einops timm opencv-python-headless
```

Then start the server from `evaluation/libero` as in the evaluation steps, with `ZE_AFFINITY_MASK=0` in place of `CUDA_VISIBLE_DEVICES=1`:

```bash
ZE_AFFINITY_MASK=0 python serve_smolvlm_libero.py --checkpoint YuankaiLuo/SimVLA-LIBERO --norm_stats ../../norm_stats/libero_norm.json --port 8102
```

The device defaults to XPU, then CUDA, then CPU. Pass `--device xpu:0`, `--device cuda:0` or `--device cpu` to choose one. The LIBERO simulator client is not run on XPU.

## Training (LIBERO Dataset)

### 1. Prepare LIBERO Dataset

Download [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) dataset, and place it in `./datasets/metas/`.

### 2. Create Training Metadata

```bash
python create_libero_meta.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./datasets/metas/libero_train.json
```

### 3. Compute Normalization Statistics

```bash
python compute_libero_norm_stats.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./norm_stats/libero_norm.json
```

### 4. Start Training

**Small Model Configuration:**
```bash
bash train_smolvlm_small.sh
```

**Large Model Configuration:**
```bash
bash train_smolvlm_large.sh
```

### 5. Evaluation

```bash
cd evaluation/libero
```

### 6. Results

<img width="506" height="1220" alt="image" src="https://github.com/user-attachments/assets/6ee1cd5e-42c5-4cf7-9cce-6dc04c1a215f" />

## Model Architecture

- **Vision-Language Backbone**: SmolVLM-500M-Instruct (576 hidden dim)
- **Action Transformer**: Configurable depth and width
  - Small: 768 hidden, 12 layers, 12 heads
  - Large: 1024 hidden, 24 layers, 16 heads

## Reference

If you find our codes useful, please consider citing our work

```
@article{luo2026simvla,
  title={SimVLA: A Simple VLA Baseline for Robotic Manipulation},
  author={Luo, Yuankai and Chen, Woping and Liang, Tong and Wang, Baiqiao and Li, Zhenguo},
  journal={arXiv preprint arXiv:2602.18224},
  year={2026}
}
```


