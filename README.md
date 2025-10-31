# PHUMA: Physically-Grounded Humanoid Locomotion Dataset

[![arXiv](https://img.shields.io/badge/arXiv-2510.26236-b31b1b.svg)](https://arxiv.org/abs/2510.26236)
[![Project Page](https://img.shields.io/badge/Project_Page-Visit-blue.svg)](https://davian-robotics.github.io/PHUMA/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/DAVIAN-Robotics/PHUMA)

> [Kyungmin Lee\*](https://kyungminn.github.io/), [Sibeen Kim\*](https://sibisibi.github.io/), [Minho Park](https://pmh9960.github.io/), [Hyunseung Kim](https://mynsng.github.io/), [Dongyoon Hwang](https://godnpeter.github.io/), [Hojoon Lee](https://joonleesky.github.io/), and [Jaegul Choo](https://sites.google.com/site/jaegulchoo/)
> 
> **DAVIAN Robotics, KAIST AI**  
> arXiv 2025. (\* indicates equal contribution)

PHUMA leverages large-scale human motion data while overcoming physical artifacts through careful data curation and physics-constrained retargeting to create a high-quality humanoid locomotion dataset.

## 🚀 Quick Start

### Prerequisites
- Python 3.9
- CUDA 12.4 (recommended)
- Conda package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DAVIAN-Robotics/PHUMA.git
   cd PHUMA
   ```

2. **Set up the environment:**
   ```bash
   conda create -n phuma python=3.9 -y
   conda activate phuma
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## 📊 Dataset Pipeline

### 1. Physics-Aware Motion Curation

Our physics-aware curation pipeline filters out problematic motions from human motion data to ensure physical plausibility.

**Starting Point:** We begin with the Humanoid-X collection as described in our paper. For more details, refer to the [Humanoid-X repository](https://github.com/sihengz02/UH-1).

**Required SMPL-X Models:** Before running the curation pipeline, you need to download the SMPL-X model files:

1. Visit [SMPL-X official website](https://smpl-x.is.tue.mpg.de/)
2. Register and download the following files:
   - `SMPLX_FEMALE.npz` and `SMPLX_FEMALE.pkl`
   - `SMPLX_MALE.npz` and `SMPLX_MALE.pkl`  
   - `SMPLX_NEUTRAL.npz` and `SMPLX_NEUTRAL.pkl`
3. Place all downloaded files in the `asset/human_model/smplx/` directory

**Example Usage:**
```bash
# Set your project directory
PROJECT_DIR="[REPLACE_WITH_YOUR_WORKING_DIRECTORY]/PHUMA"
cd $PROJECT_DIR

# We provide an example clip: data/human_pose/example/kick.npy
human_pose_file="example/kick"

python src/curation/preprocess_smplx.py \
    --project_dir $PROJECT_DIR \
    --human_pose_file $human_pose_file \
    --visualize 0
```

**Output:** 
- Preprocessed motion chunks: `example/kick_chunk_0000.npy` and `example/kick_chunk_0001.npy` under `data/human_pose_preprocessed/`
- If you set `--visualize 1`, will also save `example/kick_chunk_0000.mp4` and `example/kick_chunk_0001.mp4` under `data/video/human_pose_preprocessed/`

### 2. Physics-Constrained Motion Retargeting

To address artifacts introduced during the retargeting process, we employ **PhySINK**, our physics-constrained retargeting method that adapts curated human motion to humanoid robots while enforcing physical plausibility.

**Shape Adaptation (One-time Setup):**
```bash
# Find the SMPL-X shape that best fits a given humanoid robot
# This process only needs to be done once and can be reused for all motion files
python src/retarget/shape_adaptation.py \
    --project_dir $PROJECT_DIR \
    --robot_name g1
```

**Output:** Shape parameters saved to `asset/humanoid_model/g1/betas.npy`

**Motion Adaptation:**
```bash
# Using the curated data from the previous step for Unitree G1 humanoid robot

human_pose_preprocessed_file="example/kick_chunk_0000"

python src/retarget/motion_adaptation.py \
    --project_dir $PROJECT_DIR \
    --robot_name g1 \
    --human_pose_file $human_pose_preprocessed_file \
    --visualize 0
```

**Output:** 
- Retargeted humanoid motion data: `data/humanoid_pose/g1/example/kick_chunk_0000.npy`
- If you set `--visualize 1`, will also save `data/video/humanoid_pose/example/kick_chunk_0000.mp4`

## 🎯 Motion Tracking and Evaluation

To reproduce our reported quantitative results, use the provided data splits located in `data/split/`:
- `phuma_train.txt`
- `phuma_test.txt` 
- `unseen_video.txt`

LAFAN1 Retargeted Data: Available [here](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset).

LocoMuJoCo Retargeted Data: Available [here](https://github.com/robfiras/loco-mujoco).

For motion tracking and path following tasks, we utilize the codebase from [MaskedMimic](https://github.com/NVlabs/ProtoMotions).

## 📝 Citation

If you use this dataset or code in your research, please cite our paper:

```bibtex
@article{lee2025phuma,
  title={PHUMA: Physically-Grounded Humanoid Locomotion Dataset},
  author={Kyungmin Lee and Sibeen Kim and Minho Park and Hyunseung Kim and Dongyoon Hwang and Hojoon Lee and Jaegul Choo},
  journal={arXiv preprint arXiv:2510.26236},
  year={2025},
}
```