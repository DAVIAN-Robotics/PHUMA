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

#### **1-1) Starting Point:** 
We begin with the Humanoid-X collection as described in our paper. For more details, refer to the [Humanoid-X repository](https://github.com/sihengz02/UH-1). If you want to reproduce the PHUMA dataset, a practical starting point is [Motion-X](https://github.com/IDEA-Research/Motion-X), which provides excellent documentation on SMPL-X pose data collection.

**SMPL-X Data Format:** Motion-X produces SMPL-X data in (N, 322) format, but PHUMA requires (N, 69) format, focusing on body pose and excluding face, hands, etc. If you're processing Motion-X data, you can convert it using our preprocessing script:

```bash
python src/curation/preprocess_motionx_format.py \
    --human_pose_folder /path_to_motionx_folder/subfolder \ # motionx_folder_path/humanml
    --output_dir data/human_pose
```

This script will:
- Recursively find all `.npy` files in the input folder
- Convert Motion-X format (N, 322) to PHUMA format (N, 69) by extracting `[transl, global_orient, body_pose]`
- Preserve the directory structure (e.g., `aist/subset_0008/`) in the output folder

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
```

- **For single file:**
```bash
# We provide an example clip: data/human_pose/example/kick.npy
human_pose_file="example/kick"

python src/curation/preprocess_smplx.py \
    --project_dir $PROJECT_DIR \
    --human_pose_file $human_pose_file \
    --visualize 0
```

- **For folder:**
```bash
human_pose_folder='data/human_pose/example'

python src/curation/preprocess_smplx_folder.py \
    --project_dir $PROJECT_DIR \
    --human_pose_folder $human_pose_folder \
    --visualize 0 \
```

**Output:** 
- Preprocessed motion chunks: `example/kick_chunk_0000.npy` and `example/kick_chunk_0001.npy` under `data/human_pose_preprocessed/`
- If you set `--visualize 1`, will also save `example/kick_chunk_0000.mp4` and `example/kick_chunk_0001.mp4` under `data/video/human_pose_preprocessed/`



#### **1-2) Tuning Curation Thresholds:**

The default thresholds are tuned to preserve motions with airborne phases (e.g., jumping) while filtering out physically implausible motions. This means some motions in PHUMA may contain minor penetration or floating artifacts. If you need stricter filtering for specific locomotion types (e.g., walking only), you can adjust the thresholds:

```bash
python src/curation/preprocess_smplx.py \
    --project_dir $PROJECT_DIR \
    --human_pose_file $human_pose_file \
    --foot_contact_threshold 0.8 \  # Default: 0.6. Increase to filter out more floating/penetration
    --visualize 0
```

For a complete list of tunable parameters, see `src/curation/preprocess_smplx.py`.

### 2. Physics-Constrained Motion Retargeting

To address artifacts introduced during the retargeting process, we employ **PhySINK**, our physics-constrained retargeting method that adapts curated human motion to humanoid robots while enforcing physical plausibility.

#### **2-1) Shape Adaptation (One-time Setup):**
```bash
# Find the SMPL-X shape that best fits a given humanoid robot
# This process only needs to be done once and can be reused for all motion files
python src/retarget/shape_adaptation.py \
    --project_dir $PROJECT_DIR \
    --robot_name g1
```

**Output:** Shape parameters saved to `asset/humanoid_model/g1/betas.npy`

#### **2-2) Motion Adaptation:**

This step retargets human motion to robot motion using PhySINK optimization. You can process either a single file or an entire folder.

- **For single file:**
```bash
# Using the curated data from the previous step for Unitree G1 humanoid robot

human_pose_preprocessed_file="example/kick_chunk_0000"

python src/retarget/motion_adaptation.py \
    --project_dir $PROJECT_DIR \
    --robot_name g1 \
    --human_pose_file $human_pose_preprocessed_file
```

- **For folder (with multiprocessing support):**
```bash
human_pose_preprocessed_folder="data/human_pose_preprocessed/example"

python src/retarget/motion_adaptation_multiprocess.py \
    --project_dir $PROJECT_DIR \
    --robot_name g1 \
    --human_pose_folder $human_pose_preprocessed_folder \
    --gpu_ids 0,1,2,3 \
    --processes_per_gpu 2
```

**Multiprocessing Parameters:**
- `--gpu_ids`: Comma-separated GPU IDs (e.g., `0,1,2,3`). If not specified, uses `--device` (default: `cuda:0`).
- `--processes_per_gpu`: Number of parallel processes per GPU (default: 1). 
  - **Recommended**: 1-2 for RTX 3090 (24GB), 2-4 for A100 (40GB+)
  - Total workers = `len(gpu_ids) × processes_per_gpu`
  - Example: `--gpu_ids 0,1,2,3 --processes_per_gpu 2` → 8 workers total
- `--num_workers`: Manual override for total number of workers (default: auto-calculated from GPU settings)
  - Use `-1` to use all available CPU cores (for CPU-only processing)

**Additional Options:**
- `--visualize`: Set to `1` to generate visualization videos (default: `0`)
- `--fps`: Frame rate for output videos (default: `30`)
- `--num_iter_dof`: Number of optimization iterations (default: `3001`)
- `--lr_dof`: Learning rate for DOF optimization (default: `0.005`)
- See `python src/retarget/motion_adaptation_multiprocess.py --help` for all available options

**Output:** 
- Retargeted humanoid motion data: `data/humanoid_pose/g1/kick_chunk_0000.npy`
  - Format: Dictionary containing `root_trans`, `root_ori`, `dof_pos`, and `fps`
- If you set `--visualize 1`, will also save `data/video/humanoid_pose/g1/kick_chunk_0000.mp4`

**Custom Robot Support:** We support Unitree G1 and H1-2, but you can also retarget to custom humanoid robots. See our [Custom Robot Integration Guide](asset/humanoid_model/README.md) for details.

**Degrees of Freedom (DoF) Clarification:**

Our retargeting pipeline produces:
- **G1:** 23 DoF (without wrist joints)
- **H1-2:** 27 DoF (with wrist joints)

In many cases, researchers tend to either exclude or include wrist joints for their purposes. Thus, the PHUMA dataset on HuggingFace provides:
- **G1:** 29 DoF (23 DoF with zero-padded wrist joints)
- **H1-2:** 27 DoF (with wrist joints)

The G1 data is zero-padded as follows:

```python
import numpy as np

N = dof_pos.shape[0]
dof_pos = np.concatenate([
    dof_pos[:, :19], 
    np.zeros((N, 3)),  # Left wrist padding
    dof_pos[:, 19:23], 
    np.zeros((N, 3))   # Right wrist padding
], axis=1)  # G1 23 DoF to 29 DoF
```

If you need to exclude wrist joints, you can convert as follows:

```python
# G1: 29 DoF to 23 DoF (exclude wrists)
dof_pos = np.concatenate([
    dof_pos[:, :19], 
    dof_pos[:, 22:26], 
], axis=1)

# H1-2: 27 DoF to 21 DoF (exclude wrists)
dof_pos = np.concatenate([
    dof_pos[:, :17], 
    dof_pos[:, 20:24], 
], axis=1)
```

## 🎯 Motion Tracking and Evaluation

To reproduce our reported quantitative results, use the provided data splits located in `data/split/`:
- `phuma_train.txt`
- `phuma_test.txt` 
- `unseen_video.txt`

LAFAN1 Retargeted Data: Available [here](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset).

LocoMuJoCo Retargeted Data: Available [here](https://github.com/robfiras/loco-mujoco).

For motion tracking and path following tasks, we utilize the codebase from [MaskedMimic](https://github.com/NVlabs/ProtoMotions).

**Note:** Our experiments reported in the paper used G1 (29 DoF) and H1-2 (21 DoF, excluding wrist joints) in IsaacGym. We chose 29 DoF for G1 because MaskedMimic already supports this configuration, and 21 DoF for H1-2 by referring to existing H1 support in MaskedMimic.

## ❓ FAQ

**Q: Are you planning to release either the original or preprocessed SMPL-X human pose files?**

A: We have released the SMPL-X human pose files for PHUMA Video (`unseen_video.txt`). Unfortunately, we cannot release human pose files of PHUMA Train/Test (`phuma_train.txt` and `phuma_test.txt`) due to license issues.

**Q: I want to process custom SMPL-X files with your code, but the orientation processing seems different.**

A: For SMPL-X processing, we mainly follow the code of [Motion-X](https://github.com/IDEA-Research/Motion-X). Taking AMASS as example, we follow [this code](https://github.com/IDEA-Research/Motion-X/tree/main/mocap-dataset-process) (except face motion augmentation since we focus on locomotion).

**Q: Some motions in PHUMA seem to have minor penetration or floating. Am I doing something wrong?**

A: The default threshold values in the curation stage are tuned to preserve motions with airborne phases (e.g., jumping) while filtering out physically implausible motions. This trade-off means some motions may contain minor artifacts. If you need stricter filtering for specific locomotion types (e.g., walking only), you can adjust the curation thresholds such as `--foot_contact_threshold`. See the **Tuning Curation Thresholds** section for details.

**Q: Can I retarget motions to custom humanoid robots using the PHUMA pipeline?**

A: Yes! While PHUMA dataset is provided for Unitree G1 and H1-2, you can use our PhySINK retargeting pipeline with custom robots by following our [Custom Robot Integration Guide](asset/humanoid_model/README.md). The guide covers adding heel/toe keypoints, creating configuration files, and tuning the retargeting process for your robot.

**Q: What are the DoF configurations for G1 and H1-2 in the PHUMA dataset?**

A: The HuggingFace dataset provides G1 (29 DoF) and H1-2 (27 DoF). Our retargeting pipeline originally produces G1 (23 DoF) and H1-2 (27 DoF), but G1 data is zero-padded to 29 DoF since researchers often choose to either include or exclude wrist joints for their purposes. Our paper experiments used G1 (29 DoF) and H1-2 (21 DoF, excluding wrists). See the **Degrees of Freedom (DoF) Clarification** section for conversion code between different DoF configurations.

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