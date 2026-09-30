# PHUMA: Physically Reliable Humanoid Locomotion Dataset

[![arXiv](https://img.shields.io/badge/arXiv-2510.26236-b31b1b.svg)](https://arxiv.org/abs/2510.26236)
[![Project Page](https://img.shields.io/badge/Project_Page-Visit-blue.svg)](https://davian-robotics.github.io/PHUMA/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/DAVIAN-Robotics/PHUMA)

> [Kyungmin Lee\*](https://kyungminn.github.io/)<sup>1</sup>, [Sibeen Kim\*](https://sibisibi.github.io/)<sup>1</sup>, [Youngdo Lee](https://leeyngdo.github.io/)<sup>1,2</sup>, [Minho Park](https://pmh9960.github.io/)<sup>1</sup>, [Hyunseung Kim](https://mynsng.github.io/)<sup>1,2</sup>, [Dongyoon Hwang](https://godnpeter.github.io/)<sup>1</sup>, [Donghu Kim](https://i-am-proto.github.io/)<sup>1</sup>, [Hojoon Lee†](https://joonleesky.github.io/)<sup>1</sup>, [Jaegul Choo†](https://sites.google.com/site/jaegulchoo/)<sup>1</sup>
> 
> <sup>1</sup>**DAVIAN Robotics, KAIST AI** &nbsp; <sup>2</sup>**KRAFTON**  
> **CoRL 2026 Spotlight** (\* equal contribution, † corresponding authors)

PHUMA leverages large-scale human motion data while overcoming physical artifacts through careful data curation and physics-constrained retargeting to create a high-quality humanoid locomotion dataset.

## 📰 News

- **[2026.09]** PHUMA has been accepted to **CoRL 2026** as a **Spotlight**! 🎉
- **[2025.12]** PHUMA is now natively supported in NVIDIA's [ProtoMotions](https://github.com/NVlabs/ProtoMotions)! You can train policies directly on PHUMA — see the [PHUMA Data Preparation guide](https://github.com/NVlabs/ProtoMotions/blob/main/docs/source/getting_started/phuma_preparation.rst).

## 🚀 Quick Start

**Just want the dataset?** Run the following (requires Python 3.9 + Conda):

```bash
git clone https://github.com/DAVIAN-Robotics/PHUMA.git && cd PHUMA
conda create -n phuma python=3.9 -y && conda activate phuma
pip install -r requirements.txt && pip install -e .
bash download_phuma.sh
```

This downloads the pre-built PHUMA dataset (G1 and H1-2), and you're ready to go.

> **Want more?** To modify/add custom data, or retarget to a custom robot,
> continue with the [Dataset Pipeline](#-dataset-pipeline) below.

<details>
<summary><strong>Step-by-step installation (with explanations)</strong></summary>

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

4. **Download PHUMA:**
   ```bash
   bash download_phuma.sh
   ```
   This downloads the pre-built PHUMA dataset (G1 and H1-2). Internally it
   fetches the data from Hugging Face, then runs the LAFAN1/LocoMuJoCo
   download and G1/H1-2 preprocessing — so steps 1–3 must be completed first.

</details>

## 📊 Dataset Pipeline

### 1. Physics-Aware Motion Curation

Our physics-aware curation pipeline filters out problematic motions from human motion data to ensure physical plausibility.

#### **1-1) Starting Point:** 
We begin with the Humanoid-X collection as described in our paper. For more details, refer to the [Humanoid-X repository](https://github.com/sihengz02/UH-1). If you want to reproduce the PHUMA dataset, a practical starting point is [Motion-X](https://github.com/IDEA-Research/Motion-X), which provides excellent documentation on SMPL-X pose data collection.

<details>
<summary><strong>ⅰ) Preprocess SMPL-X Data Format</strong></summary>

Motion-X produces SMPL-X data in (N, 322) `.npy` format or raw `.npz` format (stageii), but PHUMA requires (N, 69) format, focusing on body pose and excluding face, hands, etc. Our preprocessing script handles both formats automatically:

- Recursively finds all `.npy` and `.npz` files in the input folder
- For `.npz` files: converts Motion-X stageii format to (N, 322) with Y-up to Z-up coordinate transformation and FPS downsampling
- Converts Motion-X format (N, 322) to PHUMA format (N, 69) by extracting `[transl, global_orient, body_pose]`
- Preserves the directory structure (e.g., `aist/subset_0008/`) in the output folder

```bash
python src/curation/preprocess_motionx_format.py \
    --human_pose_folder /path_to_motionx_folder/subfolder \
    --output_dir data/human_pose \
    --target_fps 30  # Target FPS for npz downsampling (default: 30)
```

</details>


<details>
<summary><strong>ⅱ) Download SMPL-X Models</strong></summary>
 Before running the curation pipeline, you need to download the SMPL-X model files:

1. Visit [SMPL-X official website](https://smpl-x.is.tue.mpg.de/)
2. Register and download the following files:
   - `SMPLX_FEMALE.npz` and `SMPLX_FEMALE.pkl`
   - `SMPLX_MALE.npz` and `SMPLX_MALE.pkl`  
   - `SMPLX_NEUTRAL.npz` and `SMPLX_NEUTRAL.pkl`
3. Place all downloaded files in the `asset/human_model/smplx/` directory

</details>

#### **1-2) Tuning Curation Thresholds:**


The default thresholds are tuned to preserve motions with airborne phases (e.g., jumping) while filtering out physically implausible motions. This means some motions in PHUMA may contain minor penetration or floating artifacts. If you need stricter filtering for specific locomotion types (e.g., walking only), you can adjust the thresholds:

- **For single file:**

```bash
# Set your project directory
PROJECT_DIR="[REPLACE_WITH_YOUR_WORKING_DIRECTORY]/PHUMA"
cd $PROJECT_DIR

# We provide an example clip: data/human_pose/example/kick.npy
human_pose_file="example/kick"

python src/curation/preprocess_smplx.py \
    --project_dir $PROJECT_DIR \
    --human_pose_file $human_pose_file \
    --foot_contact_threshold 0.8 \
    --visualize 0
# foot_contact_threshold: Default 0.6. Increase to filter out more floating/penetration.
```

- **For folder:**

```bash
# Set your project directory
PROJECT_DIR="[REPLACE_WITH_YOUR_WORKING_DIRECTORY]/PHUMA"
cd $PROJECT_DIR

human_pose_folder='data/human_pose/example'

python src/curation/preprocess_smplx_folder.py \
    --project_dir $PROJECT_DIR \
    --human_pose_folder $human_pose_folder \
    --foot_contact_threshold 0.8 \
    --visualize 0
```
<details>
<summary>Output Details</summary>

- Preprocessed motion chunks: `example/kick_chunk_0000.npy` and `example/kick_chunk_0001.npy` under `data/human_pose_preprocessed/`
- If you set `--visualize 1`, will also save `example/kick_chunk_0000.mp4` and `example/kick_chunk_0001.mp4` under `data/video/human_pose_preprocessed/`

For a complete list of tunable parameters, see `src/curation/preprocess_smplx.py`.
</details>

### 2. Physics-Constrained Motion Retargeting

To address artifacts introduced during the retargeting process, we employ **PhySINK**, our physics-constrained retargeting method that adapts curated human motion to humanoid robots while enforcing physical plausibility.

#### **2-0) Custom Robot Setup (Optional):**

If you want to retarget to a custom humanoid robot (beyond G1 and H1-2), you first need to generate the required configuration files. Our setup script automatically:
- Adds heel/toe keypoint bodies to the robot model
- Computes a T-pose with ground-adjusted root height
- Generates `custom.xml`, `scene.xml`, and `config.yaml`

```bash
# From a MuJoCo XML file
python src/utils/setup_humanoid.py \
    --input /path/to/your_robot.xml \
    --humanoid_type your_robot_name

# From a URDF file
python src/utils/setup_humanoid.py \
    --input /path/to/your_robot.urdf \
    --humanoid_type your_robot_name

# If foot bodies are not auto-detected, specify them manually
python src/utils/setup_humanoid.py \
    --input /path/to/your_robot.xml \
    --humanoid_type your_robot_name \
    --left_foot_body left_ankle_roll_link \
    --right_foot_body right_ankle_roll_link
```

**Output:** Configuration files saved to `asset/humanoid_model/<your_robot_name>/`
- `custom.xml` — Robot model with heel/toe keypoints
- `scene.xml` — MuJoCo scene file
- `config.yaml` — Robot configuration (keypoints, bone mapping, joint info, T-pose)

> **Note:** After generation, review `config.yaml` and verify that the bone mappings and keypoints are correct for your robot.

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

<details>
<summary>Details</summary>

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

</details>

#### **✩ Custom Robot Support:**

We support Unitree G1 and H1-2, but you can also retarget to custom humanoid robots. See [**2-0) Custom Robot Setup**](#2-0-custom-robot-setup-optional) above to generate the required configuration files, then run the shape and motion adaptation steps as usual with your `--robot_name`.

## 🎯 Motion Tracking and Evaluation

To reproduce our reported quantitative results, use the provided data splits located in `data/split/`:
- `phuma_train.txt`
- `phuma_test.txt` 
- `unseen_video.txt`

For motion tracking and path following tasks, we utilize the codebase from [MaskedMimic](https://github.com/NVlabs/ProtoMotions).

## ❓ FAQ 
<details>
<summary> FAQ List </summary>
**Q: Are you planning to release either the original or preprocessed human pose files?**

A: Unfortunately, we cannot release human pose files of PHUMA Train/Test (`phuma_train.txt` and `phuma_test.txt`) due to license issues. However, we will release the SMPL human pose files for PHUMA Video (`unseen_video.txt`) soon! We haven't released them yet because our current codebase only supports SMPL-X. We will update the SMPL processing code soon and release the data at the same time.

**Q: I want to process custom SMPL-X files with your code, but the orientation processing seems different.**

A: For SMPL-X processing, we mainly follow the code of [Motion-X](https://github.com/IDEA-Research/Motion-X). Taking AMASS as example, we follow [this code](https://github.com/IDEA-Research/Motion-X/tree/main/mocap-dataset-process) (except face motion augmentation since we focus on locomotion).

**Q: Some motions in PHUMA seem to have minor penetration or floating. Am I doing something wrong?**

A: The default threshold values in the curation stage are tuned to preserve motions with airborne phases (e.g., jumping) while filtering out physically implausible motions. This trade-off means some motions may contain minor artifacts. If you need stricter filtering for specific locomotion types (e.g., walking only), you can adjust the curation thresholds such as `--foot_contact_threshold`. See the **Tuning Curation Thresholds** section for details.

**Q: Can I retarget motions to custom humanoid robots using the PHUMA pipeline?**

A: Yes! While PHUMA dataset is provided for Unitree G1 and H1-2, you can use our PhySINK retargeting pipeline with custom robots. See [**2-0) Custom Robot Setup**](#2-0-custom-robot-setup-optional) to generate configuration files from your robot's XML/URDF, then run the shape and motion adaptation steps with your `--robot_name`.

</details>

## 📝 Citation

If you use this dataset or code in your research, please cite our paper:

```bibtex
@inproceedings{lee2026phuma,
  title={PHUMA: Physically Reliable Humanoid Locomotion Dataset},
  author={Kyungmin Lee and Sibeen Kim and Youngdo Lee and Minho Park and Hyunseung Kim and Dongyoon Hwang and Donghu Kim and Hojoon Lee and Jaegul Choo},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2026},
}
```