# Custom Robot Integration Guide

While we currently support Unitree humanoid robots (G1 and H1-2), custom robots can be integrated with some modifications.

## Required Files

Create a new robot folder in this directory containing:
- `meshes/` - STL files for visualization
- `config.yaml` - Robot configuration
- `custom.xml` - MuJoCo robot model with heel/toe keypoints
- `scene.xml` (optional) - Can be copied from existing g1 or h1_2 folders

## Main Challenges

### (1) Add Heel and Toe Keypoints in `custom.xml`

PhySINK requires heel and toe keypoints for accurate foot contact modeling during retargeting. These keypoints should be added as child bodies to the lowest link of your robot's foot.

**Reference implementation in G1:**
- [left_heel_keypoint](https://github.com/DAVIAN-Robotics/PHUMA/blob/main/asset/humanoid_model/g1/custom.xml#L88-L90)
- [left_toe_keypoint](https://github.com/DAVIAN-Robotics/PHUMA/blob/main/asset/humanoid_model/g1/custom.xml#L91-L93)
- [right_heel_keypoint](https://github.com/DAVIAN-Robotics/PHUMA/blob/main/asset/humanoid_model/g1/custom.xml#L133-L135)
- [right_toe_keypoint](https://github.com/DAVIAN-Robotics/PHUMA/blob/main/asset/humanoid_model/g1/custom.xml#L136-L138)

We determined the positions for heel and toe keypoints by inspecting the lowest points of the foot mesh. Heel should be at the rearmost/lowest point, and toe at the frontmost/lowest point.

![Heel and toe keypoints visualization](https://github.com/DAVIAN-Robotics/PHUMA/blob/main/docs/images/heel_toe_keypoints.png)

### (2) Create `config.yaml`

Refer to [g1/config.yaml](https://github.com/DAVIAN-Robotics/PHUMA/blob/main/asset/humanoid_model/g1/config.yaml) as a template. Most fields (`root_pos`, `root_ori`, `dof_pos`, `body_names`, `joint_names`, `joint_limits`, `joint_velocity_limits`, `dof`) are straightforward duplicates or renamed entities from your robot MJCF file.

**The two critical fields that need careful attention:**

**`bone_mapping`**: Maps corresponding bones between SMPL-X skeleton and your robot's kinematic chain. Each entry is a 4-tuple: `[smpl_parent, smpl_child, robot_parent, robot_child]`

```yaml
bone_mapping:
  # Left Leg
  - ['pelvis', 'left_hip', 'pelvis', 'left_hip_roll_link']
  - ['left_hip', 'left_knee', 'left_hip_roll_link', 'left_knee_link']
  - ['left_knee', 'left_ankle', 'left_knee_link', 'left_ankle_pitch_link']
  # Right Leg, Arms...
```
These bone correspondences are used to compute bone length and orientation losses during optimization.

**`keypoints`**: Defines correspondence between SMPL-X keypoints and robot body frames for position matching.

```yaml
keypoints:
  - { name: 'pelvis',                 body: 'pelvis' }
  - { name: 'left_hip_keypoint',      body: 'left_hip_roll_link' }
  - { name: 'left_knee_keypoint',     body: 'left_knee_link' }
  - { name: 'left_ankle_keypoint',    body: 'left_ankle_pitch_link' }
  - { name: 'left_heel_keypoint',     body: 'left_heel_keypoint' }
  - { name: 'left_toe_keypoint',      body: 'left_toe_keypoint' }
  # Right leg, torso, arms...
```
The algorithm minimizes distances between corresponding SMPL-X joints and these robot keypoints.

### (3) Update Retargeting Scripts and Tune

Add your custom robot_name to argparse choices in `src/retarget/shape_adaptation.py` and `src/retarget/motion_adaptation.py`.

Then run shape adaptation (one-time):
```bash
python src/retarget/shape_adaptation.py \
    --project_dir $PROJECT_DIR \
    --robot_name your_robot_name
```

And motion adaptation with visualization:
```bash
python src/retarget/motion_adaptation.py \
    --project_dir $PROJECT_DIR \
    --robot_name your_robot_name \
    --human_pose_file example/kick_chunk_0000 \
    --visualize 1
```

Try iterative loss weight tuning if needed alongside visualization for your specific robot.