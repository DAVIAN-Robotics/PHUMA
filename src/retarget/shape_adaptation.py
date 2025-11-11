import os
from os.path import dirname, join
import argparse
import yaml

import torch
import numpy as np
import mujoco
from tqdm import tqdm
import smplx
from smplx.joint_names import JOINT_NAMES

import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", type=str, required=True)
    parser.add_argument("--robot_name", type=str, required=True)

    parser.add_argument("--print_every", type=int, default=500)
    parser.add_argument("--num_iter_beta", type=int, default=5001)
    parser.add_argument("--lr_beta", type=float, default=0.01)
    
    parser.add_argument("--keypoint_matching_weight", type=float, default=2, help="Weight for keypoint matching loss")
    parser.add_argument("--kinematic_loss_weight", type=float, default=1, help="Weight for kinematic loss (bone length + orientation)")
    parser.add_argument("--scale_unit_loss_weight", type=float, default=0.001, help="Weight for scale unit loss")
    parser.add_argument("--scale_symmetry_loss_weight", type=float, default=1, help="Weight for scale symmetry loss")
    parser.add_argument("--ground_contact_loss_weight", type=float, default=10, help="Weight for ground contact loss")
    
    parser.add_argument("--num_betas", type=int, default=10, help="Number of SMPL shape parameters (default: 10)")
    parser.add_argument("--beta_min", type=float, default=-5.0, help="Minimum value for beta clamping (default: -5.0)")
    parser.add_argument("--beta_max", type=float, default=5.0, help="Maximum value for beta clamping (default: 5.0)")
    parser.add_argument("--heel_offset", type=float, default=0.005, help="Heel offset in meters (default: 0.005m = 5mm)")

    return parser.parse_args()

def main(args):
    human_model_dir = join(args.project_dir, "asset", "human_model")
    robot_model_dir = join(args.project_dir, "asset", "humanoid_model", args.robot_name)
    
    robot_config_path = join(robot_model_dir, "config.yaml")
    with open(robot_config_path, 'r') as file:
        robot_config = yaml.safe_load(file)

    smpl_kp_config_path = join(human_model_dir, "config.yaml")
    with open(smpl_kp_config_path, 'r') as file:
        smpl_kp_config = yaml.safe_load(file)

    smpl = smplx.create(human_model_dir, model_type="smplx", num_pca_comps=45)

    robot_path = join(robot_model_dir, "custom.xml")
    robot = mujoco.MjModel.from_xml_path(robot_path)
    pose = mujoco.MjData(robot)
    
    ROOT_POS = np.array(robot_config['root_pos'])
    ROOT_ORI = np.array(robot_config['root_ori'])
    DOF_POS = np.array(robot_config['dof_pos'])
    pose.qpos = np.hstack((ROOT_POS, ROOT_ORI, DOF_POS))

    mujoco.mj_forward(robot, pose)

    # --- Keypoint Setup ---
    body_names = [robot.body(i).name for i in range(1, robot.nbody)]
    robot_link_map = {name: i for i, name in enumerate(body_names)}
    
    # Get robot keypoints to be used for retargeting (all except heel and toe)
    robot_keypoint_names = [kp['name'] for kp in robot_config['keypoints'] if 'heel' not in kp['name'] and 'toe' not in kp['name']]
    robot_keypoint_bodies = [kp['body'] for kp in robot_config['keypoints'] if 'heel' not in kp['name'] and 'toe' not in kp['name']]
    robot_keypoint_indices = [robot_link_map[body] for body in robot_keypoint_bodies]

    keypoint_trans_gt = torch.from_numpy(np.copy(pose.xpos)[[i+1 for i in robot_keypoint_indices]]).to(torch.float32)
    
    # Get corresponding SMPL keypoints
    smpl_joint_map = {name: i for i, name in enumerate(JOINT_NAMES)} 
    smpl_kp_map = {kp['name']: kp['body'] for kp in smpl_kp_config['keypoints']}
    smpl_body_names = [smpl_kp_map[name] for name in robot_keypoint_names]
    smpl_keypoint_indices = [smpl_joint_map[name] for name in smpl_body_names]

    betas = torch.zeros((1, args.num_betas), requires_grad=True)
    offset = torch.zeros((1, 3), requires_grad=True)

    BONE_MAPPING = robot_config['bone_mapping']
    num_bones = len(BONE_MAPPING)
    link_scales = torch.ones(num_bones, dtype=torch.float32, requires_grad=True) 
    keypoint_weights = torch.ones(len(robot_keypoint_indices), dtype=torch.float32, requires_grad=False)
    
    # Do not match pelvis and torso for global keypoint matching loss
    if 'pelvis' in robot_keypoint_names:
        pelvis_idx = robot_keypoint_names.index('pelvis')
        keypoint_weights[pelvis_idx] = 0.0
    if 'torso_keypoint' in robot_keypoint_names:
        torso_idx = robot_keypoint_names.index('torso_keypoint')
        keypoint_weights[torso_idx] = 0.0

    keypoint_weights = keypoint_weights.view(1, len(robot_keypoint_indices), 1)    
    human_bone_indices = [] 
    robot_bone_indices = [] 
    for smpl_p, smpl_c, robot_p, robot_c in BONE_MAPPING: 
        human_bone_indices.append([smpl_joint_map[smpl_c], smpl_joint_map[smpl_p]]) 
        robot_bone_indices.append([robot_link_map[robot_c], robot_link_map[robot_p]]) 
     
    human_bone_indices = torch.tensor(human_bone_indices) 
    robot_bone_indices = torch.tensor(robot_bone_indices) 

    human_child_idx, human_parent_idx = human_bone_indices[:, 0], human_bone_indices[:, 1] 
    robot_child_idx, robot_parent_idx = robot_bone_indices[:, 0], robot_bone_indices[:, 1] 

    robot_body_trans_gt = torch.from_numpy(np.copy(pose.xpos)[1:]).to(torch.float32)
    robot_bones = robot_body_trans_gt[robot_child_idx] - robot_body_trans_gt[robot_parent_idx]

    optimizer = torch.optim.Adam([betas, offset, link_scales], lr=args.lr_beta)

    # --- Debug bone lengths and orientations ---
    # Define a 180-degree rotation around the Z-axis to align SMPL's heading with the robot's default pose.
    rot_180_z = torch.tensor([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]], dtype=torch.float32)

    with torch.no_grad():
        output = smpl(betas=betas, body_pose=None)
        joints = output.joints
        # Correct Y-up to Z-up conversion: (x, y, z) -> (z, x, y)
        smpl_joints = joints[0, :, [2, 0, 1]]
        smpl_joints = smpl_joints @ rot_180_z # Apply 180-degree yaw rotation
        smpl_bones = smpl_joints[human_child_idx] - smpl_joints[human_parent_idx]

    robot_bone_lengths = torch.linalg.norm(robot_bones, dim=-1)
    smpl_bone_lengths = torch.linalg.norm(smpl_bones, dim=-1)
    cosine_sim = F.cosine_similarity(robot_bones, smpl_bones, dim=-1)

    for bone_idx in range(len(BONE_MAPPING)):
        bone_name = f"{BONE_MAPPING[bone_idx][2]}-{BONE_MAPPING[bone_idx][3]}"

    for i in tqdm(range(args.num_iter_beta)):
        optimizer.zero_grad()
        
        with torch.no_grad():
            betas.data.clamp_(args.beta_min, args.beta_max)
            link_scales.clamp_(min=0.0) 

        output = smpl(betas=betas, body_pose=None)
        joints = output.joints
        vertices = output.vertices

        # Correct Y-up to Z-up conversion: (x, y, z) -> (z, x, y)
        keypoint_trans = joints[0, smpl_keypoint_indices][:, [2, 0, 1]]
        keypoint_trans = keypoint_trans @ rot_180_z # Apply 180-degree yaw rotation
        keypoint_trans += offset
        
        loss_retarget = (torch.abs(keypoint_trans - keypoint_trans_gt) * keypoint_weights).mean()
        
        smpl_joints = joints[0, :, [2, 0, 1]]
        smpl_joints = smpl_joints @ rot_180_z # Apply 180-degree yaw rotation
        smpl_bones = smpl_joints[human_child_idx] - smpl_joints[human_parent_idx]
        
        scale_factors = link_scales.view(num_bones, 1)
        loss_pos = F.mse_loss(smpl_bones, robot_bones * scale_factors) 
        loss_ang = (1.0 - F.cosine_similarity(smpl_bones, robot_bones, dim=-1)).mean() 
        loss_kinematic = loss_pos + loss_ang 

        loss_scale_unit = F.mse_loss(link_scales, torch.ones_like(link_scales)) 

        reshaped_scales = link_scales.view(-1, 6)
        left_scales = reshaped_scales[:, :3].reshape(-1)
        right_scales = reshaped_scales[:, 3:].reshape(-1)
        loss_scale_symmetry = F.mse_loss(left_scales, right_scales) 

        # Correct ground contact loss to use Z-up coordinates
        toe_indices = smpl_kp_config["left_toe_indices"] + smpl_kp_config["right_toe_indices"]
        heel_indices = smpl_kp_config["left_heel_indices"] + smpl_kp_config["right_heel_indices"]

        smpl_vertices_zup = vertices[0, :, [2, 0, 1]]
        smpl_vertices_zup = smpl_vertices_zup @ rot_180_z
        smpl_vertices_zup += offset

        toe_heights = smpl_vertices_zup[toe_indices, 2]
        heel_heights = smpl_vertices_zup[heel_indices, 2] + args.heel_offset
        ground_contact_heights = torch.cat([toe_heights, heel_heights], dim=0)
        loss_ground_contact = (torch.mean(ground_contact_heights))**2

        loss = args.keypoint_matching_weight * loss_retarget \
             + args.kinematic_loss_weight * loss_kinematic \
             + args.scale_unit_loss_weight * loss_scale_unit \
             + args.scale_symmetry_loss_weight * loss_scale_symmetry \
             + args.ground_contact_loss_weight * loss_ground_contact

        loss.backward()
        optimizer.step()

        if i % args.print_every == 0:
            print(f"Iteration {i}: Loss = {loss.item():.4f}") 
            print(f"\t Keypoint Matching Loss = {loss_retarget.item():.4f}") 
            print(f"\t Kinematic Positional Loss = {loss_pos.item():.4f}") 
            print(f"\t Kinematic Angular Loss = {loss_ang.item():.4f}") 
            print(f"\t Scale Unit Loss = {loss_scale_unit.item():.4f}") 
            print(f"\t Scale Symmetry Loss = {loss_scale_symmetry.item():.4f}") 
            print(f"\t Per-Link Scale Factors = {link_scales.cpu().detach().numpy()}") 
            print(f"\t Ground Contact Loss = {loss_ground_contact.item():.4f}")
 
            betas_value = betas[0].detach().cpu().numpy()
            offset_value = offset[0].detach().cpu().numpy()
            print(f"betas: {betas_value}")
            print(f"offset: {offset_value}")
 
    betas = betas.squeeze().detach().numpy()
    print(f"betas: {betas}")

    save_path = join(robot_model_dir, "betas.npy")
    os.makedirs(dirname(save_path), exist_ok=True)
    np.save(save_path, betas)


if __name__ == "__main__":
    args = parse_args()
    main(args)