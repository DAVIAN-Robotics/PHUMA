import os
from os.path import join, exists, dirname
import argparse
import yaml

import torch
import torch.nn.functional as F 
import numpy as np
import smplx
from smplx.joint_names import JOINT_NAMES 
from tqdm import tqdm 
from scipy.spatial.transform import Rotation as R 

from utils.smpl import load_motion_parms, find_robust_ground, _butter_lowpass_filter, _unify_quaternion_signs
from utils.robot import HumanoidRetargetKeypoint 
from utils.visualize import render_robot_pose, write_video


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", type=str, required=True)
    parser.add_argument("--robot_name", choices=["g1", "h1_2"], required=True)
    parser.add_argument("--human_pose_file", type=str, required=True)
    parser.add_argument("--visualize", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument("--global_match_weight", type=float, default=0.1)
    parser.add_argument("--local_match_weight", type=float, default=2.0)
    parser.add_argument("--joint_feasibility_weight", type=float, default=1000.0)
    parser.add_argument("--grounding_weight", type=float, default=10.0)
    parser.add_argument("--skating_weight", type=float, default=0.002)
    parser.add_argument("--smooth_weight", type=float, default=0.05)
    parser.add_argument("--low_pass_filter", type=int, default=1)

    return parser.parse_args()

def main(args):
    smpl_pose_file = join(args.project_dir, "data", "human_pose_preprocessed", f"{args.human_pose_file}.npy")
    robot_pose_file = join(args.project_dir, "data", "humanoid_pose", args.robot_name, f"{args.human_pose_file}.npy")

    human_model_dir = join(args.project_dir, "asset", "human_model")
    robot_model_dir = join(args.project_dir, "asset", "humanoid_model", args.robot_name)

    motion_parms = load_motion_parms(smpl_pose_file, foot_contact=True)
    betas = np.load(join(robot_model_dir, "betas.npy"))
    motion_parms["betas"] = torch.from_numpy(betas).unsqueeze(0).repeat(motion_parms["body_pose"].shape[0], 1).float()

    smpl = smplx.create(human_model_dir, model_type="smplx", num_pca_comps=45)

    lr_dof=0.005
    num_iter_dof=3001
    print_every=500 

    keypoint_matching_weight = args.global_match_weight

    kinematic_loss_weight = args.local_match_weight
    scale_unit_loss_weight = 0.001
    scale_symmetry_loss_weight = 0.001

    ground_contact_loss_weight = args.grounding_weight
    skating_loss_weight = args.skating_weight

    joint_range_loss_weight = args.joint_feasibility_weight

    smoothing_weight = args.smooth_weight

    fps = 30
    dt = 1.0 / fps

    robot_name = args.robot_name

    robot_config_path = join(robot_model_dir, "config.yaml")
    with open(robot_config_path, 'r') as file:
        robot_config = yaml.safe_load(file)

    robot_body_names = robot_config['body_names']

    smpl_joint_map = {name: i for i, name in enumerate(JOINT_NAMES)}
    robot_link_map = {name: i for i, name in enumerate(robot_body_names)}

    robot_keypoint_names = [kp['name'] for kp in robot_config['keypoints'] if 'heel' not in kp['name'] and 'toe' not in kp['name']]
    robot_keypoint_bodies = [kp['body'] for kp in robot_config['keypoints'] if 'heel' not in kp['name'] and 'toe' not in kp['name']]
    robot_keypoint_indices = [robot_link_map[body] for body in robot_keypoint_bodies]

    ground_contact_bodies = ["left_toe_keypoint", "left_heel_keypoint", "right_toe_keypoint", "right_heel_keypoint"]
    ground_contact_indices = [robot_link_map[body] for body in ground_contact_bodies]

    smpl_kp_config_path = join(human_model_dir, "config.yaml")
    with open(smpl_kp_config_path, 'r') as file:
        smpl_kp_config = yaml.safe_load(file)

    smpl_joint_map = {name: i for i, name in enumerate(JOINT_NAMES)} 
    smpl_kp_map = {kp['name']: kp['body'] for kp in smpl_kp_config['keypoints']}
    smpl_body_names = [smpl_kp_map[name] for name in robot_keypoint_names]
    smpl_keypoint_indices = [smpl_joint_map[name] for name in smpl_body_names]

    num_keypoint = len(smpl_keypoint_indices)
    num_frames = motion_parms['body_pose'].shape[0] 
    transl = motion_parms['transl'].cpu().numpy()
    global_orient = motion_parms['global_orient'] 

    keypoint_traj = np.zeros((num_frames, num_keypoint, 3)) 
    human_joints_traj = np.zeros((num_frames, 127, 3))

    foot_contact = motion_parms.pop("foot_contact")

    vertices = []

    for i in range(num_frames):
        frame_params = {k: v[i:i+1] for k, v in motion_parms.items()}
        output = smpl(**frame_params)
        vertices.append(output.vertices.detach().cpu().numpy().squeeze())
        human_joints_traj[i] = output.joints.detach().cpu().numpy().squeeze()

    vertices = np.array(vertices)

    contact_types = ["left_toe_indices", "left_heel_indices", "right_toe_indices", "right_heel_indices"]

    foot_contact_vertex_indices = {}
    for key in contact_types:
        foot_contact_vertex_indices[key] = smpl_kp_config[key]

    robust_ground_y = find_robust_ground(vertices, foot_contact_vertex_indices)

    transl[:, 1] -= robust_ground_y
    human_joints_traj[..., 1] -= robust_ground_y

    transl_zup = transl.copy()
    transl_zup[..., 0] = transl[..., 2]
    transl_zup[..., 1] = transl[..., 0]
    transl_zup[..., 2] = transl[..., 1]
    transl = transl_zup
    
    human_joints_traj_zup = human_joints_traj.copy()
    human_joints_traj_zup[..., 0] = human_joints_traj[..., 2]
    human_joints_traj_zup[..., 1] = human_joints_traj[..., 0]
    human_joints_traj_zup[..., 2] = human_joints_traj[..., 1]
    human_joints_traj = human_joints_traj_zup

    rotation = R.from_rotvec(global_orient)
    root_ori_matrix_yup = rotation.as_matrix()
    
    R_conv = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    root_ori_matrix_zup = R_conv[None, ...] @ root_ori_matrix_yup @ R_conv.T[None, ...]

    global_orient = R.from_matrix(root_ori_matrix_zup).as_rotvec()

    keypoint_traj[:, np.arange(num_keypoint), :] = human_joints_traj[:, smpl_keypoint_indices, :]

    mjcf_file = join(robot_model_dir, "custom.xml")
    retarget = HumanoidRetargetKeypoint(mjcf_file=mjcf_file, device='cuda:0')

    rotation = R.from_rotvec(global_orient)
    root_ori_matrix = rotation.as_matrix()

    rotation = R.from_matrix(root_ori_matrix)
    root_ori_quat = rotation.as_quat()

    rotations = R.from_quat(root_ori_quat).as_rotvec()
    root_ori_init = torch.from_numpy(rotations).to(retarget.device).to(torch.float32) 
    root_trans_init = torch.from_numpy(transl).to(retarget.device).to(torch.float32) 

    human_bone_indices = [] 
    robot_bone_indices = []
    bone_mapping = robot_config['bone_mapping']

    for smpl_p, smpl_c, robot_p, robot_c in bone_mapping: 
        human_bone_indices.append([smpl_joint_map[smpl_c], smpl_joint_map[smpl_p]]) 
        robot_bone_indices.append([robot_link_map[robot_c], robot_link_map[robot_p]]) 
     
    human_bone_indices = torch.tensor(human_bone_indices, device=retarget.device) 
    robot_bone_indices = torch.tensor(robot_bone_indices, device=retarget.device) 

    human_child_idx, human_parent_idx = human_bone_indices[:, 0], human_bone_indices[:, 1] 
    robot_child_idx, robot_parent_idx = robot_bone_indices[:, 0], robot_bone_indices[:, 1] 
     
    link_scales = torch.ones(12, device=retarget.device, dtype=torch.float32, requires_grad=True) 

    joint_pos = torch.zeros( 
        num_frames, len(robot_config['dof'])-1, 
        device=retarget.device, dtype=torch.float32, requires_grad=True) 
    root_ori = torch.zeros( 
        num_frames, 3, 
        device=retarget.device, dtype=torch.float32, requires_grad=True) 
    root_trans = torch.zeros( 
        num_frames, 3, 
        device=retarget.device, dtype=torch.float32, requires_grad=True) 

    root_ori.data = root_ori_init
    root_trans.data = root_trans_init

    optimizer = torch.optim.Adam([joint_pos, root_trans, link_scales], lr=lr_dof) 

    keypoint_gt = torch.from_numpy(keypoint_traj).to(retarget.device).to(torch.float32) 
    human_joints_gt = torch.from_numpy(human_joints_traj).to(retarget.device, dtype=torch.float32)
    
    contact_weights = foot_contact.to(retarget.device)

    min_vals = retarget.robot.joints_range[:, 0] 
    max_vals = retarget.robot.joints_range[:, 1] 

    vel_limits = torch.tensor(robot_config['joint_velocity_limits'], device=retarget.device, dtype=torch.float32) 

    for i in tqdm(range(num_iter_dof)):
        optimizer.zero_grad()

        joint_vel = torch.diff(joint_pos, n=1, dim=0) / dt
        root_vel = torch.diff(root_trans, n=1, dim=0) / dt

        last = joint_vel[0: -2, :]
        this = joint_vel[1: -1, :]
        next = joint_vel[2:   , :]
        loss_smooth = torch.mean(torch.abs(this - (last + next) * 0.5))

        last = root_vel[0: -2, :]
        this = root_vel[1: -1, :]
        next = root_vel[2:   , :]
        loss_smooth += torch.mean(torch.abs(this - (last + next) * 0.5))

        with torch.no_grad(): 
            link_scales.clamp_(min=0.0) 

        batch_len = joint_pos.shape[0] 
        pose_batch = torch.zeros( 
            1, batch_len, len(robot_body_names), 3, 
            dtype=torch.float32, device=retarget.device) 
        
        pose_batch[0, :, 0, :] = root_ori
        for k, joint_name in enumerate(robot_config["joint_names"]):
            key = joint_name.removesuffix("_joint")
            for l, body_name in enumerate(robot_config["body_names"]):
                if key in body_name:
                    pose_batch[0, :, l, robot_config["dof"][k]] = joint_pos[:, k-1]
                    break

        output = retarget.robot.kinematics.fk_batch( 
            pose=pose_batch,
            trans=root_trans.unsqueeze(0), 
            convert_to_mat=True, 
            return_full=False)
        output_trans = output['global_translation'][0, :, :, :]

        keypoint = torch.zeros( 
            batch_len, num_keypoint, 3, 
            dtype=torch.float32, device=retarget.device) 
        keypoint[:, :] = output_trans[:, robot_keypoint_indices] 
        keypoint_weights = torch.tensor([ 
            0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  
            0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  
            device=retarget.device, dtype=torch.float32).view(1, num_keypoint)

        loss_retarget_full = (torch.abs(keypoint - keypoint_gt) * keypoint_weights.unsqueeze(-1))
        loss_retarget = loss_retarget_full[loss_retarget_full > 0.0].mean()
        
        human_bones = human_joints_gt[:, human_child_idx] - human_joints_gt[:, human_parent_idx]
        robot_bones = output_trans[:, robot_child_idx] - output_trans[:, robot_parent_idx]
         
        scale_factors = link_scales.view(1, 12, 1)
        loss_pos = F.mse_loss(human_bones, robot_bones * scale_factors) 
        loss_ang = (1.0 - F.cosine_similarity(human_bones, robot_bones, dim=-1)).mean()
        loss_kinematic = loss_pos + loss_ang

        loss_scale_unit = F.mse_loss(link_scales, torch.ones_like(link_scales)) 

        reshaped_scales = link_scales.view(-1, 6)
        left_scales = reshaped_scales[:, :3].reshape(-1)
        right_scales = reshaped_scales[:, 3:].reshape(-1)
        loss_scale_symmetry = F.mse_loss(left_scales, right_scales) 

        foot_keypoints_pos = output_trans[:, ground_contact_indices]
        foot_height = foot_keypoints_pos[..., 2]
        loss_ground_z = foot_height**2
        weighted_loss_ground_z = loss_ground_z * contact_weights.squeeze()
        loss_ground_contact = weighted_loss_ground_z.mean()

        foot_velocity = torch.diff(foot_keypoints_pos, n=1, dim=0) / dt
        foot_velocity_horizontal = foot_velocity[..., :2]

        loss_skating_abs = torch.norm(foot_velocity_horizontal, dim=-1)

        contact_weights_for_skating = contact_weights[:-1, :]
        weighted_loss = loss_skating_abs * contact_weights_for_skating

        num_valid = (contact_weights_for_skating > 0.0).sum(dim=0)
        loss_skating = torch.where(
            num_valid > 0,
            weighted_loss.sum(dim=0) / (num_valid+1e-6),
            torch.tensor(0.0, device=retarget.device, dtype=torch.float32)
            ).sum()

        loss_joint_range = ((F.relu(0.98*min_vals - joint_pos) + F.relu(joint_pos - 0.98*max_vals))/2.0).mean() 
        joint_vel = torch.diff(joint_pos, n=1, dim=0) / dt 
        loss_vel_range = F.relu(torch.abs(joint_vel) - 0.98*vel_limits).mean() 

        loss = keypoint_matching_weight * loss_retarget \
             + smoothing_weight * loss_smooth \
             + kinematic_loss_weight * loss_kinematic \
             + scale_unit_loss_weight * loss_scale_unit \
             + scale_symmetry_loss_weight * loss_scale_symmetry \
             + joint_range_loss_weight * loss_joint_range \
             + joint_range_loss_weight * dt * loss_vel_range \
             + ground_contact_loss_weight * loss_ground_contact \
             + skating_loss_weight * loss_skating

        loss.backward()
        optimizer.step() 
        
        if i % print_every == 0:
            print(f"Iteration {i}: Loss = {loss.item()}") 
            print(f"\t Keypoint Matching Loss = {loss_retarget.item()}") 
            print(f"\t Smoothing Loss = {loss_smooth.item()}")
            print(f"\t Kinematic Positional Loss = {loss_pos.item()}") 
            print(f"\t Kinematic Angular Loss = {loss_ang.item()}") 
            print(f"\t Scale Unit Loss = {loss_scale_unit.item()}") 
            print(f"\t Scale Symmetry Loss = {loss_scale_symmetry.item()}") 
            print(f"\t Joint Range Loss = {loss_joint_range.item()}") 
            print(f"\t Joint Velocity Range Loss = {loss_vel_range.item()}")
            print(f"\t Ground Contact Loss = {loss_ground_contact.item()}")
            print(f"\t Skating Loss = {loss_skating.item()}")

    dof_pos = joint_pos.cpu().detach().numpy() 
    root_pos = root_trans.cpu().detach().numpy() 
    root_ori = root_ori_quat

    if args.low_pass_filter:
        root_pos_filtered = _butter_lowpass_filter(data=root_pos, cutoff=3.0, fs=fps)
        root_pos = np.ascontiguousarray(root_pos_filtered)

        quats = _unify_quaternion_signs(root_ori)
        quats_filtered = _butter_lowpass_filter(data=quats, cutoff=6.0, fs=fps)
        quats_filtered /= np.linalg.norm(quats_filtered, axis=1, keepdims=True)
        root_ori = np.ascontiguousarray(quats_filtered)

        dof_pos_filtered = _butter_lowpass_filter(data=dof_pos, cutoff=6.0, fs=fps)
        dof_pos = np.ascontiguousarray(dof_pos_filtered)

    link_orientation_error = np.degrees(np.arccos(1 - loss_ang.item()))

    keypoint = keypoint[:, [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13], :]
    keypoint_gt = keypoint_gt[:, [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13], :]
    g_mpjpe = torch.norm(keypoint - keypoint_gt, dim=-1).mean().item()

    # Foot contact distance
    foot_keypoints_pos = output_trans[:, ground_contact_indices, :].cpu().detach().numpy()
    foot_dist = np.abs(foot_keypoints_pos[:, :, 2])
    foot_contact_old = (foot_contact.cpu().detach().numpy()) == np.max(foot_contact.cpu().detach().numpy())
    foot_contact_dist = np.mean(foot_dist[foot_contact_old])

    # Skating velocity
    foot_velocity = np.diff(foot_keypoints_pos, n=1, axis=0) / dt
    foot_velocity_hor = np.linalg.norm(foot_velocity[:, :, :2], axis=-1)
    skate_mask = (foot_dist < 0.0025)[:-1, :]
    skating_velocity = np.mean(foot_velocity_hor[skate_mask])

    # Joint position violations
    min_vals = min_vals.cpu().detach().numpy()
    max_vals = max_vals.cpu().detach().numpy()

    min_violations = np.maximum(0, 0.98 * min_vals - dof_pos)
    max_violations = np.maximum(0, dof_pos - 0.98 * max_vals)
    # Check if any joint in each frame violates the threshold, then get fraction of violating frames
    frame_violations = np.any((min_violations > 0) | (max_violations > 0), axis=1)
    joint_violation = np.mean(frame_violations)

    print(f"G MPJPE: {g_mpjpe}")
    print(f"Joint violation: {joint_violation}")
    print(f"Foot contact distance: {foot_contact_dist}")
    print(f"Skating velocity: {skating_velocity}")

    np.clip(dof_pos, 0.98*min_vals, 0.98*max_vals, out=dof_pos)

    motion = np.concatenate([dof_pos, root_pos, root_ori], axis=1)
    os.makedirs(dirname(robot_pose_file), exist_ok=True)
    np.save(robot_pose_file, motion)

    if args.visualize:
        fps = args.fps
        robot_video_file = join(args.project_dir, "data", "video", "humanoid_pose", args.robot_name, f"{args.human_pose_file}.mp4")
        frames = render_robot_pose(robot_model_dir, dof_pos, root_pos, root_ori)
        write_video(robot_video_file, frames, fps=fps, reverse_rgb=True)

if __name__ == '__main__':
    args = parse_args()
    main(args)