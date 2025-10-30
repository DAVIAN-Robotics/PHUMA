import os
from os.path import join, dirname
import argparse
import yaml

import numpy as np
import smplx
import torch

from utils.smpl import *
from utils.visualize import render_smpl_pose, write_video

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", required=True)
    parser.add_argument("--human_pose_file", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--visualize", type=int, default=0)
    return parser.parse_args()

def main(args):
    human_model_dir = join(args.project_dir, "asset", "human_model")

    smpl = smplx.create(human_model_dir, model_type="smplx", num_pca_comps=45)

    smpl_config_path = join(human_model_dir, "config.yaml")
    with open(smpl_config_path, 'r') as file:
        smpl_config = yaml.safe_load(file)

    foot_contact_vertex_indices = {}
    for key in ["left_toe_indices", "left_heel_indices", "right_toe_indices", "right_heel_indices"]:
        foot_contact_vertex_indices[key] = smpl_config[key]

    motion_file = join(args.project_dir, "data", "human_pose", f"{args.human_pose_file}.npy")    
    motion_parms = load_motion_parms(motion_file, foot_contact=False)
    N = motion_parms['body_pose'].shape[0]
    if N > args.fps // 2:
        motion_parms = low_pass_filter(motion_parms, args.fps)

    vertices = []
    joints = []

    for i in range(N):
        frame_params = {k: v[i:i+1] for k, v in motion_parms.items()}
        output = smpl(**frame_params)
        vertices.append(output.vertices.detach().cpu().numpy().squeeze())
        joints.append(output.joints.detach().cpu().numpy().squeeze())

    vertices = np.array(vertices)
    joints = np.array(joints)

    robust_ground_y = find_robust_ground(vertices, foot_contact_vertex_indices)

    foot_contacts = get_foot_contact(vertices, foot_contact_vertex_indices, robust_ground_y)

    motion_parms['transl'][:, 1] -= robust_ground_y
    joints[..., 1] -= robust_ground_y

    transl = motion_parms['transl'].cpu().numpy()
    global_orient = motion_parms['global_orient'].cpu().numpy()
    body_pose = motion_parms['body_pose'].cpu().numpy()

    preprocessed_data = np.concatenate([transl, global_orient, body_pose, foot_contacts], axis=1)

    dt = 1.0 / args.fps

    chunk_size = int(4 * args.fps)
    chunk_overlap = int(0.5 * args.fps)
    chunk_min_frames = int(1 * args.fps)

    all_metrics = []
    start_frame = 0
    chunk_id = 0

    while start_frame < N:
        chunk_file = f"{args.human_pose_file}_chunk_{chunk_id:04d}"

        end_frame = start_frame + chunk_size
        if end_frame + (chunk_min_frames - chunk_overlap) > N:
            end_frame = N
        else:
            end_frame = min(end_frame, N)

        chunk_data = preprocessed_data[start_frame:end_frame].copy()
        N_chunk = chunk_data.shape[0]
        if N_chunk < chunk_min_frames:
            break

        chunk_joints = joints[start_frame:end_frame].copy()

        # Foot Contact Score
        foot_contact_score = np.max(chunk_data[:, 69:69+4], axis=-1).mean()
        if foot_contact_score <= 0.6:
            print(f"[FILTER OUT] File: {chunk_file} - Foot contact score {foot_contact_score} is lower than 0.6.")
            continue
        else:
            print(f"[PASS] File: {chunk_file} - Foot contact score {foot_contact_score} is higher than 0.6.")

        # Root Jerk
        velocity = np.diff(chunk_data[:, :3], axis=0) / dt
        acceleration = np.diff(velocity, axis=0) / dt
        jerk = np.diff(acceleration, axis=0) / dt
        jerk_magnitude = np.linalg.norm(jerk, axis=1)
        root_jerk = np.mean(jerk_magnitude)
        if root_jerk >= 50:
            print(f"[FILTER OUT] File: {chunk_file} - Root jerk {root_jerk} is higher than 50.")
            continue
        else:
            print(f"[PASS] File: {chunk_file} - Root jerk {root_jerk} is lower than 50.")

        # Pelvis Height
        min_pelvis_height = chunk_joints[:, 0, 1].min()
        if min_pelvis_height <= 0.6:
            print(f"[FILTER OUT] File: {chunk_file} - Min pelvis height {min_pelvis_height} lower than 0.6.")
            continue
        else:
            print(f"[PASS] File: {chunk_file} - Min pelvis height {min_pelvis_height} is higher than 0.6.")
        max_pelvis_height = chunk_joints[:, 0, 1].max()
        if max_pelvis_height >= 1.5:
            print(f"[FILTER OUT] File: {chunk_file} - Max pelvis height {max_pelvis_height} higher than 1.5.")
            continue
        else:
            print(f"[PASS] File: {chunk_file} - Max pelvis height {max_pelvis_height} is lower than 1.5.")

        # Distance to Base of Support
        pelvis_to_bos_distance = calculate_bos_distance(chunk_joints, target_joint_id=0)
        if pelvis_to_bos_distance >= 0.06:
            print(f"[FILTER OUT] File: {chunk_file} - Pelvis to BoS distance {pelvis_to_bos_distance} exceeds 0.06.")
            continue
        else:
            print(f"[PASS] File: {chunk_file} - Pelvis to BoS distance {pelvis_to_bos_distance} is lower than 0.06.")
        spine1_to_bos_distance = calculate_bos_distance(chunk_joints, target_joint_id=3)
        if spine1_to_bos_distance >= 0.11:
            print(f"[FILTER OUT] File: {chunk_file} - Spine1 to BoS distance {spine1_to_bos_distance} exceeds 0.11.")
            continue
        else:
            print(f"[PASS] File: {chunk_file} - Spine1 to BoS distance {spine1_to_bos_distance} is lower than 0.11.")

        preprocessed_motion_file = join(args.project_dir, "data", "human_pose_preprocessed", f"{chunk_file}.npy")
        os.makedirs(dirname(preprocessed_motion_file), exist_ok=True)
        np.save(preprocessed_motion_file, chunk_data)

        chunk_motion_parms = {
            'transl': torch.from_numpy(chunk_data[:, 0:3]).float().clone(),
            'global_orient': torch.from_numpy(chunk_data[:, 3:6]).float().clone(),
            'body_pose': torch.from_numpy(chunk_data[:, 6:69]).float().clone(),
        }

        if args.visualize > 0:
            video_file = join(args.project_dir, "data", "video", "human_pose_preprocessed", f"{chunk_file}.mp4")
            frames = render_smpl_pose(smpl, smpl_config, chunk_motion_parms)
            write_video(video_file, frames, fps=args.fps)

        start_frame += (chunk_size - chunk_overlap)
        chunk_id += 1

if __name__ == '__main__':
    args = parse_args()
    main(args)