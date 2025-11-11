import os
from os.path import join, exists, dirname
import argparse
import yaml
from multiprocessing import Pool, cpu_count

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
    parser.add_argument("--human_pose_folder", type=str, required=True)
    parser.add_argument("--visualize", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    
    # Multiprocessing parameters
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers (default: 1, use -1 for all CPUs)")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated GPU IDs for multi-GPU processing (e.g., '0,1,2,3')")
    parser.add_argument("--processes_per_gpu", type=int, default=1, help="Number of processes per GPU (default: 1). Total workers = num_gpus * processes_per_gpu")

    # Optimization parameters
    parser.add_argument("--lr_dof", type=float, default=0.005, help="Learning rate for DOF optimization")
    parser.add_argument("--num_iter_dof", type=int, default=3001, help="Number of optimization iterations")
    parser.add_argument("--print_every", type=int, default=500, help="Print loss every N iterations")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cuda:0, cpu, etc.)")
    
    # Loss weights
    parser.add_argument("--global_match_weight", type=float, default=0.1)
    parser.add_argument("--local_match_weight", type=float, default=2.0)
    parser.add_argument("--smooth_weight", type=float, default=0.05)
    parser.add_argument("--scale_unit_loss_weight", type=float, default=0.001, help="Weight for scale unit loss")
    parser.add_argument("--scale_symmetry_loss_weight", type=float, default=0.001, help="Weight for scale symmetry loss")
    parser.add_argument("--joint_feasibility_weight", type=float, default=1000.0)
    parser.add_argument("--grounding_weight", type=float, default=10.0)
    parser.add_argument("--skating_weight", type=float, default=0.002)
    
    # Link scales
    parser.add_argument("--num_link_scales", type=int, default=12, help="Number of link scales")
    
    # Low pass filter parameters
    parser.add_argument("--low_pass_filter", type=int, default=1)
    parser.add_argument("--root_pos_cutoff", type=float, default=3.0, help="Low pass filter cutoff for root position (Hz)")
    parser.add_argument("--root_ori_cutoff", type=float, default=6.0, help="Low pass filter cutoff for root orientation (Hz)")
    parser.add_argument("--dof_pos_cutoff", type=float, default=6.0, help="Low pass filter cutoff for DOF position (Hz)")
    
    # Joint range clipping
    parser.add_argument("--joint_range_margin", type=float, default=0.98, help="Joint range clipping margin (0.98 = 98% of range)")
    
    # Skating velocity threshold
    parser.add_argument("--skating_distance_threshold", type=float, default=0.0025, help="Distance threshold for skating velocity calculation (meters)")

    return parser.parse_args()

def process_single_file(args_tuple):
    """Process a single motion file. This function is called by multiprocessing workers."""
    motion_file_path, root, args, device = args_tuple
    
    try:
        # Get filename without extension
        motion_file = os.path.basename(motion_file_path)
        base_name = os.path.splitext(motion_file)[0]
        
        # Construct output paths
        robot_pose_file = join(args.project_dir, "data", "humanoid_pose", args.robot_name, motion_file)
        
        human_model_dir = join(args.project_dir, "asset", "human_model")
        robot_model_dir = join(args.project_dir, "asset", "humanoid_model", args.robot_name)

        motion_parms = load_motion_parms(motion_file_path, foot_contact=True)
        betas = np.load(join(robot_model_dir, "betas.npy"))
        motion_parms["betas"] = torch.from_numpy(betas).unsqueeze(0).repeat(motion_parms["body_pose"].shape[0], 1).float()

        smpl = smplx.create(human_model_dir, model_type="smplx", num_pca_comps=45)

        keypoint_matching_weight = args.global_match_weight
        kinematic_loss_weight = args.local_match_weight
        smoothing_weight = args.smooth_weight
        scale_unit_loss_weight = args.scale_unit_loss_weight
        scale_symmetry_loss_weight = args.scale_symmetry_loss_weight
        joint_range_loss_weight = args.joint_feasibility_weight
        ground_contact_loss_weight = args.grounding_weight
        skating_loss_weight = args.skating_weight

        dt = 1.0 / args.fps

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
        retarget = HumanoidRetargetKeypoint(mjcf_file=mjcf_file, device=device)

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
         
        link_scales = torch.ones(args.num_link_scales, device=retarget.device, dtype=torch.float32, requires_grad=True) 

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

        optimizer = torch.optim.Adam([joint_pos, root_trans, link_scales], lr=args.lr_dof) 

        keypoint_gt = torch.from_numpy(keypoint_traj).to(retarget.device).to(torch.float32) 
        human_joints_gt = torch.from_numpy(human_joints_traj).to(retarget.device, dtype=torch.float32)
        
        contact_weights = foot_contact.to(retarget.device)

        min_vals = retarget.robot.joints_range[:, 0] 
        max_vals = retarget.robot.joints_range[:, 1] 

        vel_limits = torch.tensor(robot_config['joint_velocity_limits'], device=retarget.device, dtype=torch.float32) 

        for i in tqdm(range(args.num_iter_dof)):
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
             
            scale_factors = link_scales.view(1, args.num_link_scales, 1)
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

            loss_joint_range = ((F.relu(args.joint_range_margin*min_vals - joint_pos) + F.relu(joint_pos - args.joint_range_margin*max_vals))/2.0).mean() 
            joint_vel = torch.diff(joint_pos, n=1, dim=0) / dt 
            loss_vel_range = F.relu(torch.abs(joint_vel) - args.joint_range_margin*vel_limits).mean() 

            loss = keypoint_matching_weight * loss_retarget \
                 + kinematic_loss_weight * loss_kinematic \
                 + smoothing_weight * loss_smooth \
                 + scale_unit_loss_weight * loss_scale_unit \
                 + scale_symmetry_loss_weight * loss_scale_symmetry \
                 + joint_range_loss_weight * loss_joint_range \
                 + joint_range_loss_weight * dt * loss_vel_range \
                 + ground_contact_loss_weight * loss_ground_contact \
                 + skating_loss_weight * loss_skating

            loss.backward()
            optimizer.step() 
            
            if i % args.print_every == 0:
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
            root_pos_filtered = _butter_lowpass_filter(data=root_pos, cutoff=args.root_pos_cutoff, fs=args.fps)
            root_pos = np.ascontiguousarray(root_pos_filtered)

            quats = _unify_quaternion_signs(root_ori)
            quats_filtered = _butter_lowpass_filter(data=quats, cutoff=args.root_ori_cutoff, fs=args.fps)
            quats_filtered /= np.linalg.norm(quats_filtered, axis=1, keepdims=True)
            root_ori = np.ascontiguousarray(quats_filtered)

            dof_pos_filtered = _butter_lowpass_filter(data=dof_pos, cutoff=args.dof_pos_cutoff, fs=args.fps)
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
        skate_mask = (foot_dist < args.skating_distance_threshold)[:-1, :]
        skating_velocity = np.mean(foot_velocity_hor[skate_mask])

        # Joint position violations
        min_vals = min_vals.cpu().detach().numpy()
        max_vals = max_vals.cpu().detach().numpy()

        min_violations = np.maximum(0, args.joint_range_margin * min_vals - dof_pos)
        max_violations = np.maximum(0, dof_pos - args.joint_range_margin * max_vals)
        # Check if any joint in each frame violates the threshold, then get fraction of violating frames
        frame_violations = np.any((min_violations > 0) | (max_violations > 0), axis=1)
        joint_violation = np.mean(frame_violations)

        print(f"G MPJPE: {g_mpjpe}")
        print(f"Joint violation: {joint_violation}")
        print(f"Foot contact distance: {foot_contact_dist}")
        print(f"Skating velocity: {skating_velocity}")

        np.clip(dof_pos, args.joint_range_margin*min_vals, args.joint_range_margin*max_vals, out=dof_pos)

        motion = {
            "root_trans": root_pos,
            "root_ori": root_ori,
            "dof_pos": dof_pos,
            "fps": args.fps
        }
        os.makedirs(dirname(robot_pose_file), exist_ok=True)
        np.save(robot_pose_file, motion)

        if args.visualize:
            robot_video_file = join(args.project_dir, "data", "video", "humanoid_pose", args.robot_name, f"{base_name}.mp4")
            os.makedirs(dirname(robot_video_file), exist_ok=True)
            frames = render_robot_pose(robot_model_dir, dof_pos, root_pos, root_ori)
            write_video(robot_video_file, frames, fps=args.fps, reverse_rgb=True)
        
        return f"Success: {motion_file}"
    except Exception as e:
        return f"Error: {motion_file_path} - {str(e)}"

def main(args):
    if not os.path.exists(args.human_pose_folder):
        print(f"Error: Input folder does not exist: {args.human_pose_folder}")
        return
    
    # Parse GPU IDs if provided
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(',')]
        print(f"Using GPUs: {gpu_ids}")
    
    # Determine number of workers
    num_workers = args.num_workers
    
    # If gpu_ids provided, calculate total workers based on processes_per_gpu
    if gpu_ids:
        calculated_workers = len(gpu_ids) * args.processes_per_gpu
        if num_workers == 1:  # Default value, use calculated
            num_workers = calculated_workers
            print(f"Auto-calculated workers: {len(gpu_ids)} GPUs × {args.processes_per_gpu} processes/GPU = {num_workers} workers")
        else:  # User specified num_workers, show info if different
            if num_workers != calculated_workers:
                print(f"Info: num_workers ({num_workers}) != GPUs ({len(gpu_ids)}) × processes_per_gpu ({args.processes_per_gpu}) = {calculated_workers}")
                print(f"Using specified num_workers: {num_workers}")
    
    if num_workers == -1:
        num_workers = cpu_count()
        print(f"Using all available CPU cores: {num_workers}")
    
    # Collect all motion file paths
    motion_file_paths = []
    for root, dirs, files in os.walk(args.human_pose_folder):
        for motion_file in files:
            if motion_file.endswith('.npy'):
                motion_file_path = os.path.join(root, motion_file)
                motion_file_paths.append((motion_file_path, root))
    
    # Create worker-device mapping
    worker_devices = []
    if gpu_ids:
        # Assign GPUs to workers based on processes_per_gpu
        for gpu_id in gpu_ids:
            for _ in range(args.processes_per_gpu):
                worker_devices.append(f"cuda:{gpu_id}")
        # If num_workers > calculated workers, fill remaining with round-robin
        while len(worker_devices) < num_workers:
            gpu_idx = len(worker_devices) % len(gpu_ids)
            worker_devices.append(f"cuda:{gpu_ids[gpu_idx]}")
    else:
        # All workers use the same device
        worker_devices = [args.device] * num_workers
    
    # Assign files to workers with their devices
    motion_files = []
    for idx, (motion_file_path, root) in enumerate(motion_file_paths):
        # Assign device in round-robin fashion across workers
        device = worker_devices[idx % len(worker_devices)]
        motion_files.append((motion_file_path, root, args, device))
    
    print(f"Found {len(motion_files)} motion files to process")
    print(f"Using {num_workers} worker(s)")
    
    # Print device information
    if gpu_ids:
        device_str = str(worker_devices[:min(len(worker_devices), 8)])
        if len(worker_devices) > 8:
            device_str = device_str[:-1] + ", ...]"
        print(f"Worker-GPU mapping: {device_str}")
    else:
        print(f"Device mode: {args.device}")
    
    if num_workers == 1:
        # Sequential processing
        print("Running in sequential mode...")
        results = []
        for motion_file_tuple in tqdm(motion_files, desc="Processing motion files"):
            result = process_single_file(motion_file_tuple)
            results.append(result)
            print(result)
    else:
        # Parallel processing
        print("Running in parallel mode...")
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_file, motion_files),
                total=len(motion_files),
                desc="Processing motion files"
            ))
    
    # Print all results
    print("\n" + "="*50)
    print("Processing Summary:")
    print("="*50)
    for result in results:
        print(result)
    
    # Count successes and errors
    successes = sum(1 for r in results if r.startswith("Success"))
    errors = sum(1 for r in results if r.startswith("Error"))
    print("="*50)
    print(f"Total files: {len(results)}")
    print(f"Successfully processed: {successes}")
    print(f"Errors: {errors}")
    print("="*50)

if __name__ == '__main__':
    args = parse_args()
    main(args)