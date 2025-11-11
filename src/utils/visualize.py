import os
from os.path import dirname, join
import yaml

import numpy as np
import mujoco
import pyrender
import trimesh
import cv2
from tqdm import tqdm
from smplx.joint_names import JOINT_NAMES


def render_smpl_pose(smpl, smpl_kp_config, motion_parms):
    smpl_joint_map = {name: i for i, name in enumerate(JOINT_NAMES)}
    smpl_body_names = [kp["body"] for kp in smpl_kp_config["keypoints"] if kp["body"] and kp["body"] != "pelvis"]
    smpl_keypoint_indices = [smpl_joint_map[name] for name in smpl_body_names]

    viewport_width = 1280
    viewport_height = 960
    renderer = pyrender.OffscreenRenderer(viewport_width, viewport_height)

    smpl_vertices_all_frame = []
    smpl_joints_all_frame = []

    transl = motion_parms["transl"]
    num_frames = transl.shape[0]

    for i in range(num_frames):
        motion_parms_frame = {k: v[None, i] for k, v in motion_parms.items()}

        output = smpl(**motion_parms_frame)
        vertices = output.vertices.detach().cpu().numpy().squeeze()

        smpl_vertices_all_frame.append(output.vertices.detach().cpu().numpy().squeeze())
        smpl_joints_all_frame.append(output.joints.detach().cpu().numpy().squeeze())

    avg_width = 0.5 * (min(transl[:, 0]) + max(transl[:, 0]))
    avg_depth = 0.5 * (min(transl[:, 2]) + max(transl[:, 2]))
    avg_height = 0.5 * (min(transl[:, 1]) + max(transl[:, 1]))

    frames = []

    SMPL_COLOR = [0.98, 0.855, 0.369, 1.0]
    JOINT_COLOR = [0.1, 0.1, 0.9, 1.0]
    FLOOR_COLOR = [0.8, 0.8, 0.8, 0.3]
    BG_COLOR = [0.98, 0.98, 0.98, 1.0]

    for vertices, joints in tqdm(zip(smpl_vertices_all_frame, smpl_joints_all_frame), total=num_frames):
        vertices = vertices - [avg_width, avg_height, avg_depth]
        joints = joints[smpl_keypoint_indices] - [avg_width, avg_height, avg_depth]

        vertex_colors = np.ones([vertices.shape[0], 4]) * SMPL_COLOR
        tri_mesh = trimesh.Trimesh(vertices, smpl.faces, vertex_colors=vertex_colors)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        scene = pyrender.Scene(bg_color=BG_COLOR)
        scene.add(mesh)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.5)
        cam_pose = np.eye(4)
        cam_pose[2, 3] = 2
        scene.add(camera, pose=cam_pose)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=cam_pose)

        plane_thickness = 0.001
        plane_width = 50
        plane_depth = 50
        plane_box = trimesh.primitives.Box(
            extents=[plane_width, plane_depth, plane_thickness],
            transform=np.eye(4)
        )
        R = trimesh.transformations.rotation_matrix(
            np.radians(-90), [1, 0, 0]
        )
        plane_box.apply_transform(R)
        plane_box.apply_translation([0, plane_thickness / -2.0 - avg_height, 0])
        
        plane_color = np.ones((plane_box.vertices.shape[0], 4), dtype=np.float32) * FLOOR_COLOR
        plane_box.visual.vertex_colors = plane_color
        plane_mesh = pyrender.Mesh.from_trimesh(plane_box, smooth=False)
        scene.add(plane_mesh)

        color, _ = renderer.render(scene)
        frames.append(color)

    renderer.delete()

    return frames


def render_robot_pose(robot_dir, dof_pos, root_pos, root_ori):
    scene_path = join(robot_dir, "scene.xml")
    config_path = join(robot_dir, "config.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    robot = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(robot)

    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = 0
    camera.distance = 3.5
    camera.elevation = -20.0
    camera.azimuth = -140.0

    ROOT_POS = np.array(config['root_pos'])
    ROOT_ORI = np.array(config['root_ori'])
    DOF_POS = np.array(config['dof_pos'])

    def set_qpos(root_pos=ROOT_POS, root_ori=ROOT_ORI, dof_pos=DOF_POS):
        qpos = np.zeros(len(root_pos) + len(root_ori) + len(dof_pos))
        qpos[0: 3] = root_pos
        qpos[3: 7] = root_ori[..., [3, 0, 1, 2]]
        qpos[7:  ] = dof_pos

        return qpos

    data.qpos = set_qpos()
    mujoco.mj_resetData(robot, data)

    num_frames = len(dof_pos)
    frames = []
    renderer = mujoco.Renderer(robot, 480, 640)
    
    for i in range(num_frames):
        data.qpos = set_qpos(root_pos=root_pos[i], root_ori=root_ori[i], dof_pos=dof_pos[i])
        mujoco.mj_forward(robot, data)

        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        pixels = renderer.render()

        frames.append(pixels)

    renderer.close()
    del renderer
    
    return frames


def write_video(file, frames, fps=30, reverse_rgb=False):
    frame_height, frame_width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    os.makedirs(dirname(file), exist_ok=True)
    out = cv2.VideoWriter(file, fourcc, fps, (frame_width, frame_height))

    for frame in frames:
        if reverse_rgb:
            out.write(frame[...,::-1])
        else:
            out.write(frame)

    out.release()