import torch
import numpy as np

from easydict import EasyDict
import xml.etree.ElementTree as ETree

import pytorch_kinematics.transforms.rotation_conversions as tRot


class Humanoid_Batch:
    def __init__(self, mjcf_file = "", extend_hand = True, extend_head = False, device = "cuda:0"):
        self.mjcf_data = mjcf_data = self.from_mjcf(mjcf_file)
        self.extend_hand = extend_hand
        self.extend_head = extend_head

        assert not extend_hand
        self._parents = mjcf_data['parent_indices']
        self.model_names = mjcf_data['node_names']
        self._offsets = mjcf_data['local_translation'][None, ].to(device)
        self._local_rotation = mjcf_data['local_rotation'][None, ].to(device)
        # print(self._local_rotation)
            
        assert not extend_head

        self.joints_range = mjcf_data['joints_range'].to(device)
        self._local_rotation_mat = tRot.quaternion_to_matrix(self._local_rotation).float() # w, x, y ,z
        
    def from_mjcf(self, path):
        # function from Poselib: 
        tree = ETree.parse(path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        if xml_world_body is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
        # assume this is the root
        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
            
        # xml_joint_root = xml_body_root.find("joint")
        
        node_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints_range = []

        # recursively adding all nodes into the skel_tree
        def _add_xml_node(xml_node, parent_index, node_index):
            node_name = xml_node.attrib.get("name")
            # parse the local translation into float list
            pos = np.fromstring(xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" ")
            quat = np.fromstring(xml_node.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" ")
            node_names.append(node_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(quat)
            curr_index = node_index
            node_index += 1
            all_joints = xml_node.findall("joint")
            for joint in all_joints:
                if not joint.attrib.get("range") is None: 
                    joints_range.append(np.fromstring(joint.attrib.get("range"), dtype=float, sep=" "))
            
            for next_node in xml_node.findall("body"):
                node_index = _add_xml_node(next_node, curr_index, node_index)
            return node_index
        
        _add_xml_node(xml_body_root, -1, 0)
        return {
            "node_names": node_names,
            "parent_indices": torch.from_numpy(np.array(parent_indices, dtype=np.int32)),
            "local_translation": torch.from_numpy(np.array(local_translation, dtype=np.float32)),
            "local_rotation": torch.from_numpy(np.array(local_rotation, dtype=np.float32)),
            "joints_range": torch.from_numpy(np.array(joints_range))
        }

        
    def fk_batch(self, pose, trans, convert_to_mat=True, return_full = False):
        # device, dtype = pose.device, pose.dtype
        # pose_input = pose.clone()
        B, seq_len = pose.shape[:2]
        pose = pose[..., :len(self._parents), :] # H1 fitted joints might have extra joints
        assert not self.extend_hand

        if convert_to_mat:
            pose_quat = tRot.axis_angle_to_quaternion(pose)
            pose_mat = tRot.quaternion_to_matrix(pose_quat)
        else:
            pose_mat = pose
        if pose_mat.shape != 5:
            pose_mat = pose_mat.reshape(B, seq_len, -1, 3, 3)
        # J = pose_mat.shape[2] - 1  # Exclude root
        
        wbody_pos, wbody_mat = self.forward_kinematics_batch(pose_mat[:, :, 1:], pose_mat[:, :, 0:1], trans)
        
        return_dict = EasyDict()
        
        wbody_rot = tRot.wxyz_to_xyzw(tRot.matrix_to_quaternion(wbody_mat))
        assert not self.extend_hand
        
        return_dict.global_translation = wbody_pos
        return_dict.global_rotation_mat = wbody_mat
        return_dict.global_rotation = wbody_rot

        assert not return_full
        
        return return_dict
    

    def forward_kinematics_batch(self, rotations, root_rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where B = batch size, J = number of joints):
         -- rotations: (B, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (B, 3) tensor describing the root joint positions.
        Output: joint positions (B, J, 3)
        """
        
        device, dtype = root_rotations.device, root_rotations.dtype
        B, seq_len = rotations.size()[0:2]
        J = self._offsets.shape[1]
        positions_world = []
        rotations_world = []

        expanded_offsets = (self._offsets[:, None].expand(B, seq_len, J, 3).to(device).type(dtype))

        for i in range(J):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
            else:
                jpos = (torch.matmul(rotations_world[self._parents[i]][:, :, 0], expanded_offsets[:, :, i, :, None]).squeeze(-1) + positions_world[self._parents[i]])
                rot_mat = torch.matmul(rotations_world[self._parents[i]], torch.matmul(self._local_rotation_mat[:,  (i):(i + 1)], rotations[:, :, (i - 1):i, :]))

                positions_world.append(jpos)
                rotations_world.append(rot_mat)
        
        positions_world = torch.stack(positions_world, dim=2)
        rotations_world = torch.cat(rotations_world, dim=2)
        return positions_world, rotations_world


class Config: pass


class HumanoidRetargetKeypoint:
    # def __init__(self, mjcf_file, root_translation, root_orient, keypoint_trans, device='cuda:0'): # , num_joints=27, num_bodies=36):
    def __init__(self, mjcf_file, device='cuda:0'): # , num_joints=27, num_bodies=36):
        self.device = device

        # load from data
        # self.data = Config()
        # self.data.num_frames = root_translation.shape[0]
        # self.data.root_translation = root_translation
        # self.data.root_orient = root_orient
        # self.data.keypoint_trans = keypoint_trans

        # H1m parameters
        self.robot = Config()
        self.robot.kinematics = Humanoid_Batch(
            mjcf_file=mjcf_file,
            device=self.device,
            extend_hand=False,
            extend_head=False
        )
        # print(self.robot.kinematics.mjcf_data)
        self.robot.mjcf_file = mjcf_file
        self.robot.node_names = self.robot.kinematics.mjcf_data['node_names']
        self.robot.parent_indices = self.robot.kinematics.mjcf_data['parent_indices']
        self.robot.local_translation = self.robot.kinematics.mjcf_data['local_translation'].to(torch.float32).to(self.device)
        self.robot.local_rotation = self.robot.kinematics.mjcf_data['local_rotation'].to(torch.float32).to(self.device)
        self.robot.joints_range = self.robot.kinematics.mjcf_data['joints_range'].to(torch.float32).to(self.device)
        # self.robot.num_joints = num_joints
        # self.robot.num_bodies = num_bodies