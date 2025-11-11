import os
import argparse
from tqdm import tqdm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Motion-X (N,322) to PHUMA (N,69)")
    parser.add_argument("--human_pose_folder", required=True)
    parser.add_argument("--output_dir", type=str, default="data/human_pose")
    return parser.parse_args()

def main(args):
    converted, skipped, errors = 0, 0, 0
    
    for root, dirs, files in os.walk(args.human_pose_folder):
        for motion_file in tqdm(files, desc="Processing"):
            if not motion_file.endswith('.npy'):
                continue
                
            try:
                motion_file_path = os.path.join(root, motion_file)
                human_pose_motionx = np.load(motion_file_path)
                
                if human_pose_motionx.shape[1] != 322:
                    skipped += 1
                    continue
                
                # Concatenate in the order expected by PHUMA: [transl, global_orient, body_pose]
                human_pose_phuma = np.concatenate([
                    human_pose_motionx[:, 309:309+3],  # transl: (N, 3)
                    human_pose_motionx[:, 0:0+3],      # global_orient: (N, 3)
                    human_pose_motionx[:, 3:3+63]      # body_pose: (N, 63)
                ], axis=1)  # Shape: (N, 69)
                
                # Get relative path from parent of human_pose_folder
                # e.g., if human_pose_folder = '/path/to/human_pose/aist'
                #       and root = '/path/to/human_pose/aist/subset_0008'
                #       then rel_path = 'aist/subset_0008'
                parent_dir = os.path.dirname(args.human_pose_folder)
                rel_path = os.path.relpath(root, parent_dir)
                output_path = os.path.join(args.output_dir, rel_path, motion_file)
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                np.save(output_path, human_pose_phuma)
                converted += 1
                
            except Exception as e:
                print(f"\nError: {motion_file_path} - {e}")
                errors += 1
    
    print(f"\nDone: {converted} converted, {skipped} skipped, {errors} errors")

if __name__ == '__main__':
    args = parse_args()
    main(args)