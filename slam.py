
import numpy as np
from optimizers import BundleAdjuster
import g2o
import torch
from datetime import datetime
from pathlib import Path
import os
import pickle
import shutil
from scipy.spatial.transform import Rotation as R
from datasets.dataset import ImageDataset
import cv2

from visualizations import *

# Helper functions
def mat_to_vecs(mat):
    trans_vec = mat[:3, 3]
    rot_vec = R.from_matrix(mat[:3, :3]).as_rotvec()
    return trans_vec.astype(np.float32), rot_vec.astype(np.float32)

def vecs_to_mat(trans_vec, rot_vec):
    mat = np.identity(4)
    mat[:3, 3] = trans_vec
    mat[:3, :3] = R.from_rotvec(rot_vec).as_matrix()
    return mat


# Main SLAM datastructure
# As of now, also responsible for optimization.
class SLAMStructure:
    def __init__(self, name='', output_folder='./experiments', BA_sparse_solver=True, BA_verbose=False, BA_opt_iters=20) -> None:
        # Internal storage
        self.keyframes = []
        self.poses = dict() # frame_idx: (pose, intrinsics)

        self.points = dict() # Set of point instances {id: (3d, color)}
        self.pose_point_map = dict() # Pose to point associations {pose_id: [...(point_id, 2d)]}
        self.pose_point_edges = dict() # Stores edges for every pose 


        self.pose_images = dict() # Store keyframe images for visualizations
        self.pose_depths = dict() # Store keyframe depths for possible later use
        self.pose_masks =  dict() # Store keyframe masks for possible later use

        self.valid_points = set() # Points that have been extracted from a BA run
        self.valid_frames = set() # Poses that are localized in the current map


        # Bundle adjuster + associated settings
        self.BA_sparse_solver = BA_sparse_solver
        self.BA_verbose = BA_verbose
        self.BA_opt_iters = BA_opt_iters

        self.BA = BundleAdjuster(use_sparse_solver=self.BA_sparse_solver)
        self.BA.set_verbose(self.BA_verbose)

        # Other Settings
        now = datetime.now()
        self.exp_name = now.strftime("exp_%m%d%Y_%H:%M:%S") if name == '' else name
        self.exp_root = Path(output_folder) / self.exp_name
    
    # Registering new frames

    def add_frame(self, frame_idx, pose, intrinsics):
        # Register a new frame into the SLAM structure

        # Assert that frame has not been added already
        assert frame_idx not in self.poses.keys()

        # Add new pose
        self.poses[frame_idx] = (pose, intrinsics)
        self.pose_point_map[frame_idx] = []

    def make_keyframe(self, frame_idx, image, depth, mask, fixed=False) -> None:
        # Make an existing frame into a new keyframe

        # Assert that frame has been registered and is not already a keyframe
        assert frame_idx in self.poses.keys()
        assert frame_idx not in self.keyframes

        # Include frame in bundle adjustment graph
        
        # Add pose to BA
        pose, intrinsics = self.poses[frame_idx]
        self.BA.add_pose(frame_idx, g2o.Isometry3d(pose), intrinsics, fixed=fixed)
        
        # Add existing correspondences to BA
        for (point_id, point_2d) in self.pose_point_map[frame_idx]:
            edge = self.BA.add_edge(point_id, frame_idx, point_2d)
            if frame_idx not in self.pose_point_edges.keys():
                self.pose_point_edges[frame_idx] = []
            self.pose_point_edges[frame_idx].append(edge)
        

        # Store additional information for later use (visualization, ...)
        self.pose_images[frame_idx] = image
        self.pose_depths[frame_idx] = depth
        self.pose_masks[frame_idx] = mask

        # Add frame to keyframes
        self.keyframes.append(frame_idx)

    # Registering new points and correspondences

    def add_point(self, point_3d, point_color):
        # Adding a new point

        # Get id for new point
        point_id = len(self.points)

        # Add point to current map
        # Fix first added point in order to preserve scale
        self.points[point_id] = (point_3d, point_color)
        self.BA.add_point(point_id, point_3d)

        return point_id

    def add_correspondence(self, frame_idx, point_id, point_2d):
        # Add a new correspondence
        
        # Assert that point and frame are registered
        assert point_id in self.points.keys()
        assert frame_idx in self.poses.keys()

        # Check if correspondence has already been added to this frame
        for i in range(len(self.pose_point_map[frame_idx])):
            if self.pose_point_map[frame_idx][i][0] == point_id:
                return
        
        # Add new correspondence
        self.pose_point_map[frame_idx].append((point_id, point_2d))
            
        # If frame_idx is a keyframe, also include corresponce in BA graph
        if frame_idx in self.keyframes:
            edge = self.BA.add_edge(point_id, frame_idx, point_2d)
            if frame_idx not in self.pose_point_edges.keys():
                self.pose_point_edges[frame_idx] = []
            self.pose_point_edges[frame_idx].append(edge)

    # Retrieving data

    def get_previous_poses(self, n):
        # Return the last n pose estimates
        sorted_pose_keys = sorted(list(self.poses.keys()))
        poses = [] 
        for key in sorted_pose_keys[-n:]:
            pose, _ = self.poses[key]
            poses.append(pose)
        
        return poses 

    def get_point(self, point_id):
        assert point_id in self.points.keys()
        return self.points[point_id]

    def get_pose_points(self, frame_idx):
        assert frame_idx in self.poses.keys()
        return self.pose_point_map[frame_idx].copy()

    # Pose estimation/Mapping functions
    
    def localize_frame(self, frame_idx, update_pose=False, ransac=False):
        # Localize a frame against current map

        assert frame_idx in self.poses.keys()

        current_pose, intrinsics = self.poses[frame_idx]
        
        pose_points = self.pose_point_map[frame_idx]
        pose_points_filtered = list(filter(lambda pair: pair[0] in self.valid_points, pose_points))

        if len(pose_points_filtered) <= 5:
            return current_pose
        
        trans_vec, rot_vec = mat_to_vecs(np.linalg.inv(current_pose))
        
        obj_pts = []
        img_pts = []

        for (point_id, point_2d) in pose_points_filtered:
            obj_pts.append(self.points[point_id][0])
            img_pts.append(point_2d)

        obj_pts = np.array(obj_pts)
        img_pts = np.array(img_pts)
        intrinsics_mat = np.identity(3)
        intrinsics_mat[0, 0] = intrinsics[0]
        intrinsics_mat[1, 1] = intrinsics[1]
        intrinsics_mat[0, 2] = intrinsics[2]
        intrinsics_mat[1, 2] = intrinsics[3]
        distCoeffs = np.array([0., 0., 0., 0.])

        if ransac:
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(obj_pts, img_pts, intrinsics_mat, distCoeffs, rvec=rot_vec, tvec=trans_vec)
        else:
            retval, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, intrinsics_mat, distCoeffs, rvec=rot_vec, tvec=trans_vec)


        localized_pose = np.linalg.inv(vecs_to_mat(tvec.squeeze(), rvec.squeeze()))

        if update_pose:
            self.poses[frame_idx] = (localized_pose, intrinsics)
            self.valid_frames.add(frame_idx)

        return localized_pose
    
    # Running bundle adjustment

    def run_ba(self, opt_iters=None):
        # Run bundle adjustment

        if opt_iters is not None:
            self.BA.optimize(opt_iters)
        else:
            self.BA.optimize(self.BA_opt_iters)

        self.valid_frames = set()

        self.extract_ba_data(self.BA)

    def extract_ba_data(self, BA_instance):
        # Copy data from ba graph to class storage

        for keyframe in self.keyframes:
            _, intrinsics = self.poses[keyframe]
            self.poses[keyframe] = (BA_instance.get_pose(keyframe).matrix(), intrinsics)
            
            self.valid_frames.add(keyframe)


        for point_id in self.points.keys():
            _, point_color = self.points[point_id]
            self.points[point_id] = (BA_instance.get_point(point_id), point_color)
            
            self.valid_points.add(point_id)

    

    def filter(self, max_allowed_distance=None, reprojection_error_threshold=1000000, min_view_num=0):
        """
        Only retain points and correspondences that are 
        - close to some camera pose
        - visible on at least a minimum number of views
        - have a low reprojection error
        """
        # TODO: DOES NOT UPDATE BA GRAPH, ONLY DICTS! Might have to construct a seperate BA graph for every optimization.
        # TODO: Also filter correspondences in non-keyframes
        eps = 1e-6
        
        # Collect minimum cam distances for every point
        # Collect reprojection_errors
        reprojection_errors = dict()
        min_depths = dict()

        for frame_idx in self.keyframes:
            pose, intrinsics = self.poses[frame_idx]
            pose_points = self.pose_point_map[frame_idx]
            world_to_cam = np.linalg.inv(pose)
            intrinsics_mat = np.identity(3)
            intrinsics_mat[0, 0] = intrinsics[0]
            intrinsics_mat[1, 1] = intrinsics[1]
            intrinsics_mat[0, 2] = intrinsics[2]
            intrinsics_mat[1, 2] = intrinsics[3]

            projection_mat = np.identity(4)[:3]

            for (point_id, point_2d) in pose_points:
                point_3d, _ =  self.points[point_id]
                point_3d_homogenous = np.append(point_3d, 1)

                point_2d_projected = intrinsics_mat @ projection_mat @ world_to_cam @ point_3d_homogenous
                depth, point_2d_projected = point_2d_projected[2], point_2d_projected[:2]/(eps + point_2d_projected[2])
                
                reprojection_error = np.linalg.norm(point_2d_projected - point_2d)

                if point_id not in reprojection_errors.keys():
                    reprojection_errors[point_id] = []
                    min_depths[point_id] = 10000000
                
                reprojection_errors[point_id].append((frame_idx, reprojection_error, depth))
                min_depths[point_id] = min(min_depths[point_id], depth)
        
        # If maximum allowed distance not set, take double median distance
        if max_allowed_distance is None:
            max_allowed_distance = 2 * np.median(np.array(list(min_depths.values())))

        # Find out which correspondences are valid
        visible_frames = dict() # point_id => [frame_indices]
        visible_points = dict() # frame_idx => [point_ids]

        for frame_idx in self.keyframes:
            visible_points[frame_idx] = [] 

        for point_id in self.points.keys():
            visible_frames[point_id] = []

            if point_id not in reprojection_errors.keys():
                continue

            if len(reprojection_errors[point_id]) < min_view_num:
                continue
            if min_depths[point_id] <= 0 or min_depths[point_id] > max_allowed_distance:
                continue
            
                
            for (frame_idx, reprojection_error, depth) in reprojection_errors[point_id]:
                if reprojection_error > reprojection_error_threshold:
                    continue
                
                visible_frames[point_id].append(frame_idx)
                visible_points[frame_idx].append(point_id)
        
        filtered_corr_num = 0
        for frame_idx in self.keyframes:
            if frame_idx not in self.pose_point_map.keys():
                continue
            
            num_corr_before = len(self.pose_point_map[frame_idx])
            filter_func = (lambda visible_points: (lambda pair: pair[0] in visible_points[frame_idx]))(visible_points)
            self.pose_point_map[frame_idx] = list(filter(filter_func, self.pose_point_map[frame_idx]))
            
            num_corr_after = len(self.pose_point_map[frame_idx])

            filtered_corr_num += num_corr_before - num_corr_after


        print(f"Filtered {filtered_corr_num} correspondences.")

        filtered_point_num = 0
        for point_id in visible_frames.keys():
            if len(visible_frames[point_id]) > 0:
                continue
            self.points.pop(point_id)
            filtered_point_num += 1
        
        print(f"Filtered {filtered_point_num} points.")



    # Saving data + visualizations
    # TODO: Move to external functions?

    def write_tum_poses(self, path):
        print("Writing tum poses ...")
        with open(path, "w") as tum_file:
            tum_file.write("# frame_index tx ty tz qx qy qz qw\n")
            for frame_index in self.keyframes:
                pose, _ = self.poses[frame_index]
                t = pose[:3, 3]
                q = R.from_matrix(pose[:3, :3]).as_quat()

                tum_file.write(f'{frame_index} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n')

        print(f'Wrote poses to {path}')

    def save_data(self, dataset, update_fps, tracking_fps, mapping_fps):
        self.exp_root.mkdir(parents=True, exist_ok=True)
        
        # Dump class data as pickles (expect image data)
        with open(self.exp_root / "keyframes.pickle", 'wb') as handle:
            pickle.dump(self.keyframes, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.exp_root / "poses.pickle", 'wb') as handle:
            pickle.dump(self.poses, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.write_tum_poses(self.exp_root / "poses_pred.txt")

        with open(self.exp_root / "points.pickle", 'wb') as handle:
            pickle.dump(self.points, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.exp_root / "pose_point_map.pickle", 'wb') as handle:
            pickle.dump(self.pose_point_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

        # Copy dataset
        self.exp_dataset_root = self.exp_root / 'dataset'
        shutil.copytree(dataset.data_root, self.exp_dataset_root)

        # Write info file
        with open(self.exp_root / "info.txt", 'w') as handle:
            handle.write(f"Dataset path {str(dataset.data_root)}\n")
            handle.write(f"Update FPS: {update_fps}\n")
            handle.write(f"Tracking FPS: {tracking_fps}\n")
            handle.write(f"Mapping FPS: {mapping_fps}\n")


        print(f"Written data to {self.exp_root}")

    def save_visualizations(self):
        self.visualization_root = self.exp_root / 'vis'
        self.visualization_root.mkdir(parents=True, exist_ok=True)

        plot_points(self.visualization_root / 'points.ply', self)
        plot_and_save_trajectory(self, save_name=str(self.visualization_root / 'trajectory.ply'))


        point_track_root = self.visualization_root / 'point_tracks'
        point_track_root.mkdir(parents=True, exist_ok=True)
        point_track_subsample_factor = 1

        visualize_point_correspondences(point_track_root, self, subsample_factor=point_track_subsample_factor)