import torch
import numpy as np
import time

# TODO: Rename this to point tracking for more clarity.

# Helper functions (TODO: Relocate to misc.?)
def apply_transformation(T, point_cloud):
    try:
        for i in range(len(point_cloud)):
                point_cloud[i] = T @ point_cloud[i]
            
        return point_cloud
    except:
        breakpoint()

def unproject(points_2d, depth_image, cam_to_world_pose, intrinsics):
    # Unproject n 2d points
    # points: n x 2
    # depth_image: H x W
    # cam_pose: 4 x 4 

    x_d, y_d = points_2d[:, 0], points_2d[:, 1]
    fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]

    depths = depth_image[y_d.astype(int), x_d.astype(int)]
    x = ((x_d - cx) * depths / fx)[:, None]
    y = ((y_d - cy) * depths / fy)[:, None]
    z = depths[:, None]

    points_3d = np.stack([x, y, z, np.ones_like(x)],  axis=-1).squeeze(axis=1)

    points_3d = apply_transformation(cam_to_world_pose, points_3d)

    return points_3d



class Tracking:
    def __init__(self, depth_estimator, point_resampler, tracking_network, target_device, cotracker_window_size) -> None:
        self.point_resampler = point_resampler
        self.depth_estimator = depth_estimator
        self.tracking_network = tracking_network
        self.target_device = target_device
        self.cotracker_window_size = cotracker_window_size

    def process_section(self, section_indices, dataset, slam_structure, 
                        sample_new_points=True, 
                        start_frame=0, 
                        maximum_track_num = None):        
        # NOTE: This now only does the point tracking and does not add new keyframes or aquires new initial pose estimates 
        #       requires first frame to be present in poses
        assert section_indices[start_frame] in slam_structure.poses.keys()

        # Pad section to atleast five frames
        section_valid = [True for idx in section_indices]
        while len(section_indices) < (self.cotracker_window_size//2) + 1:
            section_indices.append(section_indices[-1])
            section_valid.append(False)


        # Retrieve section data
        samples = []
        for frame_idx in section_indices:
            samples.append(dataset[frame_idx])


        new_point_ids = []

        # Point resampling process
        if sample_new_points:
            image_0 = samples[start_frame]['image'].detach().cpu().numpy()
            depth_0 = self.depth_estimator(samples[start_frame]['image'], samples[-1]['mask']).squeeze().detach().cpu().numpy()
            intrinsics_0 = samples[start_frame]['intrinsics'].detach().cpu().numpy()
            mask_0 = samples[start_frame]['mask'].squeeze().detach().cpu().numpy()
            mask_0[depth_0 < 1e-6] = 0

            pose_0, _ = slam_structure.poses[section_indices[start_frame]]

            # Get current points
            current_pose_points = slam_structure.get_pose_points(section_indices[start_frame])

            # Resample points
            kept_pose_points, new_points_2d = self.point_resampler(current_pose_points, image_0, depth_0, intrinsics_0, mask_0, slam_structure)       

            # Unproject new 2d samples
            new_points_3d = unproject(new_points_2d, depth_0, pose_0, intrinsics_0)

            # Add new points and correspondences to datastructure 
            for i in range(len(new_points_3d)):
                point_3d = new_points_3d[i, :3]
                point_2d = new_points_2d[i]

                point_color = image_0[:, int(point_2d[1]), int(point_2d[0])]

                point_id = slam_structure.add_point(point_3d, point_color)
                new_point_ids.append(point_id)
                slam_structure.add_correspondence(section_indices[start_frame], point_id, point_2d)

                kept_pose_points.append((point_id, point_2d))

        
        # Obtain currently tracked points on first frame
        pose_points = slam_structure.get_pose_points(section_indices[start_frame])

        if maximum_track_num is not None:
            pose_points = pose_points[:maximum_track_num]

        
        local_point_ids = []
        queries = []
        
        # If this hits, you ran out of tracked points
        if len(pose_points) <= 0:
            if sample_new_points:
                print("Sampling was allowed")
            else:
                print("Sampling was not allowed")
            assert False

        # Generate input data for Co-Tracker
        for (point_id, point2d) in pose_points:
            local_point_ids.append(point_id)
            if point2d[0]< 0 or  point2d[1] < 0:
                breakpoint()
            queries.append([start_frame, point2d[0], point2d[1]])
        
        image_seq = torch.cat([sample['image'][None, ...] for sample in samples])[None, ...]
        queries = torch.FloatTensor(queries)[None, ...].to(device=self.target_device)


        
        # Run Co-Tracker
        # Run tracking network
        # current_time = time.time()
        mask = samples[start_frame]['mask'][None]
        #breakpoint()
        with torch.no_grad():
            pred_tracks, pred_visibility = self.tracking_network(image_seq, 
                                                                 queries=queries,
                                                                 segm_mask=mask)
            #print("Cotracker runtime: ", time.time()- current_time)

        # Add new correspondences
        for local_idx in range(start_frame+1, len(section_indices)):
            # Check if frame was duplicated
            if not section_valid[local_idx]:
                continue
            
            # Get frame_idx and mask
            frame_idx = section_indices[local_idx]
            mask = samples[local_idx]['mask'].squeeze().detach().cpu().numpy()
            H, W = mask.shape


            # Add new correspondences
            for i in range(len(local_point_ids)):
                # Retrieve point data
                point_id = local_point_ids[i]
                
                #if point_id not in new_point_ids and frame_idx > 10:
                #    continue

                tracked_point = pred_tracks[0, local_idx, i].detach().cpu().numpy()

                # Point outside of image boundary goes out of focus
                if tracked_point[0] < 0 or tracked_point[1] < 0 or tracked_point[0] >= W or tracked_point[1] >= H:
                    pred_visibility[0, local_idx, i] = False
                    continue
                
                # Point outside of mask goes out of focus
                if mask[int(tracked_point[1]),  int(tracked_point[0])] == 0:
                    pred_visibility[0, local_idx, i] = False
                    continue
                
                # Check if point has never gone out of focus
                visible = pred_visibility[0, :local_idx+1, i]
                if not torch.all(visible):
                    continue
                
                # Add actual point
                slam_structure.add_correspondence(frame_idx, point_id, tracked_point)
        
        #breakpoint()

        # Point tracking done