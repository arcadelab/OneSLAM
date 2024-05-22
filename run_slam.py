
from datasets.dataset import ImageDataset
from datasets.transforms import *

from slam import SLAMStructure
from tracking import Tracking
from modules.depth_estimation import *
from modules.pose_guessing import *
from modules.point_resampling import *
from cotracker.predictor import CoTrackerPredictor

import torch
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import argparse
import random
import time


# TODO: Store arguments and code after execution
parser = argparse.ArgumentParser(description='Python file for running the SLAM pipeline',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data loading
parser.add_argument("--data_root", required=True, type=str, help="path to dataset")
parser.add_argument("--process_subset", action='store_true', help="use start_idx/end_idx/image_subsample to define a subset of processed frames in dataset")
parser.add_argument("--start_idx", default=-1, type=int, help="starting frame index for processing")
parser.add_argument("--end_idx", default=-1, type=int, help="end frame index for processing")
parser.add_argument("--image_subsample", default=-1, type=int, help="custom image subsample factor")
parser.add_argument("--img_width", default=-1, type=int, help="width to rescale images to, -1 for no scaling")
parser.add_argument("--img_height", default=-1, type=int, help="height to rescale images to, -1 for no scaling")
parser.add_argument("--lumen_mask_low_threshold", default=0., type=float, help="mask out pixels that are too dark")
parser.add_argument("--lumen_mask_high_threshold", default=1., type=float, help="mask out pixels that are too bright")

# General Stuff
parser.add_argument("--target_device", default='cuda:0', type=str, help="GPU to run process on")
parser.add_argument("--name", default='', type=str, help="name of current experiment, leave empty to use current time")
parser.add_argument('--output_folder',  default='./experiments', type=str, help="where to store output of slam run")
parser.add_argument("--seed", default=42, type=int, help="seed for reproducability")
parser.add_argument("--verbose", action='store_true', help="print what is happening")
parser.add_argument("--no_localize", action='store_true', help="do not run localization (does not affect mapping procedure)")


# Stuff for tracking
parser.add_argument("--section_length", default=13, type=int, help="how many frames to buffer before running section point tracking")
parser.add_argument("--past_frame_size", default=5, type=int, help="how many past frames to include in point tracking")
parser.add_argument('--keyframe_decision', type=str, choices=['subsample', 'orb'], default='subsample', 
                    help='How to make keyframe decision'
                         'subsample: subsample every keyframe_subsample pose, default 4'
                         'orb: use keyframe decision process from orb')
parser.add_argument("--keyframe_subsample", default=4, type=int, help="how often to sample a keyframe")
parser.add_argument('--pose_guesser', type=str, choices=['last_pose', 'constant_velocity'], default='last_pose', 
                    help='how to obtain initial pose guess'
                         'last_pose: use last pose as initial guess'
                         'constant_velocity: use constant velocity model to obtain initial pose guess')
parser.add_argument('--depth_scale',  default=10.0, type=float, help='scaling factor for initial depth esitmates')
parser.add_argument('--point_sampler', type=str, choices=['uniform', 'sift', 'orb', 'r2d2', 'density'], default='uniform', 
                    help='how to sample new points at each section start'
                         'uniform: uniform point sampling'
                         'sift: sample sift keypoints'
                         'orb: sample orb keypoints'
                         'r2d2: sample r2d2 keypoints'
                         'density: sample new points based on existing point density')
parser.add_argument('--tracked_point_num_min',  default=200, type=int, help='Minimum number of point to track per section.')
parser.add_argument('--tracked_point_num_max',  default=2000, type=int, help='Maximum number of point to track per section.')
parser.add_argument('--localization_track_num',  default=50, type=int, help='Maximum number of points on localization.')
parser.add_argument("--update_localized_pose", action='store_true', help="update a localized pose in the SLAM datastructure")
parser.add_argument("--ransac_localization", action='store_true', help="use ransac pnp to localize pose")
parser.add_argument('--minimum_new_points',  default=0, type=int, help='Minimum number of new points to sample after every section.')
parser.add_argument('--point_resample_cooldown',  default=1, type=int, help='How many sections to wait before resampling points again')
parser.add_argument('--cotracker_model', type=str, choices=['cotracker_stride_4_wind_8', 'cotracker_stride_4_wind_12', 'cotracker_stride_8_wind_16'], default='cotracker_stride_4_wind_8', 
                    help='cotracker model to be used')
parser.add_argument('--cotracker_window_size', type=int, choices=[8, 12, 16], default=8, 
                    help='window size of the current co-tracker model. Choos appropriately.')

# Stuff for BA
parser.add_argument("--dense_ba", action='store_true', help="use dense instead of sparse bundle adjustment")
parser.add_argument("--verbose_ba", action='store_true', help="output additional information for bundle adjustment")
parser.add_argument("--tracking_ba_iterations", default=20, type=int, help="number of ba iterations after tracking")
parser.add_argument("--local_ba_size", default=10, type=int, help="maximum number of keyframes to include in local BA, -1 to use all keyframes")


# Parse arguments
args = parser.parse_args()

# Argument consistency checks
# TODO: More potential consistency checks

# If processing subset, start_idx, end_idx and image_subsample must be set.
if args.process_subset:
    assert args.start_idx != -1
    assert args.end_idx != -1
    assert args.image_subsample != -1

assert args.keyframe_subsample < args.section_length

# Set seed for reproduceability
random.seed(args.seed)     # python random generator
np.random.seed(args.seed)  # numpy random generator

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



torch.autograd.set_detect_anomaly(mode=False)
torch.autograd.profiler.profile(enabled=False)

#################################################
# Dataset/Data management and bundle adjustment #
#################################################

# Load dataset

composed_transforms = transforms.Compose([SampleToTensor(),
                                          RescaleImages((args.img_height, args.img_width)),
                                          MaskOutLuminosity(threshold_high=args.lumen_mask_high_threshold, threshold_low=args.lumen_mask_low_threshold),
                                          SampleToDevice(args.target_device)])
dataset = ImageDataset(args.data_root, transform=composed_transforms)

# Determine frames to process
frames_to_process = list(dataset.images.keys())
if args.process_subset:
    if args.start_idx > args.end_idx:
        frames_to_process = list(filter(lambda frame_idx: frame_idx in range(args.end_idx, args.start_idx, args.image_subsample), frames_to_process))
        frames_to_process = frames_to_process[::-1]
    else:
        frames_to_process = list(filter(lambda frame_idx: frame_idx in range(args.start_idx, args.end_idx, args.image_subsample), frames_to_process))



print("Number of frames to process: ", len(dataset))

# Create SLAM structure (data storage + bundle adjustment)
# TODO: Seperate data storage and bundle adjustment
slam_structure = SLAMStructure(name=args.name, output_folder=args.output_folder,
                               BA_sparse_solver=not args.dense_ba, BA_verbose=args.verbose_ba, BA_opt_iters=args.tracking_ba_iterations)

##################################################
# Load tracking module and associated components #
##################################################

# Pose guesser component
if args.pose_guesser == 'last_pose':
    pose_guesser =  PoseGuesserLastPose()
elif args.pose_guesser == 'constant_velocity':
    pose_guesser =  PoseGuesserConstantVelocity()
else:
    raise ValueError(f'Unknown argument for --pose_guesser: {args.pose_guesser}')


# Depth estimation component
depth_estimator = DepthEstimatorConstant(args.depth_scale)

# Point sampling component
if args.point_sampler == 'uniform':
    point_sampler = PointResamplerUniform(args.tracked_point_num_min, args.tracked_point_num_max, args.minimum_new_points)
elif args.point_sampler == 'sift':
    point_sampler = PointResamplerSIFT(args.tracked_point_num_min, args.tracked_point_num_max, args.minimum_new_points)    
elif args.point_sampler == 'orb':
    point_sampler = PointResamplerORB(args.tracked_point_num_min, args.tracked_point_num_max, args.minimum_new_points)    
elif args.point_sampler == 'r2d2':
    point_sampler = PointResamplerR2D2(args.tracked_point_num_min, args.tracked_point_num_max, args.minimum_new_points)    
elif args.point_sampler == 'density':
    point_sampler = PointResamplerUniformDensity(args.tracked_point_num_min, args.tracked_point_num_max, args.minimum_new_points)
else:
    raise ValueError(f'Unknown argument for --point_sampler: {args.point_sampler}')


# Point tracking component
#cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker_w8").to(device=args.target_device)
cotracker = CoTrackerPredictor(checkpoint="./trained_models/cotracker/"+args.cotracker_model+".pth").to(device=args.target_device)
cotracker.eval()

# Create tracking module
tracking_module = Tracking(depth_estimator, point_sampler, cotracker, args.target_device, cotracker_window_size = args.cotracker_window_size)

#############################
# Running the SLAM pipeline #
#############################

# Current section and keyframes 
current_section = []
current_keyframes = []

# History of old sections
completed_sections = []


# Cooldowns
orb_keyframe_cooldown = 4
keyframe_cooldown = 0
refframe_cooldown = 0
point_resample_cooldown = 0

# For calculating tracking FPS (how long does frame localization take on average)
total_tracking_time = 0
tracking_counter = 0

# For calculating mapping FPS (how long does mapping take on average)
total_mapping_time = 0
mapping_counter = 0

completed_sections = 0

# For calculating effective update FPS (how many frame updates does the pipeline provide per second)
update_counter = 0
total_update_time = 0

# Main SLAM loop
for frame_idx in tqdm(frames_to_process):
    # Add frame to slam structure
    last_poses = slam_structure.get_previous_poses(10)

    # Special case for first iteration
    if len(slam_structure.poses.keys()) == 0:
        # Retrieve data to add frame and make into keyframe
        image = dataset[frame_idx]['image'].detach().cpu().numpy()
        depth = depth_estimator(dataset[frame_idx]['image'], dataset[frame_idx]['mask']).squeeze().detach().cpu().numpy()
        intrinsics = dataset[frame_idx]['intrinsics'].detach().cpu().numpy()
        mask = dataset[frame_idx]['mask'].squeeze().detach().cpu().numpy()
        mask[depth < 1e-6] = 0

        # Get new estimate for poses
        pose = pose_guesser(last_poses)

        # Add first frame and make into keyframe
        slam_structure.add_frame(frame_idx, pose, intrinsics)
        slam_structure.make_keyframe(frame_idx, image, depth, mask, fixed=True)
        keyframe_cooldown = args.keyframe_subsample
    else:
        # Retrieve data to add new frame
        intrinsics = dataset[frame_idx]['intrinsics'].detach().cpu().numpy()
        pose = pose_guesser(last_poses)

        # Add new frame
        slam_structure.add_frame(frame_idx, pose, intrinsics)


    # If localization is enabled and map is initiallized, localize frame
    if completed_sections > 1:
        tracking_start_time = time.time()

        localized_frame = slam_structure.poses[frame_idx][0]

        # Skip localization if disabled, good for test runs
        if not args.no_localize:
            # Run a minimal point tracking to obtain localization correspondences
            section_to_track = list(slam_structure.poses.keys())[-args.past_frame_size:]
            start_frame = max(0, args.past_frame_size - len(current_section) - 1)
            if args.verbose:
                print("Current section: ", current_section)
                print("Tracking start index: ", section_to_track[start_frame])

            tracking_module.process_section(section_to_track, dataset, slam_structure, 
                                        sample_new_points=False, 
                                        start_frame=start_frame,
                                        maximum_track_num=args.localization_track_num)
            
                

            # Localize frame
            localized_frame = slam_structure.localize_frame(frame_idx, 
                                                            update_pose=args.update_localized_pose,
                                                            ransac=args.ransac_localization)

   
        # Tracking done and new update available
        current_time = time.time()

        update_counter += 1
        total_update_time += current_time - update_start_time

        tracking_counter += 1
        total_tracking_time += current_time - tracking_start_time

        # Print update
        if args.verbose:
            print("Current frame: ", localized_frame[:3, 3])
            print("Running section: ", current_section)
            print("Running FPS: ", update_counter/total_update_time)
            if tracking_counter > 0:  print("Running Tracking FPS: ", tracking_counter/total_tracking_time)
            if mapping_counter > 0: print("Running Mapping FPS: ", mapping_counter/total_mapping_time)


        update_start_time = time.time()


    # Add frame to current section
    current_section.append(frame_idx)

    # Check if frame buffer is full.
    if len(current_section) < args.section_length:
        continue
    
    # New mapping counter starts
    mapping_start_time = time.time()

    # Section full, start MAPing

    # Remove all existing correspondences (likely to be faulty),
    # except for frist frame
    for idx in current_section[1:]:
        assert idx not in slam_structure.keyframes
        slam_structure.pose_point_map[idx] = []

    # Update point resample cooldown
    point_resample_cooldown -= 1

    # Obtain new consistent set of point correspondences
    section_to_track = np.copy(current_section)
    tracking_module.process_section(section_to_track, dataset, slam_structure, 
                                    sample_new_points=(point_resample_cooldown<=0), 
                                    start_frame=0,
                                    maximum_track_num=args.tracked_point_num_max)

    # Update point resample cooldown
    if point_resample_cooldown <=0:
        point_resample_cooldown = args.point_resample_cooldown

    # Decide to make frames into new keyframes
    new_keyframe_counter = 0
    for idx in current_section[1:]:
        # Keyframe decision

        make_keyframe = False
        
        if args.keyframe_decision == "subsample":
            keyframe_cooldown -= 1
            if keyframe_cooldown <= 0:
                keyframe_cooldown = args.keyframe_subsample
                make_keyframe = True

        if args.keyframe_decision == "orb":
            keyframe_cooldown -= 1
            if keyframe_cooldown <= 0:
                # Check if last keyframe was old
                make_keyframe = True
            else:
                last_keyframe = slam_structure.keyframes[-1]
                last_pose_points =  slam_structure.pose_point_map[last_keyframe]
                last_point_ids = set()
                for (point_id, point_2d) in last_pose_points: last_point_ids.add(point_id)

                current_pose_points = slam_structure.pose_point_map[idx]

                tracked_point_ids = set()
                for (point_id, point_2d) in current_pose_points:
                    if point_id in last_point_ids: tracked_point_ids.add(point_id)
                
                if len(tracked_point_ids)/len(last_point_ids) < 0.8:
                    make_keyframe = True

            if make_keyframe:
                keyframe_cooldown = orb_keyframe_cooldown

        if not make_keyframe:
            continue

        # Frame was choosen to be a keyframe

        image = dataset[idx]['image'].detach().cpu().numpy()
        depth = depth_estimator(dataset[idx]['image'], dataset[idx]['mask']).squeeze().detach().cpu().numpy()
        mask = dataset[idx]['mask'].squeeze().detach().cpu().numpy()
        mask[depth < 1e-6] = 0

        slam_structure.make_keyframe(idx, image, depth, mask, fixed=False)
        new_keyframe_counter += 1
       
        # TODO: Add loop closure here

    # If there are new keyframes, run local BA
    if new_keyframe_counter > 0:
        for idx in slam_structure.keyframes[:-(args.local_ba_size+new_keyframe_counter)]:
            slam_structure.BA.fix_pose(idx, fixed=True)
        for idx in slam_structure.keyframes[-(args.local_ba_size+new_keyframe_counter):]:
            slam_structure.BA.fix_pose(idx, fixed=False)
        slam_structure.BA.fix_pose(slam_structure.keyframes[0], fixed=True)
            
        slam_structure.run_ba(opt_iters=args.tracking_ba_iterations)

    # Mapping done
    current_time = time.time()
    mapping_counter += 1
    total_mapping_time += current_time - mapping_start_time

    # Update section
    current_section = current_section[-1:]
    completed_sections += 1

    if completed_sections == 2:
        # This was the map initialization, start update counter
        update_start_time = time.time()
        if args.verbose:
            print("Map initizalized.")


# Filter outliers in reconstruction
# NOTE: ONLY FILTER AFTER EVERYTHING IS OPTIMIZED, DOES NOT UPDATE 
# NOTE: Possibly filters the entire sparse reconstruction, resulting in an error when trying to save data.
#       Disable to allow evaluating the pose predictions in these cases.
# slam_structure.filter(min_view_num=2, reprojection_error_threshold=10)

slam_structure.save_visualizations()

# Save data and visualizations
slam_structure.save_data(dataset, 
                         update_fps=update_counter/total_update_time,
                         tracking_fps=tracking_counter/total_tracking_time,
                         mapping_fps=mapping_counter/total_mapping_time)

print("Done.")