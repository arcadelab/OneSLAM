# *OneSLAM* to Map Them All: A Generalized Approach to SLAM for Monocular Endoscopic Imaging based on Tracking any Point

This repository contains the code for the monocular endoscopy SLAM pipeline *OneSLAM* to be presented at IPCAI 2024. The associated paper can be found [here](https://en.wikipedia.org/wiki/Todo).  If you have any questions, please contact [Timo Teufel](https://www.linkedin.com/in/timo-teufel-b00365247/) (tteufel1@jh.edu), [Hongchao Shu](https://en.wikipedia.org/wiki/Personal_web_page) (hshu4@jh.edu) or [Mathias Unberath](https://engineering.jhu.edu/faculty/mathias-unberath/) (unberath@jhu.edu). 

## Installing

Before setting up the pipeline, please make sure cmake and conda is installed and available on your machine.
 
After downloading the repository and navigating to the root folder

```
git clone TODO
cd TODO
```

run 

```
./init_submodules.sh
./get_tap_model.sh
```

to initialize the necessary submodules and download the CoTracker model weights used by the pipeline.

Finally, run

```
source ./install.sh
```

to install the `OneSLAM` conda environment and g2opy. Please make sure to use the `source` command to ensure a proper installation of the latter into the conda environment.

The installation process was tested on Ubuntu 20.04.6 LTS.

## Minimal example

To run a minimal example on sinus anatomy, first download our example data using 

```
wget "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/tteufel1_jh_edu/EYSTSgAF2kVBgWzcMANPVTUBMSN5vzUsMHUqZ2gA4-vUuw?e=wat8oQ&download=1" -O SINUS_P07_RIGHT_EXAMPLE.zip
unzip SINUS_P07_RIGHT_EXAMPLE.zip
```

Afterwards, With the `OneSLAM` conda environment active, run

```
python run_slam.py --data_root ./SINUS_P07_RIGHT_EXAMPLE
```
 The example takes around 7 minutes to process all frames and generate the visualization on a Quadro RTX 6000. Requires around 5GB of VRAM to run.

For some quantative metrics, navigate into the corresponding experiment folder and run
```
mkdir results
evo_ape tum  ./dataset/poses_gt.txt ./poses_pred.txt -as --plot --save_plot ./plot.pdf 
evo_ape tum  ./dataset/poses_gt.txt ./poses_pred.txt -as > ./results/APE_trans.txt
evo_ape tum  ./dataset/poses_gt.txt ./poses_pred.txt -as --pose_relation angle_deg > ./results/APE_rot.txt
evo_rpe tum  ./dataset/poses_gt.txt ./poses_pred.txt -as --delta 1 --delta_unit m --all_pairs > ./results/RPE_trans.txt
evo_rpe tum  ./dataset/poses_gt.txt ./poses_pred.txt -as --pose_relation angle_deg  --delta 1 --delta_unit m --all_pairs > ./results/RPE_rot.txt
```
This will give you the RMSE in absolute (APE) and relative (RPE) (per mm) errors in translation (Trans) and rotation (Rot). 

## Running on your own data

To run on your data, setup a directory as follows

```
└── data
    ├── images
    │   ├── 00000000.jpg
    │   ├── 00000001.jpg
    │   ├── 00000002.jpg
    │   └── ...
    ├── calibration.json
    ├── [mask.bmp]
    └── [poses_gt.txt]
```

For the structure of the `calibration.json` file, please view the example provided in `example_sinus_data`. Here, `intrinsics` refer to the intrinsic camera parameters, FPS refers to the frames-per-second of the source video (note that latter is currently unused). We assume a pinhole camera model without distortion. 

The poses are required to be in [TUM format](https://github.com/MichaelGrupp/evo/wiki/Formats#tum---tum-rgb-d-dataset-trajectory-format) with the frame index being used as the timestamp. The poses are assumed to be the Camera-To-World transformations.

To start the slam pipeline run

```
python run_slam.py --data_root ./path/to/your/folder
```

After the run is finished, the output+visualizations will be saved in the folder `experiments/TIMESTAMP`.

## Acknowledgments

We want to thank [TAP-Vid](https://tapvid.github.io/), [CoTracker](https://co-tracker.github.io/), [R2D2](https://github.com/naver/r2d2),  [g2opy](https://github.com/uoip/g2opy), [EndoSLAM](https://github.com/CapsuleEndoscope/EndoSLAM/tree/master), [SAGE-SLAM](https://github.com/lppllppl920/SAGE-SLAM) and [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) for publicly releasing their code.

## Citation

If you can make use of this work in your own research, please consider citing it as

```
TODO
```
