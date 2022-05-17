# Combined Understanding of 3D Plane Articulation and Partial Human Pose Estimation

Based on paper **Understanding 3D Object Articulation in Internet Videos** and paper **Full-Body Awareness from Partial Observations**.
Please feel free to point out any mistakes in this repo!

## Setup

Set up the same environment as [Articulation](https://jasonqsy.github.io/Articulation3D/).

Set up additional environment for [SMPL](https://smpl.is.tue.mpg.de/) and [GraphCMR](https://github.com/crockwell/partial_humans/blob/master/GraphCMR/README.md). Make packeges compilable in python 3.7 environment.

## Demo

To run the model on a video, run

```bash
python combined_demo.py --config config/config.yaml --input /z/syqian/articulation_data/step2_filtered_clips/CxTFIEpSgew_34_360.mp4 --output demo_3336_output --save-obj --webvis
```

or

```bash
python combined_demo.py --config config/config.yaml --save-obj --webvis
```

To save the 3d model, add `--save-obj` and `--webvis` flags.

Per step demo can be done by running
```bash
python generate_mesh_mask.py --config config/config.yaml --input /z/syqian/articulation_data/step2_filtered_clips/CxTFIEpSgew_34_360.mp4 --output /data/siyich/cmr_art/mask_mesh_3336_output --webvis
```

then

```bash
python generate_mesh_mask.py --config config/config.yaml --input /z/syqian/articulation_data/step2_filtered_clips/CxTFIEpSgew_34_360.mp4 --output /data/siyich/cmr_art/mask_mesh_3336_output --webvis
```
```bash
python opt_single_img.py --input /data/siyich/cmr_art/mask_mesh_3336_output --frame 60 --output /data/siyich/cmr_art/opt_3336_output
```

or

```bash
python generate_mesh_mask.py --config config/config.yaml --input /z/syqian/articulation_data/step2_filtered_clips/CxTFIEpSgew_34_360.mp4 --output /data/siyich/cmr_art/mask_mesh_3336_output --webvis
```
```bash
python opt_pose_img.py --input /data/siyich/cmr_art/mask_mesh_3336_output --frame 60 --output /data/siyich/cmr_art/opt_3336_output
```


## Acknowledgment

We reuse the codebase of [SparsePlanes](https://github.com/jinlinyi/SparsePlanes) and [Mesh R-CNN](https://github.com/facebookresearch/meshrcnn).
