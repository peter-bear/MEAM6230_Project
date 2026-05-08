# MEAM6230 Project

## What This Repository Does

This repository learns stable dynamical systems from demonstrations, then evaluates and deploys them.

Main pipeline:
- Load demonstrations from datasets
- Preprocess trajectories (normalization, resampling, windows)
- Train a neural dynamical system model
- Evaluate trajectory accuracy and stability
- Run inference/simulation in task space or joint space

Relevant training/evaluation code lives in:
- [data_preprocessing/data_loader.py](data_preprocessing/data_loader.py)
- [data_preprocessing/data_preprocessor.py](data_preprocessing/data_preprocessor.py)
- [agent/contrastive_imitation.py](agent/contrastive_imitation.py)
- [agent/dynamical_system.py](agent/dynamical_system.py)
- [evaluation/evaluate.py](evaluation/evaluate.py)

## Isaac Sim Folder Overview

Files in this folder:
- [isaac_sim/follow_target_panda.py](isaac_sim/follow_target_panda.py): interactive Franka control, trajectory recording, and playback.
- [isaac_sim/convert_isaac_sim_data.py](isaac_sim/convert_isaac_sim_data.py): converts recorded CSV trajectories into dataset pickle files that the training pipeline can load.

## Data Flow

1. Record joint trajectories from Isaac Sim.
2. CSV files are saved to [datasets/isaac_sim_data](datasets/isaac_sim_data).
3. Convert CSV to pickle files under:
   - [datasets/isaac_sim_learned](datasets/isaac_sim_learned)
   - example primitive folder: [datasets/isaac_sim_learned/pick_and_placed](datasets/isaac_sim_learned/pick_and_placed)
4. Train with dataset_name set to isaac_sim_learned and the corresponding primitive id(s).

## 1) Recording and Playback in Isaac Sim

Run:
- python isaac_sim/follow_target_panda.py

Controls:
- Z: toggle recording
- X: toggle playback of latest recording
- C: run an S trajectory and auto-record
- Arrow keys: move target in X/Y plane
- W/S: move target in Z
- Q/E: yaw rotation

Output:
- CSV files named joint_positions_YYYYMMDD_HHMMSS.csv in [datasets/isaac_sim_data](datasets/isaac_sim_data)
- CSV format:
  - first column: Frame
  - next columns: J0..J6 (used by converter)
  - optional additional columns: Extra0.. (ignored by converter)

## 2) Convert CSV to Training Data

Run with defaults:
- python isaac_sim/convert_isaac_sim_data.py

Default conversion behavior:
- input_dir: datasets/isaac_sim_data
- output_base_dir: datasets
- primitive_name: isaac_sim_learned
- shape_name: pick_and_placed
- delta_t: 0.01
- uses first 7 joints after Frame column

Generated file structure:
- datasets/isaac_sim_learned/pick_and_placed/*.pk

Each pickle stores:
- q: ndarray with shape (T, 7)
- delta_t: list of length T

This format is compatible with loader path:
- [data_preprocessing/data_loader.py](data_preprocessing/data_loader.py) via load_from_dict

## 3) Train on Isaac Sim Data

Use an existing params file (or create a new one) and set:
- dataset_name = isaac_sim_learned
- selected_primitives_ids matching your primitive folder index in [datasets/dataset_keys.py](datasets/dataset_keys.py)

Then train as usual with your training entrypoint:
- [train.py](train.py)



## Result

All results will be stored in the folder `rsults`

​	

## Practical Notes

- Keep trajectory recordings smooth and reasonably long for stable learning.
- Ensure the same joint ordering is used between recording and conversion.
- If your robot exports more than 7 joints, the converter still takes J0..J6 by default.
- To change joint count, call convert_isaac_sim_to_pickle with n_joints adjusted.

## Troubleshooting

No CSV found:
- Verify [datasets/isaac_sim_data](datasets/isaac_sim_data) exists and contains joint_positions_*.csv files.

Converter column mismatch error:
- The CSV has fewer columns than expected. Check Frame + joint columns and update skip_columns or n_joints.

Training cannot find dataset:
- Confirm dataset_name is isaac_sim_learned and that [datasets/dataset_keys.py](datasets/dataset_keys.py) includes it.

Unexpected trajectory behavior:
- Recheck delta_t used during conversion and consistency with simulator timing.



## Acknowledge
We gratefully acknowledge the authors of ``Stable Motion Primitives via Imitation and Contrastive Learning'' for their open-source contributions. We also thank Prof. Nadia Figueroa for providing relevant code support for the SEDS and LPVDS systems.

* **Paper:** R. Pérez-Dattari and J. Kober, "Stable Motion Primitives via Imitation and Contrastive Learning," in *IEEE Transactions on Robotics*, vol. 39, no. 5, pp. 3909-3928, Oct. 2023. [DOI: 10.1109/TRO.2023.3289597](https://doi.org/10.1109/TRO.2023.3289597)
* **Code:** [rperezdattari/Stable-Motion-Primitives-via-Imitation-and-Contrastive-Learning](https://github.com/rperezdattari/Stable-Motion-Primitives-via-Imitation-and-Contrastive-Learning)

