"""Convert Isaac Sim CSV recordings into repository-compatible pickle trajectories.

Each output pickle stores:
- ``q``: array with shape ``(T, n_joints)``
- ``delta_t``: list with one timestep value per sample

The generated files are compatible with ``data_preprocessing.data_loader.load_from_dict``.
"""

import pickle
import numpy as np
from pathlib import Path


def convert_isaac_sim_to_pickle(
    input_dir='datasets/isaac_sim_data',
    output_base_dir='datasets',
    primitive_name='isaac_sim_learned',
    shape_name='pick_and_placed',
    delta_t=0.01,
    skip_columns=1,
    n_joints=7,
):
    """
    Convert Isaac Sim CSV files to pickle format.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing CSV files.
    output_base_dir : str
        Base datasets directory.
    primitive_name : str
        Dataset folder name under the base directory.
    shape_name : str
        Primitive folder name under the dataset folder.
    delta_t : float
        Time step between samples (seconds).
    skip_columns : int
        Number of leading columns to skip (default skips Frame column).
    n_joints : int
        Number of joint columns to keep after skipped columns.
    """

    repo_root = Path(__file__).resolve().parents[1]
    input_path = Path(input_dir)
    output_base_path = Path(output_base_dir)

    if not input_path.is_absolute():
        input_path = repo_root / input_path
    if not output_base_path.is_absolute():
        output_base_path = repo_root / output_base_path

    output_dir = output_base_path / primitive_name / shape_name
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all CSV files.
    csv_files = sorted(input_path.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to convert")
    print(f"Output directory: {output_dir}")
    print()
    
    for idx, csv_file in enumerate(csv_files, start=1):
        print(f"Processing {idx}/{len(csv_files)}: {csv_file.name}")

        # Read CSV as numeric data and keep requested joint columns.
        raw_data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)

        last_column = skip_columns + n_joints
        if raw_data.shape[1] < last_column:
            raise ValueError(
                f'File {csv_file} has {raw_data.shape[1]} columns; expected at least {last_column}.'
            )

        joint_data = raw_data[:, skip_columns:last_column].astype(np.float32, copy=False)
        
        print(f"  - Shape: {joint_data.shape} (time steps, joints)")
        print(f"  - Joint ranges: min={joint_data.min():.4f}, max={joint_data.max():.4f}")
        
        # Store one timestep value per sample to match downstream expectations.
        delta_t_list = [float(delta_t)] * joint_data.shape[0]

        data = {
            'q': joint_data,
            'delta_t': delta_t_list,
        }

        # Generate output filename.
        output_filename = output_dir / f"{csv_file.stem}.pk"

        with output_filename.open('wb') as file_obj:
            pickle.dump(data, file_obj)
        
        print(f"  ✓ Saved to {output_filename}")
        print()

if __name__ == '__main__':
    convert_isaac_sim_to_pickle(
        input_dir='datasets/isaac_sim_data',
        output_base_dir='datasets',
        primitive_name='isaac_sim_learned',
        shape_name='pick_and_placed',
        delta_t=0.01,
    )
