"""Dataset keys and helpers.

This module exposes lists of dataset primitive names and a small helper
API to look them up. Keep the original variables for backward
compatibility and also provide `DATASETS` mapping and `get_dataset_keys`.
"""

# LASA Dataset
LASA = [
        'Angle',
        'BendedLine',
        'CShape',
        'DoubleBendedLine',
        'GShape',
        'heee',
        'JShape',
        'JShape_2',
        'Khamesh',
        'Leaf_1',
        'Leaf_2',
        'Line',
        'LShape',
        'Multi_Models_1',
        'Multi_Models_2',
        'Multi_Models_3',
        'Multi_Models_4',
        'NShape',
        'PShape',
        'RShape',
        'Saeghe',
        'Sharpc',
        'Sine',
        'Snake',
        'Spoon',
        'Sshape',
        'Trapezoid',
        'Worm',
        'WShape',
        'Zshape',
]

# LAIR dataset
LAIR = [
        'e',
        'double_loop',
        'Lag',
        'double_lag',
        'phi',
        'mountain',
        'two_roads',
        'G_angle',
        'capricorn',
        'triple_loop',
        'two',
]

# Interpolation dataset
interpolation = ['interpolation_1', 'interpolation_2', 'interpolation_3']

# Optitrack dataset
optitrack = ['hammer']

# Joint space dataset
joint_space = ['cleaning_1', 'cleaning_2']

isaac_sim_learned = ['oneline', 's_shape', 'pick_and_placed']


# Mapping for programmatic access
DATASETS = {
        'LASA': LASA,
        'LAIR': LAIR,
        'interpolation': interpolation,
        'optitrack': optitrack,
        'joint_space': joint_space,
        'isaac_sim_learned': isaac_sim_learned,
}


def get_dataset_keys(name: str):
        """Return the list of primitive keys for a dataset name.

        Raises KeyError when dataset doesn't exist.
        """
        return DATASETS[name]