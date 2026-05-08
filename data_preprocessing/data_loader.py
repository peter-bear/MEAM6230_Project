from pathlib import Path
from typing import List, Tuple, Union, Callable
import pickle
import numpy as np
import scipy.io as sio
import logging

from datasets.dataset_keys import LASA, LAIR, optitrack, interpolation, joint_space, isaac_sim_learned

logger = logging.getLogger(__name__)


def load_demonstrations(dataset_name: str, selected_primitives_ids: Union[str, List[int]]):
    """Load demonstrations for a given dataset and selected primitive IDs.

    Returns a dictionary with keys:
      - 'demonstrations raw': list of numpy arrays (dim x T)
      - 'demonstrations primitive id': list of ints
      - 'n primitives': int
      - 'delta t eval': list or scalar
    """
    dataset_primitives_names = get_dataset_primitives_names(dataset_name)
    primitives_names, primitives_save_name = select_primitives(dataset_primitives_names, selected_primitives_ids)
    n_primitives = len(primitives_names)

    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = repo_root / 'datasets' / dataset_name

    loader = get_data_loader(dataset_name)
    demonstrations, demonstrations_primitive_id, delta_t_eval = loader(dataset_path, primitives_names)

    return {
        'demonstrations raw': demonstrations,
        'demonstrations primitive id': demonstrations_primitive_id,
        'n primitives': n_primitives,
        'delta t eval': delta_t_eval,
    }


def get_dataset_primitives_names(dataset_name: str) -> List[str]:
    """Return the list of primitive names for a dataset."""
    mapping = {
        'LASA': LASA,
        'LAIR': LAIR,
        'optitrack': optitrack,
        'interpolation': interpolation,
        'joint_space': joint_space,
        'isaac_sim_learned': isaac_sim_learned,
    }
    try:
        return mapping[dataset_name]
    except KeyError:
        raise NameError(f'Dataset {dataset_name} does not exist')


def select_primitives(dataset: List[str], selected_primitives_ids: Union[str, List[int]]) -> Tuple[List[str], str]:
    """Select primitives by index list or comma-separated string.

    Returns (selected_names, save_name).
    """
    if isinstance(selected_primitives_ids, str):
        ids = [int(x.strip()) for x in selected_primitives_ids.split(',') if x.strip()]
    else:
        ids = list(map(int, selected_primitives_ids))

    selected_names = [dataset[i] for i in ids]
    save_name = '_'.join(str(i) for i in ids)
    return selected_names, save_name


def get_data_loader(dataset_name: str) -> Callable:
    """Return the loader function for a dataset."""
    if dataset_name == 'LASA':
        return load_LASA
    if dataset_name in ('LAIR', 'optitrack', 'interpolation'):
        return load_numpy_file
    if dataset_name in ('joint_space', 'isaac_sim_learned'):
        return load_from_dict
    raise NameError(f'Dataset {dataset_name} does not exist')


def load_LASA(dataset_dir: Path, demonstrations_names: List[str]):
    """Load LASA .mat demonstration files.

    Each returned demo is a numpy array with shape (2, T).
    """
    demos: List[np.ndarray] = []
    primitive_id: List[int] = []
    dt_list: List[float] = []

    for i, name in enumerate(demonstrations_names):
        mat_path = Path(dataset_dir) / name
        mat = sio.loadmat(str(mat_path))
        data = mat['demos']

        for j in range(data.shape[1]):
            s_x = data[0, j]['pos'][0, 0][0]
            s_y = data[0, j]['pos'][0, 0][1]
            demo = np.vstack((s_x, s_y))
            demos.append(demo)
            dt_list.append(float(data[0, j]['dt'][0, 0][0, 0]))
            primitive_id.append(i)

    return demos, primitive_id, dt_list


def load_numpy_file(dataset_dir: Path, demonstrations_names: List[str]):
    """Load demonstrations stored as numpy arrays in subfolders.

    Returns demos as a list of arrays with shape (dim, T).
    """
    demos: List[np.ndarray] = []
    primitive_id: List[int] = []

    for i, sub in enumerate(demonstrations_names):
        folder = Path(dataset_dir) / sub
        if not folder.exists():
            logger.warning('Folder %s does not exist, skipping', folder)
            continue
        for f in sorted(folder.iterdir()):
            if not f.is_file() or f.suffix.lower() not in ('.npy', '.npz'):
                continue
            arr = np.load(str(f))
            # normalize to shape (dim, T)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            if arr.shape[0] > arr.shape[1]:
                arr = arr.T
            demos.append(arr)
            primitive_id.append(i)

    return demos, primitive_id, 1


def load_from_dict(dataset_dir: Path, demonstrations_names: List[str]):
    """Load demonstrations stored as pickled dicts containing keys 'q' and 'delta_t'."""
    demos: List[np.ndarray] = []
    primitive_id: List[int] = []
    dt_list: List[float] = []

    for i, sub in enumerate(demonstrations_names):
        folder = Path(dataset_dir) / sub
        if not folder.exists():
            logger.warning('Folder %s does not exist, skipping', folder)
            continue
        for f in sorted(folder.iterdir()):
            if not f.is_file():
                continue
            with open(f, 'rb') as fh:
                data = pickle.load(fh)
            q = np.asarray(data['q'])
            if q.ndim == 1:
                q = q[np.newaxis, :]
            demos.append(q.T)
            dt_list.append(data.get('delta_t', 1))
            primitive_id.append(i)

    return demos, primitive_id, dt_list
