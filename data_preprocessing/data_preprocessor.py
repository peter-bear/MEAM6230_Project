import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.interpolate import splprep, splev

from agent.utils.dynamical_system_operations import normalize_state
from data_preprocessing.data_loader import load_demonstrations

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, params: Any, verbose: bool = True) -> None:
        """Load preprocessing parameters from `params` object.

        `params` is expected to expose attributes used below (typical project params object).
        """
        self.trajectories_resample_length: int = params.trajectories_resample_length
        self.state_increment: float = params.state_increment
        self.dim_workspace: int = params.workspace_dimensions
        self.dynamical_system_order: int = params.dynamical_system_order
        self.dim_state: int = self.dim_workspace * self.dynamical_system_order
        self.workspace_boundaries_type: str = params.workspace_boundaries_type
        self.workspace_boundaries = np.array(params.workspace_boundaries)
        self.eval_length: int = params.evaluation_samples_length
        self.dataset_name: str = params.dataset_name
        self.selected_primitives_id = params.selected_primitives_ids
        self.spline_sample_type: str = params.spline_sample_type

        self.delta_t: float = 1.0
        self.imitation_window_size: int = params.imitation_window_size
        self.verbose = verbose

    def run(self):
        """
        Computes relevant features from the raw demonstrations
        """
        loaded_data = load_demonstrations(self.dataset_name, self.selected_primitives_id)

        features_demos = self.get_features_demos(loaded_data)
        demonstrations_train = self.generate_training_data(loaded_data, features_demos)
        limits_derivatives = self.get_limits_derivatives(demonstrations_train)

        preprocess_output: Dict[str, Any] = {'demonstrations train': demonstrations_train}
        preprocess_output.update(loaded_data)
        preprocess_output.update(features_demos)
        preprocess_output.update(limits_derivatives)
        return preprocess_output

    def get_features_demos(self, loaded_data):
        """
        Computes useful features from demonstrations
        """
        demonstrations_raw = loaded_data['demonstrations raw']
        primitive_ids = loaded_data['demonstrations primitive id']

        x_min, x_max = self.get_workspace_boundaries(demonstrations_raw)
        goals = self.get_goals(demonstrations_raw, primitive_ids)
        goals_training = normalize_state(goals, x_min, x_max)

        n_trajectories = len(demonstrations_raw)
        max_trajectory_length, trajectories_length, eval_indexes = self.get_trajectories_length(
            demonstrations_raw, n_trajectories
        )

        features_demos = {
            'x min': x_min,
            'x max': x_max,
            'goals': goals,
            'goals training': goals_training,
            'max demonstration length': max_trajectory_length,
            'demonstrations length': trajectories_length,
            'eval indexes': eval_indexes,
            'n demonstrations': n_trajectories,
        }
        return features_demos

    def get_workspace_boundaries(self, demonstrations_raw):
        """
        Computes workspace boundaries
        """
        if self.workspace_boundaries_type == 'from data':
            if not demonstrations_raw or len(demonstrations_raw) == 0:
                raise ValueError(
                    f"No demonstrations found to compute workspace boundaries. "
                    f"Dataset: {self.dataset_name}, selected primitives: {self.selected_primitives_id}. "
                    "Check dataset folder and params."
                )

            stacked_max = np.stack([np.asarray(d).max(axis=1) for d in demonstrations_raw], axis=0)
            stacked_min = np.stack([np.asarray(d).min(axis=1) for d in demonstrations_raw], axis=0)
            x_max_orig = stacked_max.max(axis=0)
            x_min_orig = stacked_min.min(axis=0)

            x_max = x_max_orig + (x_max_orig - x_min_orig) * self.state_increment / 2.0
            x_min = x_min_orig - (x_max_orig - x_min_orig) * self.state_increment / 2.0
        elif self.workspace_boundaries_type == 'custom':
            x_max = self.workspace_boundaries[:, 1]
            x_min = self.workspace_boundaries[:, 0]
        else:
            raise NameError('Selected workspace boundaries type not valid. Try: from data, custom')
        return x_min, x_max

    def get_trajectories_length(self, demonstrations_raw, n_trajectories):
        """
        Computes length trajectories, longest trajectory and evaluation indexes for fast evaluation
        """
        trajectories_length: List[int] = []
        eval_indexes: List[np.ndarray] = []
        max_trajectory_length = 0

        for demo in demonstrations_raw:
            length_demo = int(len(demo[0]))
            trajectories_length.append(length_demo)
            if length_demo > max_trajectory_length:
                max_trajectory_length = length_demo

            if length_demo > self.eval_length:
                eval_interval = int(np.floor(length_demo / self.eval_length))
                eval_indexes.append(np.arange(0, length_demo, eval_interval, dtype=np.int32))
            else:
                eval_indexes.append(np.arange(0, length_demo, 1, dtype=np.int32))

        return max_trajectory_length, trajectories_length, eval_indexes

    def get_goals(self, demonstrations_raw, primitive_ids):
        """
        Computes goal demonstrations from data
        """
        goals: List[np.ndarray] = []
        for i in np.unique(primitive_ids):
            ids_primitives = (np.array(primitive_ids) == i)
            indices = np.where(ids_primitives)[0]
            goals_primitive = [np.asarray(demonstrations_raw[idx])[:, -1] for idx in indices]
            goals.append(np.mean(np.stack(goals_primitive, axis=0), axis=0))
        return np.array(goals)

    def generate_training_data(self, loaded_data, features_demos):
        """
        Normalizes demonstrations, resamples demonstrations using spline to keep a constant distance between points,
        and creates imitation window for backpropagation through time
        """
        demonstrations_raw = loaded_data['demonstrations raw']
        n_trajectories = len(demonstrations_raw)
        resampled_positions: List[np.ndarray] = []
        error_acc: List[float] = []

        # Pad demonstrations to same length for spline fitting
        padded = []
        for i in range(features_demos['n demonstrations']):
            padding_length = features_demos['max demonstration length'] - features_demos['demonstrations length'][i]
            padded.append(np.pad(demonstrations_raw[i], ((0, 0), (0, padding_length)), mode='edge'))
        demonstrations_raw_padded = np.array(padded)

        for j in range(n_trajectories):
            if self.verbose:
                logger.info('Data preprocessing, demonstration %d / %d', j + 1, n_trajectories)

            demo = np.asarray(demonstrations_raw_padded[j]).T
            length_demo = demo.shape[0]
            demo_norm = normalize_state(demo, x_min=features_demos['x min'], x_max=features_demos['x max'])

            # compute phases
            if self.spline_sample_type == 'evenly spaced':
                curve_phases = [0.0]
                delta_phases = []
                for i in range(length_demo - 1):
                    delta_phase = float(np.linalg.norm(demo_norm[i + 1, :] - demo_norm[i, :]))
                    if delta_phase == 0.0:
                        delta_phase = 1e-15
                    curve_phases.append(curve_phases[-1] + delta_phase)
                    delta_phases.append(delta_phase)
                delta_phases.append(0.0)
                curve_phases = np.array(curve_phases)
                delta_phases = np.array(delta_phases)
                max_phase = float(curve_phases[-1])
            elif self.spline_sample_type in ('from data', 'from data resample'):
                curve_phases = np.arange(0, length_demo * self.delta_t, self.delta_t)
                delta_phases = np.ones(length_demo) * self.delta_t
                max_phase = float(np.max(curve_phases))
            else:
                raise NameError('Spline sample type not valid, check params file for options.')

            spline_input = [demo_norm[:, i] for i in range(self.dim_workspace)] + [curve_phases, delta_phases]
            spline_parameters, _ = splprep(spline_input, s=0, k=1, u=curve_phases)

            if self.spline_sample_type in ('evenly spaced', 'from data resample'):
                u = np.linspace(0, max_phase, self.trajectories_resample_length)
            else:
                u = curve_phases

            window = []
            for _ in range(self.imitation_window_size + (self.dynamical_system_order - 1)):
                spline_values = splev(u, spline_parameters)
                position_window = spline_values[: self.dim_workspace]
                window.append(position_window)

                delta_phase = spline_values[-1]
                next_t = u + delta_phase
                u = np.clip(next_t, a_min=0, a_max=max_phase)

                predicted_phase = splev(u, spline_parameters)[-2]
                error_acc.append(float(np.mean(np.abs(predicted_phase - u))))

            resampled_positions.append(window)

        if self.verbose:
            logger.info('Mean error spline resampling: %f', float(np.mean(error_acc)))

        resampled_positions = np.transpose(np.array(resampled_positions), (0, 3, 2, 1))
        return resampled_positions

    def get_limits_derivatives(self, demos):
        """
        Computes velocity and acceleration of the training demonstrations
        """
        velocity = (demos[:, :, :, 1:] - demos[:, :, :, :-1]) / self.delta_t
        acceleration = (velocity[:, :, :, 1:] - velocity[:, :, :, :-1]) / self.delta_t

        min_velocity = np.min(velocity, axis=(0, 1, 3))
        max_velocity = np.max(velocity, axis=(0, 1, 3))

        if self.dynamical_system_order == 1:
            min_acceleration = None
            max_acceleration = None
        elif self.dynamical_system_order == 2:
            min_acceleration = np.min(acceleration, axis=(0, 1, 3))
            max_acceleration = np.max(acceleration, axis=(0, 1, 3))
        else:
            raise ValueError('Selected dynamical system order not valid, options: 1, 2.')

        if self.dynamical_system_order == 2:
            max_velocity_state = max_velocity + (max_velocity - min_velocity) * self.state_increment / 2.0
            min_velocity_state = min_velocity - (max_velocity - min_velocity) * self.state_increment / 2.0
            max_velocity = max_velocity_state
            min_velocity = min_velocity_state

        limits = {
            'vel min train': min_velocity,
            'vel max train': max_velocity,
            'acc min train': min_acceleration,
            'acc max train': max_acceleration,
        }
        return limits
