import importlib
import os
from datetime import datetime

import numpy as np
import torch

from agent.utils.dynamical_system_operations import denormalize_state, normalize_state
from initializer import initialize_framework


PARAMS_NAME = '1st_order_isaac_sim'
RESULTS_BASE_DIRECTORY = './'
INITIAL_STATE = np.array([
	1.6684045,
	-0.7730321,
	-2.020638,
	-2.6644282,
	-1.4719836,
	2.458772,
	1.8989067,
], dtype=np.float32)
ROLL_OUT_STEPS = 10
DELTA_T = 0.01
PRIMITIVE_ID = 0


def parse_initial_state(initial_state_text, expected_dim):
	"""
	Parses a comma/space separated state string into a numpy vector.
	"""
	values = np.fromstring(initial_state_text.replace('[', ' ').replace(']', ' ').replace(',', ' '), sep=' ')
	if values.size != expected_dim:
		raise ValueError(f'Expected {expected_dim} initial joint values, got {values.size}.')
	return values


def denormalize_velocity(velocity_normalized, x_min, x_max):
	"""
	Converts a velocity in normalized state coordinates back to physical joint units.
	"""
	scale = (np.asarray(x_max) - np.asarray(x_min)) / 2.0
	return velocity_normalized * scale


def rollout_inference(learner, initial_state, x_min, x_max, primitive_type, steps, delta_t=1.0):
	"""
	Runs a recursive rollout where each predicted next state becomes the next input.
	"""
	current_state_phys = np.asarray(initial_state, dtype=np.float32).reshape(1, -1)
	current_state_norm = normalize_state(current_state_phys, x_min=x_min, x_max=x_max)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	current_state = torch.from_numpy(current_state_norm).float().to(device)

	dynamical_system = learner.init_dynamical_system(initial_states=current_state,
											primitive_type=primitive_type,
											delta_t=delta_t)

	state_history_phys = [current_state_phys.squeeze(0)]
	state_history_norm = [current_state_norm.squeeze(0)]
	velocity_history_phys = []
	velocity_history_norm = []

	with torch.no_grad():
		for _ in range(steps):
			transition = dynamical_system.transition(space='task')

			next_state = transition['desired state']
			next_velocity = transition['desired velocity']

			next_state_norm = next_state[:, :current_state.shape[1]].detach().cpu().numpy()
			next_state_phys = denormalize_state(next_state_norm, x_min=x_min, x_max=x_max)
			next_velocity_norm = next_velocity.detach().cpu().numpy()
			next_velocity_phys = denormalize_velocity(next_velocity_norm, x_min=x_min, x_max=x_max)

			state_history_norm.append(next_state_norm.squeeze(0))
			state_history_phys.append(next_state_phys.squeeze(0))
			velocity_history_norm.append(next_velocity_norm.squeeze(0))
			velocity_history_phys.append(next_velocity_phys.squeeze(0))

			current_state = next_state[:, :current_state.shape[1]]

	return {
		'states_phys': np.asarray(state_history_phys),
		'states_norm': np.asarray(state_history_norm),
		'velocities_phys': np.asarray(velocity_history_phys),
		'velocities_norm': np.asarray(velocity_history_norm),
	}


def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	Params = getattr(importlib.import_module('params.' + PARAMS_NAME), 'Params')
	params = Params(RESULTS_BASE_DIRECTORY)
	params.results_path += params.selected_primitives_ids + '/'
	params.load_model = True
	params.save_evaluation = False

	learner, _, data = initialize_framework(params, PARAMS_NAME, verbose=False)

	initial_state = INITIAL_STATE
	if initial_state.shape[0] != params.workspace_dimensions:
		raise ValueError(f'INITIAL_STATE must have length {params.workspace_dimensions}.')

	if params.multi_motion:
		primitive_type = torch.tensor([PRIMITIVE_ID], dtype=torch.float, device=device)
	else:
		primitive_type = None

	rollout = rollout_inference(learner=learner,
								initial_state=initial_state,
								x_min=np.asarray(data['x min']),
								x_max=np.asarray(data['x max']),
								primitive_type=primitive_type,
								steps=ROLL_OUT_STEPS,
								delta_t=DELTA_T)

	output_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'isaac_sim_data', 'inference_results')
	os.makedirs(output_dir, exist_ok=True)

	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	csv_path = os.path.join(output_dir, f'joint_positions_{timestamp}.csv')

	header = ['Frame'] + [f'J{i}' for i in range(params.workspace_dimensions)] + ['Col8', 'Col9']
	csv_data = []
	for frame_index in range(rollout['states_phys'].shape[0]):
		row = [frame_index]
		row += rollout['states_phys'][frame_index].tolist()
		row += [0.0, 0.0]
		csv_data.append(row)

	np.savetxt(csv_path, np.asarray(csv_data), delimiter=',', header=','.join(header), comments='', fmt='%.8f')

	print(f'Saved rollout to: {output_dir}')
	print(f'CSV: {csv_path}')
	print('Final joint position:', rollout['states_phys'][-1])


if __name__ == '__main__':
	main()
