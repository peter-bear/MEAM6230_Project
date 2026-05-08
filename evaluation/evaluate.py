import numpy as np
import torch
from evaluation.utils.saving import save_stats_txt, save_best_stats_txt, check_gpu
from evaluation.utils.similarity_measures import get_RMSE, get_FD, get_DTWD
from agent.utils.dynamical_system_operations import denormalize_state, normalize_state


class Evaluate:
    """
    Class for evaluating learned dynamical system
    """
    def __init__(self, learner, data, params, verbose=True):
        self.learner = learner
        self.verbose = verbose
        # Device (fall back to CPU if learner doesn't expose device)
        self.device = getattr(learner, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Params file parameters
        self.fixed_point_iteration_thr = params.fixed_point_iteration_thr
        self.dim_state = params.workspace_dimensions * params.dynamical_system_order
        self.dim_workspace = params.workspace_dimensions
        self.ignore_n_spurious = params.ignore_n_spurious
        self.multi_motion = params.multi_motion
        self.quanti_eval = params.quanti_eval
        self.quali_eval = params.quali_eval
        self.diffeo_quanti_eval = params.diffeo_quanti_eval
        self.diffeo_quali_eval = params.diffeo_quali_eval
        self.diffeo_eval = self.diffeo_quanti_eval or self.diffeo_quali_eval
        self.density = params.density
        self.simulated_trajectory_length = params.simulated_trajectory_length
        self.dynamical_system_order = params.dynamical_system_order

        # Parameters data processor
        self.primitive_ids = np.array(data['demonstrations primitive id'])
        self.n_primitives = data['n primitives']
        self.delta_t_eval = data['delta t eval']
        self.n_trajectories = data['n demonstrations']
        self.demonstrations_train = data['demonstrations train']
        self.eval_indexes = data['eval indexes']
        self.max_trajectory_length = data['max demonstration length']
        self.goals = data['goals training']
        self.demonstrations_eval = data['demonstrations raw']
        self.x_min = np.array(data['x min'])
        self.x_max = np.array(data['x max'])

        # Variables
        self.best_metric, self.best_n_spurious = 1e7, 1e7
        self.best_RMSE, self.best_DTWD, self.best_FD = 0, 0, 0
        self.RMSE, self.DTWD, self.FD = [], [], []
        self.DTWD_std = []
        self.n_spurious, self.mean_distance_to_goal = [], []
        self.best_model = False

    def get_initial_states_grid(self):
        """
        Samples initial states from a grid in the state space
        """
        # Create workspace grid [-1, 1] x [-1, 1] x ...
        starting_points = np.linspace(-1, 1, self.density)
        starting_points = [starting_points] * self.dim_workspace  # repeat points per dimension workspace
        grid = np.meshgrid(*starting_points)

        # Transform grid into tensor that pytorch can use
        initial_positions_grid = torch.empty(0)
        for i in range(self.dim_workspace):
            initial_positions_grid = torch.cat([initial_positions_grid,
                                                torch.from_numpy(grid[i].reshape(-1, 1)).float()], dim=1)

        initial_positions_grid = initial_positions_grid.to(self.device)

        # Get initial derivatives and append to initial states (for second order systems)
        initial_derivatives_grid = torch.zeros([initial_positions_grid.shape[0], self.dim_state - self.dim_workspace], device=self.device)

        # Get initial states
        initial_states_grid = torch.cat([initial_positions_grid, initial_derivatives_grid], dim=1)

        return initial_states_grid, grid

    def get_initial_states_demos(self, primitive_id):
        """
        Gets initial states present in the demonstrations
        """
        demos = self.demonstrations_train[self.primitive_ids == primitive_id]

        # Get initial positions
        initial_positions_demos = torch.empty(0)
        for i in range(self.dim_workspace):
            initial_positions_demos = torch.cat([initial_positions_demos,
                                                 torch.from_numpy(demos[:, 0, i, 0].reshape(-1, 1)).float()], dim=1)

        initial_positions_demos = initial_positions_demos.to(self.device)

        # Get initial derivatives and append to initial states (second order systems)
        initial_derivatives_demos = torch.zeros([initial_positions_demos.shape[0], self.dim_state - self.dim_workspace], device=self.device)

        # Get initial states
        initial_states_demos = torch.cat([initial_positions_demos, initial_derivatives_demos], dim=1)

        return initial_states_demos

    def simulate_system(self, primitive_id, space='task', **kwargs):
        """
        Simulates dynamical system when starting from grid initial states and demonstrations initial states
        """
        # Get initial states of demonstrations
        initial_states_demos = self.get_initial_states_demos(primitive_id)

        # Get equally-spaced initial states through the workspace
        initial_states_grid, grid = self.get_initial_states_grid()

        # Get n demos in primitive
        n_trajectories_primitive = self.demonstrations_train[self.primitive_ids == primitive_id].shape[0]

        # Get primitive number to feed model
        primitive_type = torch.ones(self.density ** self.dim_workspace + n_trajectories_primitive, device=self.device) * primitive_id

        # Combine states
        initial_states = torch.cat([initial_states_demos, initial_states_grid], dim=0)

        # Simulate trajectories
        dynamical_system = self.learner.init_dynamical_system(initial_states, primitive_type)
        visited_states, _ = dynamical_system.simulate(self.simulated_trajectory_length, space=space, **kwargs)

        # Separate grid/demos states
        visited_states_demos = visited_states[:, :n_trajectories_primitive]
        visited_states_grid = visited_states[:, n_trajectories_primitive:]

        # Out dict
        results = {'visited states demos': visited_states_demos,
                   'visited states grid': visited_states_grid,
                   'initial states grid': initial_states_grid,
                   'grid': grid}

        return results

    def get_vector_field(self, initial_states, primitive_id, **kwargs):
        """
        Computes velocity of initial states
        """
        # Get primitive number to feed model
        primitive_type = torch.ones(self.density ** self.dim_workspace, device=self.device) * primitive_id

        # Compute evaluation delta t
        delta_t_eval = np.mean(self.delta_t_eval)

        # Do one transition
        with torch.no_grad():
            dynamical_system = self.learner.init_dynamical_system(initial_states, primitive_type)
            x_t = dynamical_system.transition(space='task', **kwargs)['desired state']

        # Denormalize states
        x_init_denorm = denormalize_state(initial_states[:, :self.dim_workspace].cpu().detach().numpy(),
                                          x_min=np.array(self.x_min),
                                          x_max=np.array(self.x_max))

        x_t_denorm = denormalize_state(x_t[:, :self.dim_workspace].cpu().detach().numpy(),
                                       x_min=np.array(self.x_min),
                                       x_max=np.array(self.x_max))

        # Compute differences in X
        vel = (x_t_denorm - x_init_denorm) / delta_t_eval

        return vel.reshape(self.density, self.density, -1)

    def get_stability_metrics(self, attractor, goal):
        """
        Based on trajectories obtained when starting from grid initial states, computes stability metrics
        """
        # Get goal
        goal = denormalize_state(np.array(goal), self.x_min, self.x_max)

        # Compute distance between goal and last point
        distance_to_goal = np.linalg.norm(attractor - goal, axis=1)

        # Find spurious/unsuc. trajectory and mean distance to goal
        n_spurious = np.sum(distance_to_goal > self.fixed_point_iteration_thr)
        mean_dist_to_goal = np.mean(distance_to_goal)

        # Append results to history
        self.mean_distance_to_goal.append(mean_dist_to_goal)
        self.n_spurious.append(n_spurious)

        # Get results
        metrics_stab = {'n spurious': n_spurious,
                        'mean dist to goal': mean_dist_to_goal}

        return metrics_stab

    def get_accuracy_metrics(self, visited_states, demonstrations_eval, max_trajectory_length, eval_indexes, goal):
        """
        Gets accuracy metrics to evaluate the performance of the learned dynamical system
        """

        # Denormalize and preprocess trajectories
        sim_trajectories = denormalize_state(visited_states[:, :, :self.dim_workspace], self.x_min, self.x_max)
        demos = self.preprocess_demonstrations_eval(demonstrations_eval, visited_states.shape[1],
                                                    max_trajectory_length)
        # Denormalize goal for convergence checks.
        goal_denorm = denormalize_state(np.array(goal).reshape(1, -1), self.x_min, self.x_max)[0]

        n_trajectories = visited_states.shape[1]
        sim_length = sim_trajectories.shape[0]

        sim_full_indexes = []
        demo_full_indexes = []

        for i in range(n_trajectories):
            demo_real_length = len(demonstrations_eval[i][0])

            # Early-stop index when the trajectory gets close enough to the goal.
            dist_to_goal = np.linalg.norm(sim_trajectories[:, i, :] - goal_denorm, axis=1)

            # Relative convergence threshold at 1% of workspace span.
            traj_range = np.linalg.norm(self.x_max - self.x_min)
            convergence_thr = traj_range * 0.01
            converged_indices = np.where(dist_to_goal < convergence_thr)[0]

            if len(converged_indices) > 0:
                sim_real_length = converged_indices[0] + 1
            else:
                sim_real_length = sim_length

            if self.verbose:
                print(f'Trajectory {i}: convergence length {sim_real_length} / {sim_length}')

            sim_full_indexes.append(np.arange(sim_real_length))
            demo_full_indexes.append(np.arange(demo_real_length))

        # Calculate RMSE
        # Calculate RMSE between learned velocities and reference velocities.
        RMSE = self.get_velocity_RMSE(demonstrations_eval, eval_indexes)

        # Calculate DTWD
        DTWD = get_DTWD(sim_trajectories, demos, sim_full_indexes, demo_full_indexes, verbose=self.verbose)

        # Calculate Frechet Distance
        FD = get_FD(sim_trajectories, demos, eval_indexes, verbose=self.verbose)

        # Get average over demonstrations
        mean_RMSE = np.array(RMSE).mean()
        mean_DTWD = np.array(DTWD).mean()
        std_DTWD = np.array(DTWD).std()
        mean_FD = np.array(FD).mean()

        # Sum metrics
        self.metrics_sum = mean_RMSE + mean_DTWD + mean_FD

        # Append results to history
        self.RMSE.append(mean_RMSE)
        self.DTWD.append(mean_DTWD)
        self.DTWD_std.append(std_DTWD)
        self.FD.append(mean_FD)

        # Get results
        results = {'RMSE': mean_RMSE,
                   'DTWD': mean_DTWD,
                   'DTWD_std': std_DTWD,
                   'FD': mean_FD,
                   'metrics sum': self.metrics_sum}

        return results

    def get_velocity_RMSE(self, demonstrations_eval, eval_indexes):
        """
        Computes RMSE between the learned first-order velocity field and reference demo velocities.
        """
        RMSE = []
        for k in range(len(demonstrations_eval)):
            if self.verbose:
                print('Calculating velocity RMSE; trajectory:', k + 1)

            demo = np.array(demonstrations_eval[k]).T
            valid_indexes = eval_indexes[k][eval_indexes[k] < demo.shape[0] - 1]
            dt = self.get_demo_delta_t(k)

            if isinstance(dt, (np.ndarray, list)):
                dt = np.array(dt)
                if dt.size > 1:
                    dt = dt[valid_indexes].reshape(-1, 1)
                else:
                    dt = float(dt.item() if hasattr(dt, 'item') else dt[0])

            positions = demo[valid_indexes]
            reference_velocity = (demo[valid_indexes + 1] - demo[valid_indexes]) / dt

            positions_normalized = normalize_state(positions, self.x_min, self.x_max)
            states = torch.from_numpy(positions_normalized).float().to(self.device)

            if self.dynamical_system_order == 2:
                raise NotImplementedError('Velocity RMSE is currently implemented for first-order systems only.')

            primitive_type = torch.ones(states.shape[0], device=self.device) * self.primitive_ids[k]
            with torch.no_grad():
                dynamical_system = self.learner.init_dynamical_system(states, primitive_type)
                learned_velocity_normalized = dynamical_system.transition(states)['desired velocity']

            learned_velocity_normalized = learned_velocity_normalized.cpu().detach().numpy()
            learned_velocity = learned_velocity_normalized * (self.x_max - self.x_min) / (2 * dt)
            RMSE.append(np.sqrt(np.mean((learned_velocity - reference_velocity) ** 2)))

        return RMSE

    def get_demo_delta_t(self, demo_index):
        """
        Gets the evaluation sampling time for one demonstration.
        """
        if np.isscalar(self.delta_t_eval):
            return self.delta_t_eval
        return self.delta_t_eval[demo_index]

    def preprocess_demonstrations_eval(self, demonstrations_eval, n_trajectories, max_trajectory_length):
        """
        Add zeros at the end of demonstrations so that they have the same length and can be stored in a numpy array
        """

        # Initialize padded demos
        demos_padded = np.empty([max_trajectory_length, n_trajectories, self.dim_workspace])

        # Add zeros to each trajectory such that they all have the same length as the longest one
        for i in range(n_trajectories):
            for j in range(self.dim_workspace):
                demonstrations_eval_i_j = demonstrations_eval[i][j]
                length_diff = max_trajectory_length - len(demonstrations_eval_i_j)
                demonstrations_eval_i_j = np.append(demonstrations_eval_i_j, np.zeros(length_diff))
                demos_padded[:, i, j] = demonstrations_eval_i_j

        return demos_padded

    def compute_quanti_eval(self, sim_results, attractor, primitive_id):
        """
        Computes quantitative evaluation metrics
        """
        # Get accuracy metrics
        metrics_acc = self.get_accuracy_metrics(sim_results['visited states demos'],
                                                self.demonstrations_eval,
                                                self.max_trajectory_length,
                                                self.eval_indexes,
                                                self.goals[primitive_id])

        # Get stability metrics
        metrics_stab = self.get_stability_metrics(attractor, self.goals[primitive_id])

        # Select best model
        self.select_best_model(metrics_acc, metrics_stab)

        return metrics_acc, metrics_stab

    def select_best_model(self, metrics_acc, metrics_stab):
        """
        Based on the computed metrics, selects best model so far
        """
        self.best_model = False

        # Select best model
        lower_metric_error = metrics_acc['metrics sum'] < self.best_metric

        if not self.ignore_n_spurious:
            fewer_spurious = metrics_stab['n spurious'] < self.best_n_spurious
            same_spurious = metrics_stab['n spurious'] == self.best_n_spurious
        else:  # we only pass if the metric error is lower
            fewer_spurious = False  # ignore this value in the logical statement
            same_spurious = True  # set this one to true, so that we depend on the metric error

        if fewer_spurious or (same_spurious and lower_metric_error):  # give priority to number of spurious attractors
            self.best_n_spurious = metrics_stab['n spurious']
            self.best_metric = metrics_acc['metrics sum']
            self.best_RMSE = metrics_acc['RMSE']
            self.best_DTWD = metrics_acc['DTWD']
            self.best_DTWD_std = metrics_acc['DTWD_std']
            self.best_FD = metrics_acc['FD']
            self.best_model = True

    def compute_quali_eval(self, sim_results, attractor, primitive_id, iteration):
        """
        Qualitative evaluation, to be computed in subclass
        """
        raise NotImplementedError('Should be implemented in subclass!')

    def compute_diffeo_quanti_eval(self, sim_results, sim_results_latent):
        """
        Computes diffeomorphism mismatch error
        """
        # Get RMSE between task space and latent space trajectory
        length_demonstrations = self.demonstrations_eval[0][0].shape[0]
        visited_positions_denormalized = denormalize_state(sim_results['visited states grid'][:length_demonstrations, :, :self.dim_workspace], self.x_min, self.x_max)
        visited_positions_latent_denormalized = denormalize_state(sim_results_latent['visited states grid'][:length_demonstrations, :, :self.dim_workspace], self.x_min, self.x_max)
        RMSE_trajectory_comparison = get_RMSE(visited_positions_denormalized, visited_positions_latent_denormalized, verbose=False)

        # Compute mean
        mean_RMSE_trajectory_comparison = np.mean(RMSE_trajectory_comparison)

        return mean_RMSE_trajectory_comparison

    def compute_diffeo_quali_eval(self, sim_results, sim_results_latent, primitive_id, iteration):
        """
        Qualitative evaluation diffeomorphism, to be computed in subclass
        """
        raise NotImplementedError('Should be implemented in subclass!')

    def run(self, iteration):
        """
        Runs complete evaluation
        """
        metrics_acc, metrics_stab = {}, {}

        # Iterate when multi-motion learning
        for primitive_id in range(self.n_primitives):
            # Simulate trajectories starting from initial conditions demos and from equally-spaced grid
            sim_results = self.simulate_system(primitive_id)

            # Get last point trajectories grid
            attractor = denormalize_state(sim_results['visited states grid'][-1, :, :self.dim_workspace], self.x_min, self.x_max)

            if self.quanti_eval:
                metrics_acc, metrics_stab = self.compute_quanti_eval(sim_results, attractor, primitive_id)

            if self.quali_eval:
                self.compute_quali_eval(sim_results, attractor, primitive_id, iteration)

            if self.diffeo_eval:
                # Simulate trajectories starting from initial conditions demos and from equally-spaced grid
                sim_results_latent = self.simulate_system(primitive_id, space='latent')

                if self.diffeo_quanti_eval:
                    metrics_stab['diffeo mismatch'] = self.compute_diffeo_quanti_eval(sim_results, sim_results_latent)

                if self.diffeo_quali_eval:
                    self.compute_diffeo_quali_eval(sim_results, sim_results_latent, primitive_id, iteration)

        return metrics_acc, metrics_stab

    def save_progress(self, save_path, i, model, writer):
        """
        Save and log
        """

        # Log metrics to tensorboard
        writer.add_scalar('eval/RMSE', np.mean(self.RMSE[-1]), i)
        writer.add_scalar('eval/DTWD', np.mean(self.DTWD[-1]), i)
        writer.add_scalar('eval/DTWD_std', self.DTWD_std[-1], i)
        writer.add_scalar('eval/FD', np.mean(self.FD[-1]), i)
        writer.add_scalar('eval/n_spurious', self.n_spurious[-1], i)
        writer.add_scalar('eval/mean_distance_to_goal', self.mean_distance_to_goal[-1], i)

        # Save metrics history
        np.save(save_path + 'stats/n_unsuccesful_trajs', self.n_spurious)
        np.save(save_path + 'stats/mean_distance_to_goal', self.mean_distance_to_goal)
        np.save(save_path + 'stats/RMSE', self.RMSE)
        np.save(save_path + 'stats/DTWD', self.DTWD)
        np.save(save_path + 'stats/FD', self.FD)

        # Save states
        save_stats_txt(save_path, self.n_spurious, self.metrics_sum, self.RMSE, self.DTWD, self.FD, i)

        if self.best_model or self.multi_motion:  # metric evaluation is not yet properly done in multi task case
            # Save torch
            torch.save(model.state_dict(), save_path + 'model')

            # Save best stats in txt
            gpu_status = check_gpu()
            save_best_stats_txt(save_path, self.best_n_spurious, self.best_metric, self.best_RMSE, self.best_DTWD,
                                self.best_FD, self.best_DTWD_std, gpu_status, i)