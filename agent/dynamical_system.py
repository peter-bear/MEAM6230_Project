from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from agent.utils.dynamical_system_operations import (
    denormalize_derivative,
    euler_integration,
    normalize_state,
    denormalize_state,
    get_derivative_normalized_state,
)
import logging

logger = logging.getLogger(__name__)


class DynamicalSystem:
    """Dynamical System that uses a learned latent model for transitions.

    The implementation wraps a provided `model` and handles mapping between
    task and latent spaces, integration, and optional obstacle avoidance.
    """

    def __init__(
        self,
        x_init: torch.Tensor,
        order: int,
        min_state_derivative: Tuple[torch.Tensor, Optional[torch.Tensor]],
        max_state_derivative: Tuple[torch.Tensor, Optional[torch.Tensor]],
        saturate_transition: bool,
        primitive_type: Any,
        model: Any,
        dim_state: int,
        delta_t: float,
        x_min: np.ndarray,
        x_max: np.ndarray,
    ) -> None:
        # Model and device
        self.model = model
        try:
            self.device = next(model.parameters()).device
        except Exception:
            self.device = torch.device('cpu')

        # Initialize parameters
        self.order = int(order)
        self.saturate_transition = bool(saturate_transition)
        self.primitive_type = primitive_type
        self.dim_state = int(dim_state)
        self.dim_workspace = dim_state // order

        # derivatives limits (velocity, acceleration)
        self.min_vel = min_state_derivative[0]
        self.max_vel = max_state_derivative[0]
        self.max_vel_norm = torch.max(-self.min_vel, self.max_vel)
        self.min_acc = min_state_derivative[1]
        self.max_acc = max_state_derivative[1]
        if self.min_acc is not None and self.max_acc is not None:
            self.max_acc_norm = torch.max(-self.min_acc, self.max_acc)

        self.delta_t = float(delta_t)
        self.x_min = np.array(x_min)
        self.x_max = np.array(x_max)
        self.batch_size = int(x_init.shape[0])

        # Init dynamical system state
        self.x_t_d = x_init.to(self.device)
        self.y_t: Dict[str, Optional[torch.Tensor]] = {'task': None, 'latent': None}
        self.y_t_d = self.get_latent_state(x_init)

    def get_latent_state(self, x_t=None, space='task'):
        """
        Obtains current latent state by either mapping task state or from previous latent system transition
        """
        if space == 'task':
            # Map state to latent state (psi)
            y_t = self.model.encoder(x_t.to(self.device), self.primitive_type)
            self.y_t['task'] = y_t
        elif space == 'latent':
            # Transition following f^{L}
            _, y_t = self.transition_latent_system()
            self.y_t['latent'] = y_t
        else:
            raise ValueError('Selected transition space not valid, options: task, latent.')

        return y_t

    def transition_latent_system(self, y_t=None, gap=None):
        """
        Computes one-step transition in latent space.
        gap: ||y_t_task - y_t_latent|| from previous step, passed to gain network.
        """
        # If no state provided, assume perfect transition from previous desired state
        if y_t is None:
            y_t = self.y_t_d

        # Get derivative
        dy_t_d = self.model.latent_dynamical_system(y_t, self.primitive_type, gap=gap)

        # Integrate
        self.y_t_d = euler_integration(y_t, dy_t_d, self.delta_t)

        return self.y_t_d, y_t

    def map_to_velocity(self, y_t):
        """
        Maps latent state to task state derivative
        """
        # Get desired velocity (phi)
        dx_t_d_normalized = self.model.decoder_dx(y_t)

        # Denormalize velocity/acceleration
        if self.order == 1:
            dx_t_d = denormalize_derivative(dx_t_d_normalized, self.max_vel_norm)
        elif self.order == 2:
            dx_t_d = denormalize_derivative(dx_t_d_normalized, self.max_acc_norm)
        else:
            raise ValueError('Selected dynamical system order not valid, options: 1, 2.')

        return dx_t_d

    def integrate_1st_order(self, x_t, vel_t_d):
        """
        Saturates and integrates state derivative for first-order systems
        """
        # Clip position (through the velocity)
        if self.saturate_transition:
            max_vel_t_d = (1 - x_t) / self.delta_t
            min_vel_t_d = (-1 - x_t) / self.delta_t
            vel_t_d = torch.clamp(vel_t_d, min_vel_t_d, max_vel_t_d)

        # Integrate
        x_t_d = euler_integration(x_t, vel_t_d, self.delta_t)

        return x_t_d, vel_t_d

    def integrate_2nd_order(self, x_t, acc_t_d):
        """
        Saturates and integrates state derivative for second-order systems
        """
        # Separate state in position and velocity
        pos_t = x_t[:, : self.dim_workspace]
        vel_t = denormalize_state(x_t[:, self.dim_workspace :], self.min_vel, self.max_vel)

        # Clip position and velocity (through the acceleration)
        if self.saturate_transition:
            # Position safety
            max_acc_t_d = (1 - pos_t - vel_t * self.delta_t) / (self.delta_t ** 2)
            min_acc_t_d = (-1 - pos_t - vel_t * self.delta_t) / (self.delta_t ** 2)
            acc_t_d = torch.clamp(acc_t_d, min_acc_t_d, max_acc_t_d)

            # Velocity safety limits
            max_acc_t_d = (self.max_vel - vel_t) / self.delta_t
            min_acc_t_d = (self.min_vel - vel_t) / self.delta_t
            acc_t_d = torch.clamp(acc_t_d, min_acc_t_d, max_acc_t_d)

        # Integrate
        vel_t_d = euler_integration(vel_t, acc_t_d, self.delta_t)
        pos_t_d = euler_integration(pos_t, vel_t_d, self.delta_t)

        # Normalize velocity to have a state space between -1 and 1
        vel_t_d_norm = normalize_state(vel_t_d, self.min_vel, self.max_vel)

        # Create desired state
        x_t_d = torch.cat([pos_t_d, vel_t_d_norm], dim=1)

        return x_t_d, vel_t_d

    def transition(self, x_t=None, space='task', **kwargs):
        """
        Computes dynamical system one-step transition
        """
        # If no state provided, assume perfect transition from previous desired state
        if x_t is None:
            x_t = self.x_t_d

        # Map task state to latent state (psi)
        y_t = self.get_latent_state(x_t, space)

        # Map latent state to task state derivative (vel/acc) (phi)
        dx_t_d = self.map_to_velocity(y_t)

        # Saturate (to keep state inside boundary) and integrate derivative
        if self.order == 1:
            self.x_t_d, vel_t_d = self.integrate_1st_order(x_t, dx_t_d)
        elif self.order == 2:
            self.x_t_d, vel_t_d = self.integrate_2nd_order(x_t, dx_t_d)
        else:
            raise ValueError('Selected dynamical system order not valid, options: 1, 2.')

        # Obstacle avoidance
        if 'obstacles' in kwargs:
            self.x_t_d = self.obstacle_avoidance(x_t, vel_t_d, kwargs['obstacles'])

        # Collect transition info
        transition_info = {'desired state': self.x_t_d,
                           'desired velocity': vel_t_d,
                           'latent state': y_t}

        return transition_info

    def simulate(self, simulation_steps, space='task', **kwargs):
        """
        Simulates dynamical system
        """
        states_history = [self.x_t_d.cpu().detach().numpy()]
        latent_states_history = []
        with torch.no_grad():
            for _ in range(simulation_steps - 1):
                transition_info = self.transition(space=space, **kwargs)
                x_t = transition_info['desired state']
                y_t = transition_info['latent state']

                states_history.append(x_t.cpu().detach().numpy())
                latent_states_history.append(y_t.cpu().detach().numpy())

        return np.array(states_history), np.array(latent_states_history)

    def obstacle_avoidance(self, x_t, dx_t, obstacles):  # TODO: only tested with first order systems
        """
        Ellipsoidal multi-obstacle avoidance (paper: https://cs.stanford.edu/people/khansari/ObstacleAvoidance.html)
        """
        batch_size = dx_t.shape[0]
        n_obs = len(obstacles['centers'])
        x_t = x_t.view(batch_size, self.dim_state)

        delta_x = dx_t * self.delta_t

        if not obstacles['centers']:
            x_t_d = x_t + delta_x
        else:
            # repeat state for each obstacle and transpose to (batch, n_obs, dim_state)
            x_t_rep = x_t.repeat_interleave(repeats=n_obs, dim=1).view(batch_size, self.dim_state, n_obs).transpose(1, 2)

            device = self.device
            obs = torch.tensor(
                normalize_state(np.array(obstacles['centers']), x_min=self.x_min, x_max=self.x_max),
                dtype=torch.float,
                device=device,
            ).unsqueeze(0).repeat(batch_size, 1, 1)

            sf = torch.tensor(obstacles['safety_margins'], dtype=torch.float, device=device).unsqueeze(0).repeat(
                batch_size, 1, 1
            )

            a = torch.tensor(
                get_derivative_normalized_state(np.array(obstacles['axes']), x_min=self.x_min, x_max=self.x_max),
                dtype=torch.float,
                device=device,
            ).unsqueeze(0).repeat(batch_size, 1, 1)

            x_ell = x_t_rep - obs

            a = a * sf
            Gamma = torch.sum((x_ell / a) ** 2, dim=2)
            Gamma = torch.where(Gamma < 1, torch.full_like(Gamma, 1e3), Gamma)

            # Weights computation
            Gamma_k = Gamma.view(batch_size, n_obs, 1).repeat(1, 1, n_obs)
            Gamma_i = Gamma.repeat_interleave(repeats=n_obs, dim=1).view(batch_size, n_obs, n_obs).transpose(1, 2)
            filter_i_eq_k = 1e30 * torch.eye(n_obs, device=device).unsqueeze(0).repeat(batch_size, 1, 1) + torch.ones(
                n_obs, device=device
            ).unsqueeze(0).repeat(batch_size, 1, 1)
            Gamma_i = filter_i_eq_k * Gamma_i
            w = torch.prod((Gamma_i - 1) / ((Gamma_k - 1) + (Gamma_i - 1)), dim=2)

            nv = (2 / a) * (x_ell / a)
            E = torch.zeros(batch_size, n_obs, self.dim_state, self.dim_state, device=device)
            E[:, :, :, 0] = nv
            E[:, :, 0, 1:self.dim_state] = nv[:, :, 1:self.dim_state]

            I = torch.eye(self.dim_state - 1, device=device).repeat(batch_size * n_obs, 1, 1)
            e_last = nv.view(batch_size * n_obs, self.dim_state)[:, 0].view(batch_size * n_obs, 1)[:, :, None]
            E[:, :, 1:self.dim_state, 1:self.dim_state] = (-I * e_last).view(
                batch_size, n_obs, self.dim_state - 1, self.dim_state - 1
            )

            D = torch.zeros(batch_size, n_obs, self.dim_state, self.dim_state, device=device)
            D[:, :, 0, 0] = 1 - (w / Gamma)
            for i in range(self.dim_state - 1):
                D[:, :, i + 1, i + 1] = 1 + (w / Gamma)

            E_flat = E.view(batch_size * n_obs, self.dim_state, self.dim_state)
            D_flat = D.view(batch_size * n_obs, self.dim_state, self.dim_state)
            M = torch.bmm(torch.bmm(E_flat, D_flat), torch.inverse(E_flat))
            M = M.view(batch_size, n_obs, self.dim_state, self.dim_state)

            delta_x_mod = delta_x.view(batch_size, self.dim_state, 1)
            for i in range(n_obs):
                delta_x_mod = torch.bmm(M[:, i, :, :], delta_x_mod)
            delta_x_mod = delta_x_mod.view(batch_size, self.dim_state)

            x_t_d = x_t_rep[:, 0, :] + delta_x_mod

        # Clamp values inside workspace
        for i in range(self.dim_state):
            x_t_d[:, i] = torch.clamp(x_t_d[:, i], self.x_min[i], self.x_max[i])

        return x_t_d
