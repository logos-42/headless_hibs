"""
Twistor-LMT Agent Module
=========================
Agent wrapper for reinforcement learning / control tasks.

Usage:
    agent = TwistorAgent(obs_dim=4, action_dim=2, hidden_dim=32)
    obs = env.reset()
    agent.reset()

    for step in range(max_steps):
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        if done:
            agent.reset()
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union


class TwistorAgent:
    """
    Agent wrapper for Twistor-LMT.

    Provides:
    - Stateful interaction with environment
    - Action selection from observations
    - Reset functionality for new episodes

    Can be initialized in two ways:
    1. With a pre-created model: TwistorAgent(model)
    2. With dimensions: TwistorAgent(obs_dim=4, action_dim=2, hidden_dim=32)
    """

    def __init__(
        self,
        model: nn.Module = None,
        device: str = "cpu",
        obs_dim: int = None,
        action_dim: int = None,
        hidden_dim: int = 32,
    ):
        """
        Initialize agent.

        Args:
            model: TwistorLMT model (or None if using obs_dim/action_dim)
            device: Device for computation
            obs_dim: Observation dimension (if model not provided)
            action_dim: Action dimension (if model not provided)
            hidden_dim: Hidden dimension (if model not provided)
        """
        # If model not provided, create one from dimensions
        if model is None:
            if obs_dim is None or action_dim is None:
                raise ValueError(
                    "Either model or (obs_dim, action_dim) must be provided"
                )
            from .core import TwistorLMT

            model = TwistorLMT(
                input_dim=obs_dim, hidden_dim=hidden_dim, output_dim=action_dim
            )

        self.model = model
        self.device = device
        self.z = None
        self.hidden_dim = model.hidden_dim
        self.obs_dim = model.input_dim
        self.action_dim = model.output_dim

    def reset(self, batch_size: int = 1) -> torch.Tensor:
        """
        Reset agent state for new episode.

        Args:
            batch_size: Number of parallel environments

        Returns:
            Initial hidden state
        """
        self.z = self.model.reset_state(batch_size, self.device)
        return self.z

    def act(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Get action from observation.

        Args:
            obs: Observation (obs_dim,) or (batch, obs_dim)
            deterministic: If True, return raw output (for continuous)

        Returns:
            action: Action (action_dim,) or (batch, action_dim)
        """
        if self.z is None:
            self.reset()

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()

        obs = obs.to(self.device)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            batch_mode = False
        else:
            batch_mode = True

        with torch.no_grad():
            self.z, action = self.model.step(self.z, obs)

        if not batch_mode:
            action = action.squeeze(0)

        return action.cpu().numpy()

    def step(
        self,
        obs: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Step and return both action and state.

        Args:
            obs: Observation

        Returns:
            action: Selected action
            state: New hidden state
        """
        if self.z is None:
            self.reset()

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()

        obs = obs.to(self.device)

        with torch.no_grad():
            self.z, action = self.model.step(self.z, obs)

        return action.cpu().numpy(), self.z

    def update(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update state without getting action (for inference).

        Args:
            obs: Observation tensor

        Returns:
            output: Model output
        """
        if self.z is None:
            self.reset(obs.size(0))

        with torch.no_grad():
            self.z, output = self.model.step(self.z, obs)

        return output


class TwistorAgentWithPolicy(TwistorAgent):
    """
    Agent with additional policy head for RL.
    """

    def __init__(
        self,
        model: nn.Module,
        policy_net: nn.Module,
        device: str = "cpu",
    ):
        super().__init__(model, device)
        self.policy_net = policy_net

    def act_with_policy(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        epsilon: float = 0.1,
    ) -> Tuple[np.ndarray, dict]:
        """
        Act with epsilon-greedy policy.

        Args:
            obs: Observation
            epsilon: Exploration rate

        Returns:
            action: Selected action
            info: Additional info (logprob, value, etc.)
        """
        if self.z is None:
            self.reset()

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()

        obs = obs.to(self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            self.z, raw_output = self.model.step(self.z, obs)

            policy_input = self.model.z.real if hasattr(self.model, "z") else raw_output
            policy_output = self.policy_net(policy_input)

            if np.random.rand() < epsilon:
                action = torch.randint_like(policy_output, policy_output.shape[-1])
            else:
                action = policy_output.argmax(dim=-1)

        info = {
            "raw_output": raw_output.cpu().numpy(),
            "policy_output": policy_output.cpu().numpy(),
        }

        return action.cpu().numpy(), info


class MultiAgent:
    """
    Multi-agent system with multiple TwistorAgents.
    """

    def __init__(
        self,
        model_class,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 32,
        shared_model: bool = True,
        device: str = "cpu",
    ):
        self.num_agents = num_agents
        self.shared_model = shared_model
        self.device = device

        if shared_model:
            self.model = model_class(obs_dim, hidden_dim, action_dim)
            self.agents = [TwistorAgent(self.model, device) for _ in range(num_agents)]
        else:
            self.agents = []
            for _ in range(num_agents):
                model = model_class(obs_dim, hidden_dim, action_dim)
                agent = TwistorAgent(model, device)
                self.agents.append(agent)

    def reset_all(self):
        """Reset all agent states."""
        for agent in self.agents:
            agent.reset()

    def act_all(
        self,
        obs: np.ndarray,
    ) -> np.ndarray:
        """
        Get actions for all agents.

        Args:
            obs: Observations (num_agents, obs_dim)

        Returns:
            actions: Actions (num_agents, action_dim)
        """
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.act(obs[i])
            actions.append(action)
        return np.stack(actions)

    def step_all(
        self,
        obs: np.ndarray,
    ) -> Tuple[np.ndarray, list]:
        """
        Step all agents.

        Args:
            obs: Observations

        Returns:
            actions: All actions
            states: All hidden states
        """
        actions = []
        states = []
        for i, agent in enumerate(self.agents):
            action, state = agent.step(obs[i])
            actions.append(action)
            states.append(state)
        return np.stack(actions), states


def create_agent(
    model_class,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int = 32,
    use_rk4: bool = False,
    device: str = "cpu",
    **model_kwargs,
) -> TwistorAgent:
    """
    Factory function to create TwistorAgent.

    Args:
        model_class: TwistorLMT class
        obs_dim: Observation dimension
        action_dim: Action dimension
        hidden_dim: Hidden dimension
        use_rk4: Use RK4 integration
        device: Device
        **model_kwargs: Additional model arguments

    Returns:
        agent: TwistorAgent instance
    """
    from .core import TwistorLMT

    model = TwistorLMT(
        input_dim=obs_dim, hidden_dim=hidden_dim, output_dim=action_dim, **model_kwargs
    )

    return TwistorAgent(model, device)
