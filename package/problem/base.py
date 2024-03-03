import abc
from copy import copy, deepcopy
from abc import ABC
from typing import Tuple, Type, List

import torch
from abstractcp import Abstract, abstract_class_property
from gym import Env

__all__ = []
__all__.extend([
    'LearningProblem'
])

class GraphLearningProblem(Env, ABC, Abstract):
    _prob_registry = {}

    def __init__(self, d, device='cpu', **kwargs):
        super().__init__()
        self.device = device

        self.g = deepcopy(d)
        self.state = deepcopy(d)

        self._initial_state = self.state.clone()
        self.completed_state = []

        self.states = []
        self.actions = []
        self.rewards = []
        self.steps_taken = 0
        self.r = 0.0

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        prob_name = cls.__name__.strip().lower()
        if prob_name in cls._prob_registry:
            raise TypeError(
                f"There is already a LearningProblem subclass with name "
                f"'{prob_name}.")
        cls._prob_registry[prob_name] = cls

    @classmethod
    def get_prob_cls(cls, prob_name: str) -> Type['LearningProblem']:
        try:
            return cls._prob_registry[str(prob_name).strip().lower()]
        except KeyError:
            raise ValueError(
                f"There is no LearningProblem subclass with name '"
                f"{prob_name}'.")

    @classmethod
    def create(cls, prob_name: str, **prob_kwargs) -> 'LearningProblem':
        prob_cls = cls.get_prob_cls(prob_name)
        return prob_cls(**prob_kwargs)

    @classmethod
    def valid_action_mask(cls, d) -> torch.tensor:
        """ Boolean vector specifying whether each option is a valid action. """
        return ~d.y.bool()

    @property
    @abc.abstractmethod
    def done(self) -> bool:
        """ Whether or not the current state is terminal. """
        pass

    @abc.abstractmethod
    def reward(self, node) -> float:
        """ Reward for taking action specified by node. """
        pass

    @property
    @abc.abstractmethod
    def sol_size(self) -> float:
        """ Human-interpretable quantification of the current solution size."""
        pass

    @abc.abstractmethod
    def update_node(self, node) -> None:
        """ Logic for updating state upon taking action specified by node. """
        pass

    @abc.abstractmethod
    def update_edge(self, edge, rev_edge) -> None:
        """ Logic for updating state upon taking action specified by edge. """
        pass

    @property
    def action_space(self) -> torch.tensor:
        return torch.nonzero(self.valid_action_mask(self.state))

    def step(self, action) -> Tuple:
        # record current status in history
        self.states.append(self.state.shallow_clone())
        self.actions.append(action)

        self.rewards.append(self.r)

        # update state and reward
        dr = self.reward(action)
        self.r += dr
        self.update(action)
        self.steps_taken += 1

        return self.state, dr, self.done, None

    def reset_game(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.r = 0.0
        self.steps_taken = 0
        self.state = self._initial_state.clone()
        torch.cuda.empty_cache()

    def render(self, mode: str = 'human'):
        raise NotImplementedError(
            "Rendering not supported.")
