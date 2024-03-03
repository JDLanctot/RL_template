import random
from collections import namedtuple, deque
from typing import Tuple
import torch

from co2.problem import GraphLearningProblem

__all__ = []
__all__.extend([
    'PiTransition',
    'QTransition',
    'ReplayBuffer',
])

PiTransition = namedtuple('Transition',
                        ('s', 'a', 'r', 'terminal'))

QTransition = namedtuple('Transition',
                        ('s', 'a', 's_next', 'dr', 'terminal', 'dt'))


class ReplayBuffer(object):
    """ Replay buffer that memorizes n-step transitions an a reinforcement
    learning environment. """

    def __init__(self, capacity: int, step_diff: int,
                 type: str, device: str = 'cpu'):
        """
        Args:
            capacity: Maximum number of `Transition` objects to store in
                replay buffer before overwriting the earliest.
            step_diff: Positive integer giving the 'N' in 'NStep'.
            type: The type of the Neural Net (PiNet, QNet).
            device: Device to return samples on.
        """
        self.capacity = capacity
        self.step_diff = step_diff
        self.type = type
        self.device = device
        self.memory = deque(maxlen=capacity)

    def store(self, e) -> None:
        """ Memorize all n-step transitions in a terminal environment.

        Args:
            e: A terminal environment.
            player: which player we are storing for.
        """

        assert e.done

        # NEW TRANSITION METHOD
        player_dict = {
            'type': self.type,
            'steps_taken': e.steps_taken,
            'actions': e.actions,
            'states': e.states,
            'completed_state': e.completed_state if self.type == 'Q' else None,
            'memory': self.memory,
            'transition': QTransition if self.type == 'Q' else PiTransition,
            'total_reward': e.r,
            'reward_func': lambda i: e.rewards[i]
        }

        for i in range(player_dict['steps_taken']):
            # required for both Transitions
            i_next = i + self.step_diff
            a = player_dict['actions'][i]
            r = player_dict['reward_func'](i)
            s = player_dict['states'][i]

            # Handle the terminal case of the game
            if i_next >= player_dict['steps_taken']:
                terminal = True
                # only required for Qnet Transition
                if player_dict['type'] == 'qnet':
                    r_next = player_dict['reward_func'](i_next) if player == 1 else e.r
                    s_next = player_dict['completed_state'].clone()
                    dt = player_dict['steps_taken'] - i
            # Handle the all other cases of the game
            else:
                terminal = False
                # only required for Qnet Transition
                if player_dict['type'] == 'qnet':
                    r_next = player_dict['reward_func'](i_next)
                    s_next = player_dict['states'][i_next]
                    dt = self.step_diff
            # save the Transition to memory
            if player_dict['type'] == 'qnet':
                transition = player_dict['transition'](s, a, s_next, r_next - r, terminal, dt)
            else:
                transition = player_dict(s, a, player_dict['total_reward'] - r, terminal)
            player_dict['memory'].append(transition)
            del transition

    def sample(self, batch_size: int) -> Tuple:
        """
        Args:
            batch_size: Number of transitions to randomly sample and batch
                        together.

        Returns:
            A five-tuple (`s_prev`, `a`, `g`, `dr`, `terminal`), where `g`
            (`s_prev`) is the final (initial) state, `a` is the action taken
            in `g`, `dr` is the difference in cumulative reward between the
            final/initial states, and `terminal` is whether `g` is a terminal
            state.
        """
        assert self.type == 'Q'

        device = self.device
        s, a, s_next, dr, terminal, dt = list(
            zip(*random.sample(self.memory, batch_size)))

        a = torch.tensor(a, dtype=torch.long, device=device)
        dr = torch.tensor(dr, dtype=torch.float, device=device)
        terminal = torch.tensor(terminal, dtype=torch.bool, device=device)
        dt = torch.tensor(dt, dtype=torch.float, device=device)

        return s, a, s_next, dr, terminal, dt

    def get(self, batch_size: int) -> Tuple:
        """
        Args:
            batch_size: Number of transitions to sample and batch`
                        together.

        Returns:
            A five-tuple (`r`, `a`, `log_pi`, `entropy`, `terminal`), where `r`
            are the differences in cumulative reward between the final/initial states,
            `a` are the actions taken, and `terminal` is whether the graph is in a terminal state.
        """
        assert self.type == 'Pi'

        memory_queue = self.memory

        if not memory_queue:
            return 0, 0, 0, 0

        dequeued_items = []
        terminal_count = 0
        while memory_queue and terminal_count < batch_size:
            item = memory_queue.popleft()
            dequeued_items.append(item)
            _, _, _, terminal = item
            terminal_count += int(terminal)

        s, a, r, terminal = list(zip(*dequeued_items))
        a = torch.tensor(a, dtype=torch.long, device=self.device)
        r = torch.tensor(r, dtype=torch.float, device=self.device)

        return s, a, r, terminal

    def clear(self) -> None:
        del self.memory
        self.memory = deque(maxlen=self.capacity)
        torch.cuda.empty_cache()

    def __len__(self) -> float:
        return len(self.memory)
