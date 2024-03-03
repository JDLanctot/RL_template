from __future__ import annotations

from typing import Union, List, Tuple
import torch
from more_itertools import chunked
from tqdm import tqdm

from co2.problem import GraphLearningProblem
from co2.pinet import PiNet
from co2.qnet import QNet

__all__ = []
__all__.extend([
    'Agent'
])


class Agent(object):

    def __init__(self, net: Union[PiNet, QNet], eps: float = 0.0,
                 policy: str = 'pinet'):
        """
        Args:
            net: The evaluation (Pi or Q) network. Its forward method should have
                 the signature (g, a, a_idx=None) -> log_pi, where g is a
                 GraphLearningState or GraphLearningBatch, a is a 1D tensor of
                 actions (node indices) to evaluate, and a_idx (optional)
                 is an equal-length tensor giving the graph index of each
                 node (if g is batched).
            eps: epsilon for random selection
            policy: The type of learning
        """
        self.net = net
        self.eps = eps
        self.policy = policy


    def get_action(self, s,
                   valid: torch.tensor,
                   full_output=False) -> Union[torch.tensor, Tuple[torch.tensor, torch.tensor, torch.tensor]]:
        """ Determine the best action to take on a given data state,
            and optionally return the associated log_pi-value. Works on batched
            states also, instead returning a tensor of best actions (and a
            tensor of associated log_pi-values).

        Args:
            s: The data state.
            valid: boolean mask indicating which nodes in `s` are valid moves.
            full_output: Whether to output the log probabilities and entropy as well.

        Returns:
            If `full_output` is `True`, a tuple (a, log_pi) where `a` is the best
            action and `log_pi` is the log_pi value. Otherwise just `a`.
        """

        device = s.edge_index.device
        with torch.no_grad():
            # Get the  respective neural net output
            a_offsets = get_edge_offsets(s).to(device=device)

            # net as an Pi Learning Agent
            if self.policy == 'Pi':
                log_pi = self.net(s)

            else:
                q = self.net(s).view(-1)

                # randomly select data that should receive random actions
                random = torch.rand(s.num_graphs, device=q.device).lt(self.eps)

                # fill those data q values with random numbers
                q[random[s.batch]] = torch.rand_like(q[random[s.batch]])

            # pick the actions that will be returned based on the learning policy
            if self.policy == 'Q':
                # set the q value of invalid moves to be -inf (so never selected)
                q[~valid] = float("-inf")

                q_best, idx_best = ts.scatter_max(q, s.batch)

                a_best = idx_best - a_offsets

                # idx_best will be equal to len(q) if there is no maximum
                # (i.e., if all q values for a given graph are -inf, meaning
                # there are no valid moves). These correspond to terminal states,
                # so the action should be irrelevant. Set to 0.
                no_valid_action = idx_best == len(q)
                a_best[no_valid_action] = 0

            else:
                # set the log_pi value of invalid moves to be -inf (so never selected)
                log_pi[~valid] = float("-inf")
                _, idx_best = ts.scatter_max(log_pi + Gumbel(0, 1).sample(log_pi.shape).to(device), s.batch)
                a_best = idx_best - a_offsets

                # idx_best will be equal to len(log_pi) if there is no maximum
                # (i.e., if all log_pi values for a given graph are -inf, meaning
                # there are no valid moves). These correspond to terminal states,
                # so the action should be irrelevant. Set to 0.
                no_valid_action = idx_best == len(log_pi)
                a_best[no_valid_action] = 0

            # return correct parameters
            if full_output:
                if self.policy == 'qnet':
                    return a_best, q_best, torch.zeros_like(q_best)
                else:
                    entropy = ts.scatter_add(-(torch.nan_to_num(log_pi, neginf=0) * log_pi.exp()), s.batch).detach()
                    return a_best, log_pi[idx_best], entropy
            else:
                return a_best

    def play(self, envs: List,
             batch_size: int = None,
             pbar: Union[bool, tqdm] = False) -> None:
        """ Play through one or more problem from start to finish, using
            an epsilon-greedy strategy. Steps through the problem
            in-place so that they can later be, e.g., stored in a replay
            buffer or analyzed.

        Args:
            envs: Collection of problem to play through.
            batch_size: Number of problem to play through at one time.
            pbar: If an instance of `tqdm` is supplied, will show progress
                using an existing progress bar. If `True`, will create a new
                one. If `False', no progress will be shown.
        """
        envs = list(envs)

        if batch_size is None:
            batch_size = len(envs)

        if pbar is True:
            pbar = tqdm(total=len(envs), desc="Playing")

        if pbar:
            pbar.update(sum(e.done for e in envs))

        for env_batch in chunked(envs, batch_size):
            while unfinished := list(filter(lambda e: not e.done, env_batch)):
                # batch the states together
                batch = [e.state for e in unfinished]

                valid = unfinished.valid_action_mask(batch)

                actions = self.get_action(batch, valid)
                for env, a in zip(unfinished, actions):
                    _, done, _ = env.step(a.item())
                    if pbar:
                        pbar.update(done)
                        pbar.refresh()

                # garbage collect to avoid memory leaks with Tensors
                del actions, valid, a
                torch.cuda.empty_cache()

