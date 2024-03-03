import random
from copy import deepcopy
from typing import Iterable, Callable, Type, List, Union, Tuple

import networkx as nx
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from functools import partial

from co2.agent import Agent
from co2.params import TrainParams
from co2.problem import GraphLearningProblem
from co2.pinet import PiNet
from co2.qnet import QNet
from co2.replay import ReplayBuffer

__all__ = []
__all__.extend([
    'Trainer'
])

optimizers = {'adam': optim.Adam,
              'adamw': optim.AdamW,
              'sgd': optim.SGD,
              'rmsprop': optim.RMSprop,
              'adadelta': optim.Adadelta}

BOLD = '\033[1m'
END = '\033[0m'

l_bar = '{desc}: {percentage:3.0f}%|'
r_bar = '| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]'
pbar_format = f"{l_bar}{{bar}}{r_bar}"


class Trainer(object):
    """ Trains an Agent via stochastic gradient descent using Double-Q learning
    with hard target updates. Agent uses an epsilon-greedy strategy to balance
    exploration vs. exploitation, with a linear decay in epsilon. """

    def __init__(self, net: Union[PiNet,QNet],
                 prob_cls: Type[GraphLearningProblem],
                 p: TrainParams,
                 player_type: str = 'Q',
                 *,
                 device: str = 'cpu',
                 prob_kwargs: dict = None) -> None:
        """
        Args:
            net: Pi-network or Q-network, with weights already initialized, to be trained
            prob_cls: Subclass of GraphLearningProblem corresponding to the
                game being played. Its constructor must be invocable as
                `prob_cls(g, **prob_kwargs)`, where `g` is a NetworkX
                (Di)Graph.
            p: Hyper parameters to use in training process.
            player_type: Q for QLearning and Pi for PiLeaning.
            device: Device ('cpu', 'cuda', etc.) on which to perform all
                forward/backward operations.
            prob_kwargs: Dictionary of keyword args to pass to `prob_cls`.
        """

        super().__init__()

        # set up nets for Learning
        self.net = net
        self.net.to(device)

        # set up target nets if QLearning
        self.target_net = deepcopy(self.net)
        self.loss_func = torch.nn.MSELoss()

        # save macro parameters
        self.p = p
        self.player_type = player_type
        self.prob_cls = prob_cls
        self.device = device

        if prob_kwargs is None:
            prob_kwargs = dict()
        self.prob_kwargs = prob_kwargs

        # initialize the game and memory
        self.envs = None
        self.epoch = 0
        self.skip_pi = True

        # set up the optimizers and schedulers
        weights = self.net.parameters()
        optimizer_kwargs = p.optimizer_kwargs
        optimizer_name = p.optimizer.strip().lower()

        Optimizer = partial(optimizers[optimizer_name])
        Pi_Scheduler = partial(optim.lr_scheduler.OneCycleLR,
                               max_lr=self.p.optimizer_kwargs.get('lr'),
                               total_steps=int(self.p.max_epochs / self.p.rollout_freq))
        Q_Scheduler = partial(optim.lr_scheduler.OneCycleLR,
                              max_lr=self.p.optimizer_kwargs.get('lr'),
                              total_steps=self.p.max_epochs)

        self.Optimizer = Optimizer(weights, **optimizer_kwargs)
        self.Scheduler = Q_Scheduler(optim) if self.player_types == 'Q' else Pi_Scheduler(optim)

        # set up the training and validation data
        self.train_data = []
        self.validation_data = []

        # setup the progress bars
        self.train_pbar = tqdm(total=p.max_epochs, position=0,
                               desc="Training", unit=" epochs",
                               leave=False, disable=None)
        self.status_pbar = tqdm(position=2, total=0,
                                bar_format=f"{BOLD}loss: {{postfix[0]:1.2e}}"
                                           f"  |  "
                                           f"Avg. sol. size: "
                                           f"{{postfix[1]:.3f}}{END}",
                                postfix=[np.nan, np.nan],
                                disable=None)
        self.play_pbar = tqdm(total=p.num_play, position=1,
                              desc="    Playing", unit=" experiences",
                              bar_format=pbar_format,
                              disable=None)
        self.validate_pbar = tqdm(total=p.validation_set_size*2, position=1,
                                  desc="    Validating", unit=" experiences",
                                  bar_format=pbar_format, leave=False,
                                  disable=None)

        # Progress display parameters
        self.loss = []
        self.performance = []

    @property
    def eps(self) -> float:
        """ Current value of epsilon (random action probability). """
        e1, e2, t = self.p.eps_start, self.p.eps_end, self.p.eps_anneal_epochs
        return e2 + max(0.0, (e1 - e2) * (t - self.epoch) / t)

    def create_agent(self, greedy: bool = False) -> Agent:
        """ Create the agent that will take actions depending on the player and the agent type """
        return Agent(self.net, eps=(0.0 if greedy else self.eps),
                     policy=self.player_type)

    def update_status(self, loss, perf) -> None:
        """ Update Status Bar with the performance and loss"""
        if not self.status_pbar.disable:
            self.status_pbar.postfix[0] = loss
            self.status_pbar.postfix[1] = perf
            self.loss.append(loss)
            self.performance.append(perf)
            self.status_pbar.refresh()

    def update_train_data(self) -> None:
        """ Replace working set of training data with random new ones. """
        self.train_data = []

    def schedule_and_optimize(self) -> None:
        """ Do the steps required for using schedulers and optimizers in fitting """
        clip_grad_norm_(self.net.parameters(), self.p.max_grad_norm)
        self.Optimizer.step()
        self.Scheduler.step()
        self.Optimizer.zero_grad()

    def cleanup_loss(self, loss)-> Tuple[float, float]:
        """ House keeping to manually garbage collect """
        if isinstance(loss, torch.Tensor):
            l = loss.detach().item()
            del loss
        else:
            l = loss

        torch.cuda.empty_cache()
        return l

    ''' Q LEARNING ONLY '''
    def update_target_net(self) -> None:
        """ Update target net for QLearning """
        if self.epoch % self.p.target_update_freq == 0:
            state_dict = deepcopy(self.net.state_dict())
            self.target_net.load_state_dict(state_dict)

    ''' PI LEARNING ONLY '''
    def process_batch(self, s, a, r, batch, batch_lengths, total_loss=0) -> torch.tensor:
        """ Do the steps for fitting with Gradient Accumulation for PiLearning """
        # sort out loss based on valid actions
        batch = torch.tensor(batch, dtype=int).to(r.device)
        valid = self.prob_cls.valid_action_mask(s)
        log_pi, entropy = self.net(s, a, valid)
        del s, a, valid

        # Compute the loss
        policy_loss = ts.scatter_mean(-(r) * log_pi, batch)[1:]
        entropy_loss = ts.scatter_mean(entropy, batch)[1:]
        entropy_loss *= torch.tensor(np.divide(batch_lengths, sum(batch_lengths)), device=r.device)

        # Calculate the fraction of the total trajectories in this batch (this will be 1 for no gradient accumulation)
        fraction_of_total = len(batch) / self.p.num_play

        loss = fraction_of_total * ((policy_loss).mean() + self.p.entropy_coeff * entropy_loss.mean())
        total_loss += loss
        loss.backward()
        del loss
        return total_loss

    ''' PI LEARNING ONLY '''
    def discount_rewards_in_place(self, r, idx, discount_factor) -> None:
        """ If we are discounting rewards, discount the rewards in place """
        device = r.device
        gammas = torch.full((idx[-1],), discount_factor, device=device)
        for i in range(len(idx) - 1):
            start, end = idx[i], idx[i + 1]
            discounted_gammas = gammas[start:end] ** torch.arange(end - start, device=device)
            r[start:end] *= discounted_gammas

    def play(self, data, pbar: tqdm = None,
             greedy: bool = False,
             memorize: bool = False) -> Union[List,None]:
        """ Running the game """
        envs = data

        agent = self.create_agent(greedy=greedy)
        agent.play(envs, pbar=pbar, batch_size=self.p.batch_size)

        if memorize:
            for e in envs:
                assert e.done
                self.memory.store(e)

            # Sanity check to garbage collect to avoid memory leaks with Tensors (popping should be already taking care of this)
            del self.envs
            self.envs = None
            torch.cuda.empty_cache()
            return None
        else:
            return envs

    def burn_in(self) -> None:
        """ Fill memory before playing QLearning """
        pbar = tqdm(total=self.p.num_burn_in, position=0,
                    desc="Burning in", unit=" networks",
                    bar_format=pbar_format, leave=False,
                    disable=None)
        data = random.choices(self.train_data, k=self.p.num_burn_in)
        self.net.train()
        self.play(data, pbar=pbar, memorize=False, greedy=False)
        pbar.close()

    def rollout(self) -> None:
        """ Play n complete problem drawn from the current set of
            training data using an epsilon-greedy strategy, then
            store completed envs in replay memory. """
        data = random.choices(self.train_data, k=self.p.num_play)
        self.data.eval()
        self.play_pbar.reset()
        self.envs = self.play(data, pbar=self.play_pbar, greedy=False, memorize=True)
        self.play_pbar.clear()

    def fit(self, prev_loss, skip_pi: bool = True) -> Tuple[float, float]:
        """ Perform one optimization step, sampling a batch of transitions
            from replay memory and performing stochastic gradient descent.
        """
        # train network
        self.net.train()

        # train target nets if QLearning
        if self.player_type == 'Q':
            self.target_net.train()

        # get the correct transition and calculate loss based on learning type
        # QNet
        if self.player_type == 'Q':
            # get transition
            s, a, s_next, dr, terminal, dt = self.memory.sample(self.p.batch_size)

            # get best action in next state from POLICY NET
            valid = self.prob_cls.valid_action_mask(s_next)
            agent = self.create_agent(greedy=True)

            a_next, _, _ = agent.get_action(s_next, valid, full_output=True)

            # ..evaluate it with TARGET net
            q_next = self.target_net(s_next, a=a_next)
            q_next[terminal] = 0.0
            q_next = q_next.detach()

            # sort out loss
            # get past q estimate using POLICY net
            q = self.net(s, a=a)

            # update weights of policy net
            self.Optimizer.zero_grad()
            loss = self.loss_func(q, dr + (self.p.df ** dt) * q_next)
            loss.backward()

            self.schedule_and_optimize()

        # PiNet
        elif not skip_pi and (self.player_type == 'Pi'):
            # get transition
            s, a, r, terminal = self.memory.get(
                self.p.batch_size if self.p.gradient_accumulation else self.p.num_play)
            if s == 0:
                return 0, 0

            # Process transitions
            idx = np.concatenate(([0], np.where(np.asarray(terminal))[0] + 1))
            batch_lengths = np.diff(idx)
            batch = np.repeat(np.arange(1, len(batch_lengths) + 1), batch_lengths)

            # Discount the rewards
            if self.p.discounted_rewards < 1.0:
                self.discount_rewards_in_place(r, idx, self.p.discounted_rewards)

            # Gradient accumulation
            if self.p.gradient_accumulation:
                total_loss = 0
                batch = batch.tolist()
                current_traj_idx = 1
                while current_traj_idx < self.p.num_play:
                    total_loss = self.process_batch(player, s, a, r, batch, batch_lengths, total_loss=total_loss)
                    current_traj_idx += min(self.p.batch_size, self.p.num_play - current_traj_idx)
                    torch.cuda.empty_cache()

                self.schedule_and_optimize()

                loss = total_loss
                del total_loss
                torch.cuda.empty_cache()

            # No gradient accumulation
            else:
                loss = self.process_batch(s, a, r, batch, batch_lengths)
                self.schedule_and_optimize()

        # Skip Pi for speed, return previous loss
        else:
            loss = prev_loss

        return self.cleanup_loss(loss)

    def validate(self) -> float:
        """ Validate current policy net.

        Returns:
            Average solution quality over the set of validation data.
            Solution quality is defined by the environment in question.
        """
        self.net.eval()
        self.validate_pbar.reset()
        envs = self.play(self.validation_data,  greedy=True,
                         pbar=self.validate_pbar)
        self.validate_pbar.clear()

        avg = np.mean([e.sol_size for e in envs]).item()

        # garbage collect to avoid memory leaks with Tensors
        del envs
        torch.cuda.empty_cache()

        return avg

    def train(self) -> Tuple[Union[PiNet,QNet], List, List]:
        """ Train the neural net in question """
        p = self.p

        # play games before learning if QLearning
        if self.player_type == 'Q':
            self.burn_in()

        loss = np.nan
        self.train_pbar.refresh()

        while self.epoch < p.max_epochs:
            # update target net if QLearning
            if self.epoch % p.target_update_freq == 0:
                if self.player_type == 'Q':
                    self.update_target_net()

            # refresh playable data with new ones
            if self.epoch % p.train_set_update_freq == 0:
                self.update_train_data()

            # run the training. Because Pi Learning learns from the whole set of experiences during rollout,
            # we need to wipe the memories that have been used already before rolling out new ones. Also we
            # need to skip the fitting for this agent type after every fitting for this agent except when the
            # memories are fresh. That way it only learns once from the experiences for each roll out but
            # we preserve the discrete rolling out and fitting required for the Q learning.
            self.skip_pi = True
            if self.epoch % p.rollout_freq == 0:
                # clear the memory before each rollout for pinet
                if self.player_type == 'Pi':
                    self.memory.clear()
                self.rollout()

                self.skip_pi = False

            # check performance on validation set
            if self.epoch % p.validation_freq == 0:
                perf = self.validate()
                self.update_status(loss, perf)

            # do one fit iteration
            loss = self.fit(loss, skip_pi=self.skip_pi)
            self.train_pbar.update(1)
            self.train_pbar.refresh()

            self.epoch += 1

        return self.net, self.loss, self.performance
