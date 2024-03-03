from dataclasses import dataclass
from simple_parsing import choice, Serializable

__all__ = []
__all__.extend([
    'NetParams',
    'ReplayParams',
    'TrainParams',
    'TestParams',
    'HyperParams'
])


@dataclass
class NetParams(Serializable):
    # number of BP iterations
    depth: int

    # size of embedding
    embed_dim: int

    # how to pool over neighbors ('sum' or 'mean')
    graph_agg: str = choice('sum', 'mean', default='sum')

    # how to pool all nodes' states into graph state
    nbr_agg: str = choice('sum', 'mean', default='sum')


@dataclass
class ReplayParams(Serializable):
    # maximum number of Q transitions to store
    capacity: int

    # the "N" in "N-step replay memory"
    step_diff: int


@dataclass
class TrainParams(Serializable):
    # Graph Size to Seed from
    seed_n_low: int
    seed_n_high: int

    # minibatch size
    batch_size: int

    # maximum number of training iterations to perform
    max_epochs: int

    # play through/store training examples every this many epochs
    rollout_freq: int

    # number of training examples to play each time
    num_play: int

    # discount factor
    df: float

    # rate at which to refresh training graphs
    train_set_update_freq: int

    # number of graphs in training set
    train_set_size: int

    # initial random move probability for greedy epsilon strategy
    eps_start: float

    # final random move probability for greedy epsilon strategy
    eps_end: float

    # number of epochs over which to anneal eps from eps_start to eps_end
    eps_anneal_epochs: float

    # rate at which to "hard update" target net
    target_update_freq: int

    # number of complete environments to play through/store before training
    num_burn_in: int

    # pytorch optimizer
    optimizer: str

    # number of graphs to generate for validation
    validation_set_size: int

    # perform validation every this manny epochs
    validation_freq: int

    # coefficient for modulated entropy
    entropy_coeff: float

    # clip gradient norm each players' parameters to this
    max_grad_norm: float

    # allow for gradient accumulation if True
    gradient_accumulation: bool

    # options for optimizer
    optimizer_kwargs: dict = None

    # whether to plot the performance after completion
    plotting: bool = False

@dataclass
class TestParams(Serializable):
    # minibatch size
    batch_size: int

    # maximum number of experiment iterations to perform
    max_experiments: int

    # number of graphs in testing set
    test_set_size: int

    # whether to plot the performance after completion
    plotting: bool = False

@dataclass
class HyperParams(Serializable):
    """ Parameters representing a complete training run """
    prob: str
    device: str
    net: NetParams
    replay: ReplayParams
    training: TrainParams
    testing: TestParams
    prob_kwargs: dict = None

