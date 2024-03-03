from dataclasses import asdict
from dataclasses import dataclass
import os
from pathlib import Path

import matplotlib.pyplot as plt

import torch
from simple_parsing import field, ArgumentParser

from co2.trainer import Trainer
from co2.util import set_seed, init_weights
import shutil

# may improve speed of certain cuda operations
torch.backends.cudnn.benchmark = True


@dataclass
class Options:
    """ options """
    # .yml file containing HyperParams
    config_file: str = field(alias='-c', required=True)

    # where to save training results
    output_file: str = field(alias='-o', required=True)

    # device to train on
    device: str = 'cpu'

    # random seed
    seed: int = field(alias='-s', default=None, required=False)


def main(config_file: str, output_file: str, device: str = 'cpu', seed: int = None):

    if device != 'cpu':
        torch.cuda.set_per_process_memory_fraction(0.98)

    if seed is not None:
        set_seed(seed)

    # File locations and parameters
    hp = HyperParams.load(Path(os.getcwd() + config_file))

    # Problem setup, graph generation and memory
    # prob_cls = GraphLearningProblem.get_prob_cls(hp.prob)

    # Trainer Setup and training of the net
    init_weights(net)
    trainer = Trainer(net, prob_cls, hp.training, device=device,
                      prob_kwargs=hp.prob_kwargs)
    net, losses, performance = trainer.train()

    # Saving the nets
    file_name = ""
    file_path = os.path.join(output_file, 'nets', f"{file_name}.pt")
    config_file_path = os.path.join(os.getcwd(), config_file[1:])
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    net.save(file_path)
    shutil.copyfile(config_file_path, f"{os.path.dirname(file_path)}/{file_name}.yml")

    # Plot the test results
    if hp.training.plotting:
        from netrl.util import set_mpl
        set_mpl()
        xlist = [(i + 1) * hp.training.validation_freq for i in range(len(losses[0]) - 1)]
        fig1, (ax1, ax2, ax3) = plt.subplots(3)
        ax1.plot(xlist, losses[0][1:], 'bo', linestyle="dashed", label='Player 1')
        ax1.set_xlabel("Epochs", fontsize=16)
        ax1.set_ylabel("Loss", fontsize=16)
        ax1.legend(loc='best')
        ax2.plot(xlist, losses[1][1:], 'ro', linestyle="dashed", label='Player 2')
        ax2.set_xlabel("Epochs", fontsize=16)
        ax2.set_ylabel("Loss", fontsize=16)
        ax2.legend(loc='best')
        ax3.plot(xlist, performance[1:], 'go', linestyle="dashed")
        ax3.set_xlabel("Epochs", fontsize=16)
        ax3.set_ylabel("Performance", fontsize=16)
        plt.show()

if __name__ == "__main__":
    parser = ArgumentParser(add_dest_to_option_strings=False,
                            add_option_string_dash_variants=True)
    parser.add_arguments(Options, "options")
    args = parser.parse_args()

    main(**asdict(args.options))
