from os import makedirs

import torch
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, RandomSampler

import argparse

from archs import load_architecture
from utilities import (
    get_loss_and_acc,
    compute_losses,
    save_files,
    save_files_final,
    get_hessian_eigenvalues,
    iterate_dataset,
    get_gd_directory
)
from data import load_dataset, take_first, DATASETS


def main(dataset: str, arch_id: str, loss: str, lr: float, max_steps: int,
         batch_size: int = 128, neigs: int = 0,
         eig_freq: int = -1, iterate_freq: int = -1, save_freq: int = -1,
         save_model: bool = False, seed: int = 0, abridged_size: int = 5000):

    ############################################################
    # DIRECTORY (IMPORTANT)
    ############################################################
    opt = "gd"     # SGD stored under gd directory
    beta = 0.0

    directory = get_gd_directory(dataset, lr, arch_id, seed, opt, loss, beta)
    print(f"output directory: {directory}")
    makedirs(directory, exist_ok=True)

    ############################################################
    # DATASET
    ############################################################
    train_dataset, test_dataset = load_dataset(dataset, loss)
    abridged_train = take_first(train_dataset, abridged_size)

    loss_fn, acc_fn = get_loss_and_acc(loss)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=RandomSampler(train_dataset),
        drop_last=True,
    )

    #############################################
