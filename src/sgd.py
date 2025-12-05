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
    # DIRECTORY (IMPORTANT FIX)
    ############################################################
    opt = "gd"      # <-- MUST be "gd" or "polyak" (NOT "sgd")
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

    ############################################################
    # MODEL + OPTIMIZER
    ############################################################
    torch.manual_seed(seed)
    network = load_architecture(arch_id, dataset).cuda()

    optimizer = torch.optim.SGD(network.parameters(), lr=lr)

    ############################################################
    # STORAGE
    ############################################################
    train_loss = torch.zeros(max_steps)
    test_loss  = torch.zeros(max_steps)
    train_acc  = torch.zeros(max_steps)
    test_acc   = torch.zeros(max_steps)

    eigs = torch.zeros(max_steps // eig_freq if eig_freq > 0 else 0, neigs)
    iterates = torch.zeros(
        max_steps // iterate_freq if iterate_freq > 0 else 0,
        len(parameters_to_vector(network.parameters()))
    )

    ############################################################
    # TRAINING LOOP
    ############################################################
    step = 0
    loader_iter = iter(train_loader)

    while step < max_steps:

        # Compute train + test loss
        train_loss[step], train_acc[step] = compute_losses(
            network, [loss_fn, acc_fn], train_dataset, batch_size
        )
        test_loss[step], test_acc[step] = compute_losses(
            network, [loss_fn, acc_fn], test_dataset, batch_size
        )

        # Hessian eval
        if eig_freq != -1 and step % eig_freq == 0:
            eigs[step // eig_freq, :] = get_hessian_eigenvalues(
                network, loss_fn, abridged_train, neigs=neigs,
                physical_batch_size=batch_size
            )
            print("eigenvalues:", eigs[step // eig_freq, :])

        # Iterates
        if iterate_freq != -1 and step % iterate_freq == 0:
            iterates[step // iterate_freq, :] = parameters_to_vector(
                network.parameters()
            ).detach().cpu()

        # Save partial
        if save_freq != -1 and step % save_freq == 0:
            save_files(directory, [
                ("eigs", eigs[:step // eig_freq]),
                ("iterates", iterates[:step // iterate_freq]),
                ("train_loss", train_loss[:step]),
                ("test_loss", test_loss[:step]),
                ("train_acc", train_acc[:step]),
                ("test_acc", test_acc[:step]),
            ])

        print(f"{step}\t{train_loss[step]:.3f}\t{train_acc[step]:.3f}\t"
              f"{test_loss[step]:.3f}\t{test_acc[step]:.3f}")

        ############################################################
        # SGD UPDATE
        ############################################################
        try:
            X, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            X, y = next(loader_iter)

        X, y = X.cuda(), y.cuda()

        optimizer.zero_grad()
        loss_val = loss_fn(network(X), y)
        loss_val.backward()
        optimizer.step()

        step += 1

    ############################################################
    # SAVE FINAL FILES
    ############################################################
    save_files_final(directory, [
        ("eigs", eigs[:step // eig_freq]),
        ("iterates", iterates[:step // iterate_freq]),
        ("train_loss", train_loss[:step]),
        ("test_loss", test_loss[:step]),
        ("train_acc", train_acc[:step]),
        ("test_acc", test_acc[:step]),
    ])

    if save_model:
        torch.save(network.state_dict(), f"{directory}/snapshot_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGD Training Script.")
    parser.add_argument("dataset", type=str, choices=DATASETS)
    parser.add_argument("arch_id", type=str)
    parser.add_argument("loss", type=str, choices=["ce", "mse"])
    parser.add_argument("lr", type=float)
    parser.add_argument("max_steps", type=int)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neigs", type=int, default=0)
    parser.add_argument("--eig_freq", type=int, default=-1)
    parser.add_argument("--iterate_freq", type=int, default=-1)
    parser.add_argument("--save_freq", type=int, default=-1)
    parser.add_argument("--abridged_size", type=int, default=5000)
    parser.add_argument("--save_model", type=bool, default=False)

    args = parser.parse_args()

    main(
        dataset=args.dataset,
        arch_id=args.arch_id,
        loss=args.loss,
        lr=args.lr,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        neigs=args.neigs,
        eig_freq=args.eig_freq,
        iterate_freq=args.iterate_freq,
        save_freq=args.save_freq,
        save_model=args.save_model,
        seed=args.seed,
        abridged_size=args.abridged_size
    )
