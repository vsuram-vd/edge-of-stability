
from os import makedirs
import os

import torch
from torch.nn.utils import parameters_to_vector
import argparse

from archs import load_architecture
from utilities import (
    get_loss_and_acc,
    compute_losses,
    save_files,
    save_files_final,
    get_hessian_eigenvalues,
)
from data import load_dataset, take_first, DATASETS


def get_sgd_directory(dataset: str, lr: float, arch_id: str, seed: int,
                      loss: str, batch_size: int) -> str:
    base = os.environ.get("RESULTS", ".")
    return f"{base}/{dataset}/{arch_id}/seed_{seed}/{loss}/sgd/lr_{lr}_batch_{batch_size}"


def main(dataset: str, arch_id: str, loss: str, lr: float, max_steps: int,
         batch_size: int = 128, neigs: int = 0,
         physical_batch_size: int = 1000, eig_freq: int = -1, iterate_freq: int = -1,
         save_freq: int = -1, save_model: bool = False,
         loss_goal: float = None, acc_goal: float = None,
         abridged_size: int = 5000, seed: int = 0, nproj: int = 0):

    directory = get_sgd_directory(dataset, lr, arch_id, seed, loss, batch_size)
    print(f"output directory: {directory}")
    makedirs(directory, exist_ok=True)

   
    train_dataset, test_dataset = load_dataset(dataset, loss)
    abridged_train = take_first(train_dataset, abridged_size)

    loss_fn, acc_fn = get_loss_and_acc(loss)

    torch.manual_seed(seed)
    network = load_architecture(arch_id, dataset).cuda()

    torch.manual_seed(7)
    projectors = torch.randn(nproj, len(parameters_to_vector(network.parameters())))

    optimizer = torch.optim.SGD(network.parameters(), lr=lr)

    train_loss = torch.zeros(max_steps)
    test_loss = torch.zeros(max_steps)
    train_acc = torch.zeros(max_steps)
    test_acc = torch.zeros(max_steps)
    iterates = torch.zeros(max_steps // iterate_freq if iterate_freq > 0 else 0, len(projectors))
    eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)

    num_train = len(train_dataset)

    for step in range(0, max_steps):
        train_loss[step], train_acc[step] = compute_losses(
            network, [loss_fn, acc_fn], train_dataset, physical_batch_size
        )
        test_loss[step], test_acc[step] = compute_losses(
            network, [loss_fn, acc_fn], test_dataset, physical_batch_size
        )

        if eig_freq != -1 and step % eig_freq == 0:
            eigs[step // eig_freq, :] = get_hessian_eigenvalues(
                network, loss_fn, abridged_train, neigs=neigs,
                physical_batch_size=physical_batch_size
            )
            print("eigenvalues: ", eigs[step // eig_freq, :])

        if iterate_freq != -1 and step % iterate_freq == 0:
            iterates[step // iterate_freq, :] = projectors.mv(
                parameters_to_vector(network.parameters()).cpu().detach()
            )

        if save_freq != -1 and step % save_freq == 0:
            save_files(
                directory,
                [
                    ("eigs", eigs[: step // eig_freq]),
                    ("iterates", iterates[: step // iterate_freq]),
                    ("train_loss", train_loss[:step]),
                    ("test_loss", test_loss[:step]),
                    ("train_acc", train_acc[:step]),
                    ("test_acc", test_acc[:step]),
                ],
            )

        print(
            f"{step}\t{train_loss[step]:.3f}\t{train_acc[step]:.3f}\t"
            f"{test_loss[step]:.3f}\t{test_acc[step]:.3f}"
        )

        # stopping criteria 
        if (loss_goal is not None and train_loss[step] < loss_goal) or (
            acc_goal is not None and train_acc[step] > acc_goal
        ):
            break

       
        optimizer.zero_grad()
        idx = torch.randint(num_train, (batch_size,))
        xs, ys = [], []
        for i in idx:
            x_i, y_i = train_dataset[i]
            xs.append(x_i)
            ys.append(y_i)
        X = torch.stack(xs).cuda()
        y = torch.stack(ys).cuda()

        step_loss = loss_fn(network(X), y) / batch_size
        step_loss.backward()
        optimizer.step()

   
    save_files_final(
        directory,
        [
            ("eigs", eigs[: (step + 1) // eig_freq] if eig_freq != -1 else eigs),
            ("iterates", iterates[: (step + 1) // iterate_freq] if iterate_freq != -1 else iterates),
            ("train_loss", train_loss[: step + 1]),
            ("test_loss", test_loss[: step + 1]),
            ("train_acc", train_acc[: step + 1]),
            ("test_acc", test_acc[: step + 1]),
        ],
    )
    if save_model:
        torch.save(network.state_dict(), f"{directory}/snapshot_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train using stochastic gradient descent (SGD)."
    )
    parser.add_argument("dataset", type=str, choices=DATASETS,
                        help="which dataset to train")
    parser.add_argument("arch_id", type=str,
                        help="which network architecture to train")
    parser.add_argument("loss", type=str, choices=["ce", "mse"],
                        help="which loss function to use")
    parser.add_argument("lr", type=float,
                        help="the learning rate")
    parser.add_argument("max_steps", type=int,
                        help="the maximum number of SGD steps to train for")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="SGD mini-batch size")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed")
    parser.add_argument("--physical_batch_size", type=int, default=1000,
                        help="max examples for metrics/Hessian computations")
    parser.add_argument("--acc_goal", type=float,
                        help="stop if train accuracy crosses this value")
    parser.add_argument("--loss_goal", type=float,
                        help="stop if train loss crosses this value")
    parser.add_argument("--neigs", type=int, default=0,
                        help="number of top eigenvalues to compute")
    parser.add_argument("--eig_freq", type=int, default=-1,
                        help="frequency for Hessian eigenvalues (-1 = never)")
    parser.add_argument("--nproj", type=int, default=0,
                        help="dimension of random projections")
    parser.add_argument("--iterate_freq", type=int, default=-1,
                        help="frequency for saving random projections")
    parser.add_argument("--abridged_size", type=int, default=5000,
                        help="size of abridged dataset for Hessian eigenvalues")
    parser.add_argument("--save_freq", type=int, default=-1,
                        help="frequency for saving intermediate results")
    parser.add_argument("--save_model", type=bool, default=False,
                        help="if true, save model at end of training")

    args = parser.parse_args()
    main(**vars(args))

