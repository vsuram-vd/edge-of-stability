import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler

# Import from repo
from datasets import get_dataset
from archs import build_architecture
from hessian import compute_top_eigenvalues  # same function used in gd.py
from utils import AverageMeter, accuracy, set_seed


############################################################
#                  SGD TRAINING LOOP
############################################################

def train_sgd(
    dataset_name,
    arch_name,
    loss_name,
    lr,
    batch_size,
    max_steps,
    neigs,
    eig_freq,
    seed,
    device="cuda"
):

    set_seed(seed)

    print(f"\n=== SGD TRAINING ===")
    print(f"Dataset: {dataset_name}")
    print(f"Arch: {arch_name}")
    print(f"Loss: {loss_name}")
    print(f"Learning rate: {lr}")
    print(f"Batch size: {batch_size}")
    print(f"Max steps: {max_steps}")
    print(f"Eigen freq: {eig_freq}\n")

    ############################################################
    #                 LOAD DATA
    ############################################################
    train_data, test_data, num_classes = get_dataset(dataset_name)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=RandomSampler(train_data),
        drop_last=True
    )

    ############################################################
    #                 BUILD MODEL
    ############################################################
    model = build_architecture(arch_name, num_classes=num_classes).to(device)

    if loss_name == "mse":
        criterion = nn.MSELoss()
    elif loss_name == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Unsupported loss type")

    optimizer = optim.SGD(model.parameters(), lr=lr)

    ############################################################
    #       TRAINING METRICS + STORAGE
    ############################################################
    losses = []
    accs = []
    eig_history = []

    step = 0
    model.train()

    ############################################################
    #                 TRAINING LOOP
    ############################################################
    while step < max_steps:
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            # record
            losses.append(loss.item())

            # compute accuracy
            if loss_name == "ce":
                accs.append(accuracy(out, yb))
            else:
                accs.append(0)  # placeholder for MSE

            # Hessian eigenvalue measurement
            if neigs > 0 and step % eig_freq == 0:
                print(f"[Step {step}] Computing top-{neigs} Hessian eigenvalues...")
                eigvals = compute_top_eigenvalues(
                    model=model,
                    dataset=train_data,
                    criterion=criterion,
                    device=device,
                    top_k=neigs
                )
                eig_history.append(eigvals)

            step += 1
            if step >= max_steps:
                break

    ############################################################
    #                SAVE RESULTS 
    ############################################################
    base_dir = os.environ.get("RESULTS", "./results")
    out_dir = os.path.join(
        base_dir,
        dataset_name,
        arch_name,
        "sgd",
        f"lr_{lr}_bs_{batch_size}_seed_{seed}"
    )

    os.makedirs(out_dir, exist_ok=True)

    torch.save({
        "losses": torch.tensor(losses),
        "accs": torch.tensor(accs),
        "eigvals": torch.tensor(eig_history),
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed
    }, os.path.join(out_dir, "results.pt"))

    print(f"\n=== Saved results to: {out_dir} ===\n")



############################################################
#                  MAIN SCRIPT ENTRY
############################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", type=str)
    parser.add_argument("arch", type=str)
    parser.add_argument("loss", type=str)

    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_steps", type=int, default=20000)

    parser.add_argument("--neigs", type=int, default=1)
    parser.add_argument("--eig_freq", type=int, default=200)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    train_sgd(
        dataset_name=args.dataset,
        arch_name=args.arch,
        loss_name=args.loss,
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        neigs=args.neigs,
        eig_freq=args.eig_freq,
        seed=args.seed,
        device=args.device
    )
