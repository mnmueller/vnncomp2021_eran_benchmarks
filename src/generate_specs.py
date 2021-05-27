import argparse
import numpy as np
from torchvision import transforms, datasets
import os


def write_vnn_spec(dataset, index, eps, dir_path="./", prefix="spec", data_lb=0, data_ub=1, n_class=10):
    x, y = dataset[index]
    x = np.array(x).reshape(-1)
    x_lb = np.clip(x - eps, data_lb, data_ub)
    x_ub = np.clip(x + eps, data_lb, data_ub)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    spec_path = os.path.join(dir_path, f"{prefix}_idx_{index}_eps_{eps:.5f}.vnnlib")

    with open(spec_path, "w") as f:
        f.write(f"; Spec for sample id {index} and epsilon {eps:.5f}\n")

        f.write(f"\n; Definition of input variables\n")
        for i in range(len(x_ub)):
            f.write(f"(declare-const X_{i} Real)\n")

        f.write(f"\n; Definition of output variables\n")
        for i in range(n_class):
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write(f"\n; Definition of input constraints\n")
        for i in range(len(x_ub)):
            f.write(f"(assert (<= X_{i} {x_ub[i]:.8f}))\n")
            f.write(f"(assert (>= X_{i} {x_lb[i]:.8f}))\n")

        f.write(f"\n; Definition of output constraints\n")
        for i in range(n_class):
            if i == y: continue
            f.write(f"(assert (>= Y_{i} Y_{y}))\n")
    return spec_path


def get_sample_idx(n, block=True, seed=42, n_max=10000, start_idx=None):
    np.random.seed(seed)
    assert n <= n_max, f"only {n_max} samples are available"
    if block:
        if start_idx is None:
            start_idx = np.random.choice(n_max,1,replace=False)
        else:
            start_idx = start_idx % n_max
        idx = list(np.arange(start_idx,min(start_idx+n,n_max)))
        idx += list(np.arange(0,n-len(idx)))
    else:
        idx = list(np.random.choice(n_max,n,replace=False))
    return np.array(idx)


def get_cifar10():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
    return datasets.CIFAR10(data_path, train=False, download=True, transform=transforms.ToTensor())


def get_mnist():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
    return datasets.MNIST(data_path, train=False, download=True, transform=transforms.ToTensor())


def main():
    parser = argparse.ArgumentParser(description='VNN spec generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, required=True, choices=["mnist", "cifar10"],
                        help='The dataset to generate specs for')
    parser.add_argument('--epsilon', type=float, required=True, help='The epsilon for L_infinity perturbation')
    parser.add_argument('--n', type=int, default=25, help='The number of specs to generate')
    parser.add_argument('--block', action="store_true", default=False, help='Generate specs in a block')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for idx generation')
    parser.add_argument('--start_idx', type=int, default=None, help='Enforce block mode and return deterministic indices')
    args = parser.parse_args()

    if args.start_idx is not None:
        args.block = True
        print(f"Producing deterministic {args.n} indices starting from {args.start_idx}.")

    if args.dataset == "mnist":
        dataset = get_mnist()
    elif args.dataset == "cifar10":
        dataset = get_cifar10()
    else:
        assert False, "Unkown dataset" # Should be unreachable

    idxs = get_sample_idx(args.n, block=args.block, seed=args.seed, n_max=len(dataset), start_idx=args.start_idx)
    spec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../specs", args.dataset)
    for idx in idxs:
        write_vnn_spec(dataset, idx, args.epsilon, dir_path=spec_path, prefix=args.dataset + "_spec", data_lb=0, data_ub=1, n_class=10)


if __name__ == "__main__":
    main()