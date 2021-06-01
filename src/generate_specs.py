import argparse
import numpy as np
from torchvision import transforms, datasets
import os
import onnxruntime as rt


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
        f.write(f"(assert (or\n")
        for i in range(n_class):
            if i == y: continue
            f.write(f"\t(and (>= Y_{i} Y_{y}))\n")
        f.write(f"))\n")
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
    return idx


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
    parser.add_argument("--network", type=str, default=None, help="Network to evaluate as .onnx file.")
    parser.add_argument('--mean', nargs='+', type=float, default=0.0, help='the mean used to normalize the data with')
    parser.add_argument('--std', nargs='+', type=float, default=1.0, help='the standard deviation used to normalize the data with')
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

    if args.network is not None:
        sess = rt.InferenceSession(args.network)
        input_name = sess.get_inputs()[0].name

        mean = np.array(args.mean).reshape((1,-1,1,1)).astype(np.float32)
        std = np.array(args.std).reshape((1,-1,1,1)).astype(np.float32)

    idxs = get_sample_idx(args.n, block=args.block, seed=args.seed, n_max=len(dataset), start_idx=args.start_idx)
    spec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../specs", args.dataset)

    i = 0
    ii = 1
    while i<len(idxs):
        idx = idxs[i]
        i += 1
        if args.network is not None:
            x, y = dataset[idx]
            x = x.unsqueeze(0).numpy().astype(np.float32)
            x = (x-mean)/std
            pred_onx = sess.run(None, {input_name: x})[0]
            y_pred = np.argmax(pred_onx, axis=-1)

        if args.network is None or all(y == y_pred):
            write_vnn_spec(dataset, idx, args.epsilon, dir_path=spec_path, prefix=args.dataset + "_spec", data_lb=0, data_ub=1, n_class=10)
        else:
            if len(idxs) < len(dataset): # only sample idxs while there are still new samples to be found
                if args.block: # if we want samples in a block, just get the next one
                    idxs.append(*get_sample_idx(1, True, n_max=len(dataset), start_idx=idxs[-1]+1))
                else: # otherwise sample deterministicly (for given seed) until we find a new sample
                    tmp_idx = get_sample_idx(1, False, seed=args.seed+ii, n_max=len(dataset))
                    ii += 1
                    while tmp_idx in idxs:
                        tmp_idx = get_sample_idx(1, False, seed=args.seed + ii, n_max=len(dataset))
                        ii += 1
                    idxs.append(tmp_idx)
    print(f"{len(idxs)-args.n} samples were misclassified and replacement samples drawn.")

if __name__ == "__main__":
    main()