import time
import logging
import argparse
from termcolor import cprint
import scanpy as sc
from train_model_sample import train_scMPCL

def get_args_key(args):
    return "-".join([args.dataset_name])

def get_args(dataset_name, dataset_class, log_path , train_path, seed) -> argparse.Namespace:
    # yaml_path = yaml_path or os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")
    parser = argparse.ArgumentParser(description='Parser for Simple Unsupervised Graph Representation Learning')
    # Basics
    parser.add_argument("--seed", default=seed)
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 32)')
    parser.add_argument('--log_path', type=str, default=log_path)

    # Dataset
    parser.add_argument("--dataset_class", default=dataset_class,
                        help='labels for data holding classified information')
    parser.add_argument("--dataset_name", default=dataset_name)
    # Train
    parser.add_argument('--train_epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--train_lr', default=0.0025, type=float,
                        help='initial learning rate of train')
    parser.add_argument('--train_path', type = str, default=train_path,
                        help='Save the model after training has finished. If no file '
                             'is available run pre-training again.')
    parser.add_argument('--batch_size', default=256, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--T', default=0.07, type=float, help='softmax temperature (default: 0.07)')
    parser.add_argument('--eval_freq', default=50, type=int, metavar='N',
                        help='Save frequency (default: 10)')

    parser.add_argument("--hidden_dim", default=128, type=int, nargs="+")
    parser.add_argument("--clu_cfg", default=[128, 64], type=int, nargs="+")
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--weight_decay", default=0.0001, type=float)

    args = parser.parse_args()
    return args

def pprint_args(_args: argparse.Namespace):
    cprint("Args PPRINT: {}".format(get_args_key(_args)), "yellow")
    for k, v in sorted(_args.__dict__.items()):
        print("\t- {}: {}".format(k, v))


if __name__ == '__main__':

    dataset_name = "Klein"
    dataset_path = "dataset/"+dataset_name+"_preprocessed.h5ad"
    dataset_class = "Group"
    log_path = "logs/"+dataset_name+"_preprocessed_train.txt"

    cprint("## Loading Dataset ##", "yellow")
    # loading data
    adata = sc.read_h5ad(dataset_path)
    print(adata)

    for seed in [0, 2, 4, 8, 16]:
        # path to store the model
        train_path = "logs/"+dataset_name+"_preprocessed_" + str(seed) + "_train_model.pth"
        main_args = get_args(
            # information of dataset
            dataset_name=dataset_name, dataset_class=dataset_class,
            log_path=log_path, train_path=train_path,
            # random seed
            seed=seed )
        pprint_args(main_args)

        logging.basicConfig(level=logging.INFO,
                            filename=log_path,
                            filemode='w',
                            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

        t0 = time.perf_counter()
        train_scMPCL(main_args, adata)
        cprint("Done!")
