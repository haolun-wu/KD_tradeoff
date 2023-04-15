import numpy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import numpy as np
from model.MF import model_MF
from argparse import ArgumentParser
from utils import EarlyStopping
from torchmetrics.classification import BinaryAccuracy

binary_accuracy_metric = BinaryAccuracy()


def parse_args():
    parser = ArgumentParser(description="KD_MF")
    parser.add_argument("--data_name", type=str, default="ml100k")
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--is_logging', type=bool, default=False)
    # Seed
    parser.add_argument('--seed', type=int, default=2023, help="Seed (For reproducability)")
    # Model
    parser.add_argument('--dim', type=int, default=128, help="Dimension for embedding")
    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--wd', type=float, default=1e-5, help="Weight decay factor")
    # Training
    parser.add_argument('--n_epochs', type=int, default=1000, help="Number of epoch during training")
    parser.add_argument('--every', type=int, default=1,
                        help="Period for evaluating precision and recall during training")
    parser.add_argument('--patience', type=int, default=20, help="patience")
    parser.add_argument('--batch_size', type=int, default=1024, help="batch_size")

    return parser.parse_args()


class MfDataset(Dataset):
    def __init__(self, u_id, i_id, rating):
        self.u_id = u_id
        self.i_id = i_id
        self.rating = rating

    def __getitem__(self, index):
        return self.u_id[index], self.i_id[index], self.rating[index]

    def __len__(self):
        return len(self.rating)


def train_test_accuray(gt_label_train, gt_label_test, model_logits_path):
    model_logits_train = np.loadtxt("{}_train.txt".format(model_logits_path))
    model_logits_test = np.loadtxt("{}_test.txt".format(model_logits_path))

    train_accuracy = binary_accuracy_metric(torch.tensor(model_logits_train), torch.tensor(gt_label_train)).item()
    test_accuracy = binary_accuracy_metric(torch.tensor(model_logits_test), torch.tensor(gt_label_test)).item()
    return train_accuracy, test_accuracy


def eval_all_model(args):
    gt_label_train = np.loadtxt('saved/{}/gt_label_train.txt'.format(args.data_name))
    gt_label_test = np.loadtxt('saved/{}/gt_label_test.txt'.format(args.data_name))

    saved_model_dict = {
        'teacher': 'saved/{}/logits_MF/teacher_dim128'.format(args.data_name),
        'student_pure_GT': 'saved/{}/logits_MF/student_dim4_pure_GT'.format(args.data_name),
        'student_pure_KD': 'saved/{}/logits_MF/student_dim4_pure_KD'.format(args.data_name),
        'student_normal': 'saved/{}/logits_MF/student_dim4_normal'.format(args.data_name),
    }

    for model_name in saved_model_dict:
        train_acc, test_acc = train_test_accuray(gt_label_train, gt_label_test, saved_model_dict[model_name])
        print(model_name)
        print("train_acc: {:.4f}, test_acc: {:.4f}".format(train_acc, test_acc))


if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print("device:", args.device)

    eval_all_model(args)
