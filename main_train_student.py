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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def parse_args():
    parser = ArgumentParser(description="KD")
    parser.add_argument("--data_name", type=str, default="ml1m")
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--is_logging', type=bool, default=False)
    # Seed
    parser.add_argument('--seed', type=int, default=2023, help="Seed (For reproducability)")
    # Model
    parser.add_argument("--model_name", type=str, default="MF")
    parser.add_argument('--dim', type=int, default=32, help="Dimension for embedding")
    parser.add_argument('--alpha', type=float, default=0.7, help="trade-off")
    parser.add_argument('--temp', type=float, default=1.0, help="temperature")
    # Different methods
    parser.add_argument("--mode", type=str, default="pure_KD", choices=['pure_GT', 'pure_KD', 'normal'])
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


class Dataset_origin(Dataset):
    def __init__(self, u_id, i_id, rating):
        self.u_id = u_id
        self.i_id = i_id
        self.rating = rating

    def __getitem__(self, index):
        return self.u_id[index], self.i_id[index], self.rating[index]

    def __len__(self):
        return len(self.rating)


class Dataset_teacher_augment(Dataset):
    def __init__(self, u_id, i_id, rating, teacher_logits):
        self.u_id = u_id
        self.i_id = i_id
        self.rating = rating
        self.teacher_logits = teacher_logits

    def __getitem__(self, index):
        return self.u_id[index], self.i_id[index], self.rating[index], self.teacher_logits[index]

    def __len__(self):
        return len(self.rating)


def train(args):
    path = 'data_raw/{}/'.format(args.data_name)
    if args.data_name == 'ml100k':
        df = pd.read_csv(path + 'u.data', header=None, delimiter='\t')
    elif args.data_name == 'ml1m':
        df = pd.read_csv(path + 'ratings.dat', header=None, delimiter='::')
    x, y = df.iloc[:, :2], df.iloc[:, 2]

    # {4, 5} --> 1, {1, 2, 3} --> 0
    y[y <= 3] = 0
    y[y > 3] = 1
    # print(y.to_numpy())

    # print(y)
    x_teacher, x_student, y_teacher, y_student = train_test_split(x, y, test_size=0.3, random_state=args.seed)
    x_train, x_val_test, y_train, y_val_test = train_test_split(x_student, y_student, test_size=0.2, random_state=args.seed)
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=args.seed)

    print("x_train shape:{}, y_train shape{}".format(x_train.shape, y_train.shape))
    print("x_val shape:{}, y_val shape{}".format(x_val.shape, y_val.shape))
    print("x_test shape:{}, y_test shape{}".format(x_test.shape, y_test.shape))

    # load teacher logits_MF
    user_emb_teacher = torch.load('./saved/{}/emb_{}/teacher_dim64_user.pt'.format(args.data_name, args.model_name))
    item_emb_teacher = torch.load('./saved/{}/emb_{}/teacher_dim64_item.pt'.format(args.data_name, args.model_name))

    teacher_logits_train = user_emb_teacher[x_train[0].tolist()] * item_emb_teacher[x_train[1].tolist()]
    teacher_logits_train = np.array(teacher_logits_train.sum(1))
    print("teacher_logits_train:", teacher_logits_train.shape)

    train_dataset = Dataset_teacher_augment(np.array(x_train[0]), np.array(x_train[1]),
                                              np.array(y_train).astype(np.float32),
                                              teacher_logits_train.astype(np.float32))
    val_dataset = Dataset_origin(np.array(x_val[0]), np.array(x_val[1]),
                            np.array(y_val).astype(np.float32))
    test_dataset = Dataset_origin(np.array(x_test[0]), np.array(x_test[1]),
                             np.array(y_test).astype(np.float32))
    # construct dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    num_users, num_items = max(df[0]) + 1, max(df[1]) + 1
    model = model_MF(num_users, num_items, args.dim).to(args.device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_func = torch.nn.BCELoss().to(args.device)

    saved_path = './saved/{}/model_MF/student_dim{}_{}.pt'.format(args.data_name, args.dim, args.mode)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=saved_path)

    for epoch in range(args.n_epochs):
        model.train()
        total_loss, total_len = 0, 0
        for x_u, x_i, y, y_t in train_dataloader:
            x_u, x_i, y, y_t = x_u.to(args.device), x_i.to(args.device), y.to(args.device), y_t.to(args.device)
            # print("x_u:", x_u[:5])
            y_pre = model(x_u, x_i)
            if args.mode == 'pure_GT':
                loss = loss_func(torch.sigmoid(y_pre), y)
            elif args.mode == 'pure_KD':
                loss = args.temp ** 2 * loss_func(torch.sigmoid(y_pre / args.temp), torch.sigmoid(y_t / args.temp))
            elif args.mode == 'normal':
                loss = args.alpha * loss_func(torch.sigmoid(y_pre), y) + \
                       (1 - args.alpha) * args.temp ** 2 * loss_func(torch.sigmoid(y_pre / args.temp),
                                                                     torch.sigmoid(y_t / args.temp))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            total_len += len(y)
        train_loss = total_loss / total_len
        print("epoch {}, train loss is {:.6f}".format(epoch, train_loss))

        model.eval()
        labels, predicts = [], []
        with torch.no_grad():
            for x_u, x_i, y in val_dataloader:
                x_u, x_i, y = x_u.to(args.device), x_i.to(args.device), y.to(args.device)
                y_pre = model(x_u, x_i)
                labels.extend(y.tolist())
                predicts.extend(y_pre.tolist())
            valid_acc = binary_accuracy_metric(torch.tensor(predicts), torch.tensor(labels)).item()

            early_stopping(valid_acc, model)

            if early_stopping.early_stop:
                print("Early stopping")

                break

    print("mode:", args.mode)
    model.load_state_dict(torch.load(saved_path))
    predicts, GT_labels = [], []
    for x_u, x_i, y, _ in train_dataloader:
        x_u, x_i, y, _ = x_u.to(args.device), x_i.to(args.device), y.to(args.device), _
        y_pre = model(x_u, x_i)
        predicts.extend(y_pre.tolist())
        GT_labels.extend(y.tolist())
    np.savetxt('./saved/{}/logits_MF/student_dim{}_{}_train.txt'.format(args.data_name, args.dim, args.mode), np.array(predicts), delimiter=',')
    train_accuracy = binary_accuracy_metric(torch.tensor(predicts), torch.tensor(GT_labels)).item()
    print("train accuracy:", train_accuracy)

    predicts, GT_labels = [], []
    for x_u, x_i, y in test_dataloader:
        x_u, x_i, y = x_u.to(args.device), x_i.to(args.device), y.to(args.device)
        y_pre = model(x_u, x_i)
        predicts.extend(y_pre.tolist())
        GT_labels.extend(y.tolist())
    np.savetxt('./saved/{}/logits_MF/student_dim{}_{}_test.txt'.format(args.data_name, args.dim, args.mode), np.array(predicts), delimiter=',')
    test_accuracy = binary_accuracy_metric(torch.tensor(predicts), torch.tensor(GT_labels)).item()
    print("test accuracy:", test_accuracy)
    return train_accuracy, test_accuracy




if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print("device:", args.device)
    train_accuracy_list, test_accuracy_list = [], []
    for i in range(1):
        print("Round:{}".format(i))
        train_accuracy, test_accuracy = train(args)
        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)
        print("------------------------------")

    print("train accuracy:", np.mean(train_accuracy_list))
    print("test accuracy:", np.mean(test_accuracy_list))
