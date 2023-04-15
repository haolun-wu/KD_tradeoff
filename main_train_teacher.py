import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import numpy as np
from model.MF import model_MF
from model.FM import model_FM
from argparse import ArgumentParser
from utils import EarlyStopping
from torchmetrics.classification import BinaryAccuracy

binary_accuracy_metric = BinaryAccuracy()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def parse_args():
    parser = ArgumentParser(description="KD")
    parser.add_argument("--data_name", type=str, default="ml1m")

    # parser.add_argument('--val_ratio', type=float, default=0.1)
    # parser.add_argument('--test_ratio', type=float, default=0.3)
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--is_logging', type=bool, default=False)
    # Seed
    parser.add_argument('--seed', type=int, default=2023, help="Seed (For reproducability)")
    # Model
    parser.add_argument("--model_name", type=str, default="MF")
    parser.add_argument('--dim', type=int, default=64, help="Dimension for embedding")
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
    x_train, x_val_test, y_train, y_val_test = train_test_split(x_teacher, y_teacher, test_size=0.2, random_state=args.seed)
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=args.seed)

    np.savetxt('./saved/{}/gt_label_train.txt'.format(args.data_name), np.array(y_train.to_numpy()), delimiter=',')
    np.savetxt('./saved/{}/gt_label_val.txt'.format(args.data_name), np.array(y_val.to_numpy()), delimiter=',')
    np.savetxt('./saved/{}/gt_label_test.txt'.format(args.data_name), np.array(y_test.to_numpy()), delimiter=',')

    print("x_train shape:{}, y_train shape{}".format(x_train.shape, y_train.shape))
    print("x_val shape:{}, y_val shape{}".format(x_val.shape, y_val.shape))
    print("x_test shape:{}, y_test shape{}".format(x_test.shape, y_test.shape))

    # 需要将数据全部转化为np.array, 否则后面的dataloader会报错， pytorch与numpy之间转换较好，与pandas转化容易出错
    train_dataset = Dataset_origin(np.array(x_train[0]), np.array(x_train[1]),
                                   np.array(y_train).astype(np.float32))  # 将标签设为np.float32类型， 否则会报错
    val_dataset = Dataset_origin(np.array(x_val[0]), np.array(x_val[1]),
                                 np.array(y_val).astype(np.float32))  # 将标签设为np.float32类型， 否则会报错
    test_dataset = Dataset_origin(np.array(x_test[0]), np.array(x_test[1]),
                                  np.array(y_test).astype(np.float32))  # 将标签设为np.float32类型， 否则会报错
    # construct dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    num_users, num_items = max(df[0]) + 1, max(df[1]) + 1
    if args.model_name == 'MF':
        model = model_MF(num_users, num_items, args.dim).to(args.device)
    # elif args.model_name == 'FM':
    #     model = model_FM(num_users, num_items, args.dim).to(args.device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_func = torch.nn.BCELoss().to(args.device)

    saved_path = './saved/{}/model_{}/teacher_dim{}.pt'.format(args.data_name, args.model_name, args.dim)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=saved_path)

    for epoch in range(args.n_epochs):
        model.train()
        total_loss, total_len = 0, 0
        for x_u, x_i, y in train_dataloader:
            x_u, x_i, y = x_u.to(args.device), x_i.to(args.device), y.to(args.device)
            # print("x_u:", x_u[:5])
            y_pre = model(x_u, x_i)
            loss = loss_func(torch.sigmoid(y_pre), y)
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
        # with torch.no_grad():
        #     total_loss_valid, total_len_valid = 0, 0
        #     for x_u, x_i, y in val_dataloader:
        #         x_u, x_i, y = x_u.to(args.device), x_i.to(args.device), y.to(args.device)
        #         y_pre = model_MF(x_u, x_i)
        #         valid_loss = loss_func(torch.sigmoid(y_pre), y)
        #
        #         total_loss_valid += valid_loss.item() * len(y)
        #         total_len_valid += len(y)
        #     total_loss_valid = total_loss_valid / total_len_valid
        #
        #     early_stopping(total_loss_valid, model_MF)
        #
        #     if early_stopping.early_stop:
        #         print("Early stopping")
        #
        #         break

    model.load_state_dict(torch.load(saved_path))
    predicts, GT_labels = [], []
    for x_u, x_i, y in train_dataloader:
        x_u, x_i, y = x_u.to(args.device), x_i.to(args.device), y.to(args.device)
        y_pre = model(x_u, x_i)
        predicts.extend(y_pre.tolist())
        GT_labels.extend(y.tolist())
    np.savetxt('./saved/{}/logits_{}/teacher_dim{}_train.txt'.format(args.data_name, args.model_name, args.dim),
               np.array(predicts),
               delimiter=',')
    train_accuracy = binary_accuracy_metric(torch.tensor(predicts), torch.tensor(GT_labels)).item()
    print("train accuracy:", train_accuracy)

    predicts, GT_labels = [], []
    for x_u, x_i, y in test_dataloader:
        x_u, x_i, y = x_u.to(args.device), x_i.to(args.device), y.to(args.device)
        y_pre = model(x_u, x_i)
        predicts.extend(y_pre.tolist())
        GT_labels.extend(y.tolist())
    np.savetxt('./saved/{}/logits_{}/teacher_dim{}_test.txt'.format(args.data_name, args.model_name, args.dim),
               np.array(predicts),
               delimiter=',')
    test_accuracy = binary_accuracy_metric(torch.tensor(predicts), torch.tensor(GT_labels)).item()
    print("test accuracy:", test_accuracy)
    torch.save(model.user_emb.weight.data, './saved/{}/emb_{}/teacher_dim{}_user.pt'.format(args.data_name, args.model_name, args.dim))
    torch.save(model.item_emb.weight.data, './saved/{}/emb_{}/teacher_dim{}_item.pt'.format(args.data_name, args.model_name, args.dim))
    return train_accuracy, test_accuracy


if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print("device:", args.device)

    dim_list = [64]
    train_accuracy_list, test_accuracy_list = [], []
    for dim in dim_list:
        args.dim = dim
        train_accuracy, test_accuracy = train(args)
        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)

    print("train_accuracy_list:", train_accuracy_list)
    print("test_accuracy_list:", test_accuracy_list)
