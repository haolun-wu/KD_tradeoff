import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import numpy as np
from model.Wide_Deep_BCE import Wide_Deep
from argparse import ArgumentParser
from utils import EarlyStopping
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torchmetrics.classification import BinaryAccuracy
from prettytable import PrettyTable

binary_accuracy_metric = BinaryAccuracy()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def parse_args():
    parser = ArgumentParser(description="KD")
    parser.add_argument("--data_name", type=str, default="criteo")

    # parser.add_argument('--val_ratio', type=float, default=0.1)
    # parser.add_argument('--test_ratio', type=float, default=0.3)
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--is_logging', type=bool, default=False)
    # Seed
    parser.add_argument('--seed', type=int, default=2023, help="Seed (For reproducability)")
    # Model
    parser.add_argument("--model_name", type=str, default="Wide_Deep")
    parser.add_argument('--dim', type=int, default=32, help="Dimension for embedding")
    parser.add_argument('--dnn_hidden_units', nargs='+', type=int, default=[64, 32], help='hidden layer dimensions')
    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--wd', type=float, default=1e-5, help="Weight decay factor")
    # Training
    parser.add_argument('--n_epochs', type=int, default=1000, help="Number of epoch during training")
    parser.add_argument('--every', type=int, default=1,
                        help="Period for evaluating precision and recall during training")
    parser.add_argument('--patience', type=int, default=10, help="patience")
    parser.add_argument('--batch_size', type=int, default=512, help="batch_size")

    return parser.parse_args()


def train(args):
    path = 'data_raw/{}/'.format(args.data_name)
    if args.data_name == 'criteo':
        sparse_feature = ['C' + str(i) for i in range(1, 27)]
        dense_feature = ['I' + str(i) for i in range(1, 14)]
        col_names = ['label'] + dense_feature + sparse_feature
        print("col_names:", col_names)
        df = pd.read_csv(path + 'dac_sample.txt', names=col_names, sep='\t')

        df[sparse_feature] = df[sparse_feature].fillna('-1', )
        df[dense_feature] = df[dense_feature].fillna('0', )

        feat_sizes = {}
        feat_sizes_dense = {feat: 1 for feat in dense_feature}
        feat_sizes_sparse = {feat: len(df[feat].unique()) for feat in sparse_feature}
        feat_sizes.update(feat_sizes_dense)
        feat_sizes.update(feat_sizes_sparse)

        for feat in sparse_feature:
            lbe = LabelEncoder()
            df[feat] = lbe.fit_transform(df[feat])
        nms = MinMaxScaler(feature_range=(0, 1))
        df[dense_feature] = nms.fit_transform(df[dense_feature])

        fixlen_feature_columns = [(feat, 'sparse') for feat in sparse_feature] + [(feat, 'dense') for feat in
                                                                                  dense_feature]
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        x, y = df.iloc[:, 1:], df.iloc[:, 0]

    print

    x_teacher, x_student, y_teacher, y_student = train_test_split(x, y, test_size=0.3, random_state=args.seed)
    x_train, x_val_test, y_train, y_val_test = train_test_split(x_teacher, y_teacher, test_size=0.2,
                                                                random_state=args.seed)
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=args.seed)

    np.savetxt('./saved/{}/gt_label_train.txt'.format(args.data_name), np.array(y_train.to_numpy()), delimiter=',')
    np.savetxt('./saved/{}/gt_label_val.txt'.format(args.data_name), np.array(y_val.to_numpy()), delimiter=',')
    np.savetxt('./saved/{}/gt_label_test.txt'.format(args.data_name), np.array(y_test.to_numpy()), delimiter=',')

    print("x_train shape:{}, y_train shape{}".format(x_train.shape, y_train.shape))
    print("x_val shape:{}, y_val shape{}".format(x_val.shape, y_val.shape))
    print("x_test shape:{}, y_test shape{}".format(x_test.shape, y_test.shape))

    train_dataset = TensorDataset(torch.from_numpy(np.array(x_train)), torch.from_numpy(np.array(y_train)))
    val_dataset = TensorDataset(torch.from_numpy(np.array(x_val)), torch.from_numpy(np.array(y_val)))
    test_dataset = TensorDataset(torch.from_numpy(np.array(x_test)), torch.from_numpy(np.array(y_test)))

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)

    # num_users, num_items = max(df[0]) + 1, max(df[1]) + 1
    print("args.dnn_hidden_units:", args.dnn_hidden_units)
    if args.model_name == 'Wide_Deep':
        model = Wide_Deep(feat_sizes, args.dim, linear_feature_columns, dnn_feature_columns, args.dnn_hidden_units).to(
            args.device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
    # loss_func = torch.nn.BCELoss().to(args.device)

    loss_func = torch.nn.BCELoss().to(args.device)
    # loss_func = torch.nn.CrossEntropyLoss().to(args.device)

    saved_path = './saved/{}/model_{}/teacher_dim{}_BCE.pt'.format(args.data_name, args.model_name, args.dim)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=saved_path)

    for epoch in range(args.n_epochs):
        model.train()
        total_loss, total_len = 0, 0
        for x, y_g in train_dataloader:
            x, y_g = x.to(args.device).float(), y_g.to(args.device).float().reshape(-1, 1)
            # print("x_u:", x_u[:5])

            y_pre = model(x)
            loss = loss_func(torch.sigmoid(y_pre), y_g)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_g)
            total_len += len(y_g)
        train_loss = total_loss / total_len
        print("epoch {}, train loss is {:.6f}".format(epoch, train_loss))

        if epoch % args.every == 0:
            model.eval()
            labels, predicts = [], []
            with torch.no_grad():
                for x, y_g in val_dataloader:
                    x, y_g = x.to(args.device).float(), y_g.to(args.device).float().reshape(-1, 1)
                    y_pre = model(x)
                    labels.extend(y_g.tolist())
                    predicts.extend(y_pre.tolist())
                valid_acc = binary_accuracy_metric(torch.tensor(predicts), torch.tensor(labels)).item()

                early_stopping(valid_acc, model)

                if early_stopping.early_stop:
                    print("Early stopping")

                    break

    model.load_state_dict(torch.load(saved_path))
    predicts, GT_labels = [], []
    for x, y_g in train_dataloader:
        x, y_g = x.to(args.device).float(), y_g.to(args.device).float().reshape(-1, 1)
        y_pre = model(x)
        predicts.extend(y_pre.tolist())
        GT_labels.extend(y_g.tolist())
    train_accuracy = binary_accuracy_metric(torch.tensor(predicts), torch.tensor(GT_labels)).item()
    print("train accuracy:", train_accuracy)

    predicts, GT_labels = [], []
    for x, y_g in test_dataloader:
        x, y_g = x.to(args.device).float(), y_g.to(args.device).float().reshape(-1, 1)
        y_pre = model(x)
        predicts.extend(y_pre.tolist())
        GT_labels.extend(y_g.tolist())
    test_accuracy = binary_accuracy_metric(torch.tensor(predicts), torch.tensor(GT_labels)).item()
    print("test accuracy:", test_accuracy)

    return train_accuracy, test_accuracy


if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print("device:", args.device)
    train_accuracy_list, test_accuracy_list = [], []

    data_list = ['criteo']
    dim_list = [32]
    dnn_hidden_units_list = [
        # [128, 64, 32, 16, 8],
        # [128, 64, 32, 16],
        # [64, 32, 16, 8],
        # [128, 64, 32],
        # [64, 32, 16],
        # [32, 16, 8],
        # [128, 64],
        [64, 32],
        # [32, 16],
        # [16, 8]
    ]
    result_table = PrettyTable(['Dataset', 'dim', 'hidden_units', 'ACC_train', 'ACC_test'])
    for data in data_list:
        for dim in dim_list:
            for dnn_hidden_units in dnn_hidden_units_list:
                args.data_name = data
                args.dim = dim
                args.dnn_hidden_units = dnn_hidden_units
                train_accuracy, test_accuracy = train(args)
                train_accuracy_list.append(train_accuracy)
                test_accuracy_list.append(test_accuracy)
                result_table.add_row([data, dim, dnn_hidden_units, train_accuracy, test_accuracy])

    print("train_accuracy_list:", train_accuracy_list)
    print("test_accuracy_list:", test_accuracy_list)
    print(result_table)
