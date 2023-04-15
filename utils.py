import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from operator import itemgetter
from heapq import nlargest
import torch
import random
import itertools
import time


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.eval_flag = False
        self.val_metric_best = -np.Inf
        self.delta = delta
        self.path = path
        # self.path_ku = path_ku
        # self.path_ki = path_ki
        self.trace_func = trace_func

    def __call__(self, val_metric, model):

        score = val_metric

        if self.best_score is None:
            self.eval_flag = True
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:
            self.eval_flag = False
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.eval_flag = True
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        '''Saves model_MF when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'valid acc increase ({self.val_metric_best:.6f} --> {val_metric:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_metric_best = val_metric
