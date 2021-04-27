import torch
import torch.nn as nn
import math
import dgl


def accuracy(scores, targets):
    scores = scores.argmax(axis=1)
    acc = (scores == targets.squeeze()).sum()
    acc = acc / len(targets)
    return acc


def train_iter(
    model, g, train_idx, optimizer,
):
    pass


def evaluate(model, g, val_idx):
    pass
