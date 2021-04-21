import torch
import torch.nn as nn
import math
import dgl


def accuracy(scores, targets):
    scores = scores.argmax(dim=1)
    acc = (scores == targets.squeeze()).float().sum().item()
    acc = acc / len(targets)
    return acc


def train_iter_batched(
    model, g, input_nodes, output_nodes, blocks, optimizer, device, epoch
):

    model = model.to(device)
    blocks = [b.to(device) for b in blocks]

    model.train()

    # x = blocks[0].srcdata["feat"]
    # lap_pos_enc = blocks[0].srcdata["lap_pos_enc"]
    # labels = blocks[-1].dstdata["label"]

    x = g.ndata["feat"].to(device)
    lap_pos_enc = g.ndata["lap_pos_enc"].to(device)
    labels = blocks[-1].dstdata["label"]

    optimizer.zero_grad()

    scores = model(blocks, x, lap_pos_enc, input_nodes, output_nodes)

    loss = model.loss(scores, labels)
    loss.backward(retain_graph=True)
    optimizer.step()
    acc = accuracy(scores, labels)
    return loss.item(), acc, optimizer


def evaluate_batched(model, g, input_nodes, output_nodes, blocks, device):

    model = model.to(device)
    blocks = [b.to(device) for b in blocks]

    model.eval()

    # x = blocks[0].srcdata["feat"]
    # lap_pos_enc = blocks[0].srcdata["lap_pos_enc"]
    # labels = blocks[-1].dstdata["label"]

    x = g.ndata["feat"].to(device)
    lap_pos_enc = g.ndata["lap_pos_enc"].to(device)

    labels = blocks[-1].dstdata["label"]

    scores = model(blocks, x, lap_pos_enc, input_nodes, output_nodes)
    loss = model.loss(scores, labels)

    acc = accuracy(scores, labels)
    return loss.item(), acc


def train_iter(model, g, mask, optimizer, device, epoch):

    model.train()
    epoch_loss = epoch_train_acc = 0

    x = g.ndata["feat"].to(device)
    try:
        e = g.edata["feat"].to(device)
    except:
        e = None

    labels = g.ndata["label"].to(device)
    lap_pos_enc = g.ndata["lap_pos_enc"].to(device)
    optimizer.zero_grad()

    scores = model(g, x, e, lap_pos_enc)

    loss = model.loss(scores[mask], labels[mask])
    loss.backward(retain_graph=True)
    optimizer.step()
    epoch_loss += loss.item()
    epoch_train_acc += accuracy(scores[mask], labels[mask])
    return epoch_loss, epoch_train_acc, optimizer


def evaluate(model, g, mask, device):

    model.eval()
    with torch.no_grad():
        x = g.ndata["feat"].to(device)
        try:
            e = g.edata["feat"].to(device)
        except:
            e = None

        labels = g.ndata["label"].to(device)
        lap_pos_enc = g.ndata["lap_pos_enc"].to(device)

        scores = model(g, x, e, lap_pos_enc)
        loss = model.loss(scores[mask], labels[mask])

    epoch_loss = loss.item()
    epoch_train_acc = accuracy(scores[mask], labels[mask])
    return epoch_loss, epoch_train_acc

