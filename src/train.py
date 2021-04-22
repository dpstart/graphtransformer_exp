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

    x = g.ndata["feat"].to(device)
    lap_pos_enc = g.ndata["lap_pos_enc"].to(device)
    sign_flip = torch.rand(lap_pos_enc.size(1)).to(device)
    sign_flip[sign_flip >= 0.5] = 1.0
    sign_flip[sign_flip < 0.5] = -1.0
    lap_pos_enc = lap_pos_enc * sign_flip.unsqueeze(0)
    labels = blocks[-1].dstdata["label"]

    optimizer.zero_grad()

    scores = model(blocks, x, lap_pos_enc, input_nodes, output_nodes)

    loss = model.loss(scores, labels)
    loss.backward()
    optimizer.step()
    acc = accuracy(scores, labels)
    return loss.item(), acc, optimizer


def evaluate_batched(model, g, input_nodes, output_nodes, blocks, device):

    model = model.to(device)
    blocks = [b.to(device) for b in blocks]

    model.eval()

    with torch.no_grad():

        x = g.ndata["feat"].to(device)
        lap_pos_enc = g.ndata["lap_pos_enc"].to(device)
        sign_flip = torch.rand(lap_pos_enc.size(1)).to(device)
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        lap_pos_enc = lap_pos_enc * sign_flip.unsqueeze(0)

        labels = blocks[-1].dstdata["label"]

        scores = model(blocks, x, lap_pos_enc, input_nodes, output_nodes)
        loss = model.loss(scores, labels)

        acc = accuracy(scores, labels)
    return loss.item(), acc
