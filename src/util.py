import torch
import dgl


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


def print_args(args):
    args_dict = vars(args)
    print("+------------------------------------------------------+")
    for k, v in args_dict.items():
        print("|", k, v)
    print("+------------------------------------------------------+")


def get_dataloaders(g, args, *idx):

    train_idx, valid_idx, test_idx = idx

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)
    train_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_idx,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    val_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        valid_idx,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=1,
    )

    test_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        test_idx,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=1,
    )

    return train_dataloader, val_dataloader, test_dataloader
