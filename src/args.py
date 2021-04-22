import argparse


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Please give a config.json file with training/model/data/param details",
    )
    parser.add_argument("--out_dir", help="Please give a value for out_dir")
    parser.add_argument(
        "--epochs", type=int, help="Please give a value for epochs", default=100
    )
    parser.add_argument("--batch_size", help="Please give a value for batch_size")
    parser.add_argument(
        "--init_lr", help="Please give a value for init_lr", default=0.001
    )
    parser.add_argument(
        "--lr_reduce_factor",
        help="Please give a value for lr_reduce_factor",
        default=0.5,
    )
    parser.add_argument(
        "--lr_schedule_patience",
        help="Please give a value for lr_schedule_patience",
        default=10,
    )
    parser.add_argument("--min_lr", help="Please give a value for min_lr")
    parser.add_argument(
        "--weight_decay", help="Please give a value for weight_decay", default=0.0
    )
    parser.add_argument("--L", help="Please give a value for L")
    parser.add_argument(
        "--hidden_dim", help="Please give a value for hidden_dim", default=128
    )
    parser.add_argument(
        "--out_dim", help="Please give a value for out_dim", default=128
    )
    parser.add_argument("--edge_feat", help="Please give a value for edge_feat")
    parser.add_argument("--readout", help="Please give a value for readout")
    parser.add_argument(
        "--num_heads", help="Please give a value for n_heads", default=8
    )
    parser.add_argument(
        "--in_feat_dropout", help="Please give a value for in_feat_dropout", default=0.0
    )
    parser.add_argument(
        "--dropout", help="Please give a value for dropout", default=0.0
    )
    parser.add_argument("--self_loop", help="Please give a value for self_loop")
    parser.add_argument(
        "--pos_enc_dim", help="Please give a value for pos_enc_dim", default=10
    )
    parser.add_argument(
        "--lap_pos_enc", help="Please give a value for lap_pos_enc", default=True
    )

    parser.add_argument(
        "--num_workers", help="Please give a value for num-workers", default=4
    )
    parser.add_argument(
        "--dataset", help="Please give a value for dataset", default="arxiv"
    )
    return parser.parse_args()
