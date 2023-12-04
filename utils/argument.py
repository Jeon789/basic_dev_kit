import argparse
from .util import str2bool

def create_argparser():
    defaults = dict(
        data_dir="/data/jan4021/0001_NOISE_SRGB",
        image_size = 128,
        sigma= 0.01
        # val_data_dir="",
        # noised=True,
        # iterations=150000,
        # lr=3e-4,
        # weight_decay=0.0,
        # anneal_lr=False,
        # batch_size=1,
        # microbatch=-1,
        # schedule_sampler="uniform",
        # resume_checkpoint="",
        # log_interval=10,
        # eval_interval=5,
        # save_interval=10000,
        # use_mpi=False
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


# parser
parser = argparse.ArgumentParser()
parser.add_argument('--yaml', type=str, default='/home/jan4021/zzz/config.yaml')



# parser.add_argument('--site', type=str, default='sionyu')
# parser.add_argument('--loss_fn', type=str, default='L1Loss')
# parser.add_argument('--data_per_oneday', type=int, default=1)


# parser.add_argument('--model', type=str, default='res10')
# parser.add_argument('--seconds', type=int, default=25)

# parser.add_argument('--epochs', type=int, default='100')

# parser.add_argument('--decay', type=float, default=0.98)
# parser.add_argument('--seed', type=int, default=0)
# parser.add_argument('--config', type=str, default='config.yaml')
# parser.add_argument('--debug', type=str2bool, default=False)
# parser.add_argument('--', type=str, default='')

args = parser.parse_args()