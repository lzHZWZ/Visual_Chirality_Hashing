import argparse
import torch

arg_lists = []
parser = argparse.ArgumentParser(description='Config for digital imaging pipeline+chirality training.')


parser = argparse.ArgumentParser()

parser.add_argument("--net_arch",
                    default='Alexnet',
                    help="Network model.")

parser.add_argument("--batch_size",
                    type=int, default=256,
                    help="Default is 4. Make sure it is an even number")
parser.add_argument("--learning_rate",
                    type=float, default=0.001,
                    help="The learning rate for optimizer. Default is 0.001")
parser.add_argument('--momentum',
                    type=float, default=0.9,
                    help="The momentum for SGD optimizer")

parser.add_argument("-optim", "--optimizer",
                    choices=['RMSprop', 'sgd'],
                    default="sgd",
                    help="The optimizer to use. 1/ sgd, 2/ adam. Default is sgd. ")

parser.add_argument("--weight_decay",
                    default=1e-5,
                    type=float,
                    help="Whether or not to use weight decay.")

parser.add_argument("--decay_step",
                    type=int, default=100,
                    help="After [decay_step] epochs, decay the learning rate by 0.1. Default is 17")

parser.add_argument("--num_workers",
                    type=int, default=4,
                    help="Default is 4")

parser.add_argument("--alpha",
                    default=0.1,
                    help="alpha")

parser.add_argument("--info",
                    default="[HashNet]",
                    help="print info")

parser.add_argument("--bit_size",
                    type=int,default=48,
                    help="length of hash code")

parser.add_argument("--step_continuation",
                    default=20,
                    help="step_continuation")

parser.add_argument("--resize_size",
                    default=224,
                    help="size of resize image")

parser.add_argument("--crop_size",
                    default=224,
                    help="size of crop image")

parser.add_argument("--dataset",
                    default="voc2007/chiral_0_achiral_100",
                    help="which dataset to load from ./data")

parser.add_argument("--data_path",
                    default="Your_Path",
                    help="path of train data")

parser.add_argument("--epoch",
                    type=int, default=50,
                    help="training epoch")

parser.add_argument("--n_class",
                    type=int, default=20,
                    help="the class of data,mscoco=90,mirflickr=38")

parser.add_argument("--test_map",
                    default=15,
                    help="test_map")

parser.add_argument("--rotation_size",
                    default=10,
                    help="degree of rotation")

parser.add_argument("--save_path",
                    default="save/HashNet",
                    help="save_path")

parser.add_argument("--device",
                    default= torch.device("cuda:0"),
                    help="save_path")

parser.add_argument("--topK",
                    default= -1,
                    help="save_path")

parser.add_argument("--data",
                    default={})

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
