import datetime
import numpy as np
import random as python_random
import torch

def set_seed(seed):
    # Sets the seed for random number generation
    torch.manual_seed(seed)
    np.random.seed(seed)
    python_random.seed(seed)

def generate_simid():
    now = datetime.datetime.now()
    simid = now.strftime("%y-%m-%d-%H-%M-%S")
    return simid

def stat_label(p, criteria=(0.05, 0.01, 0.001)):
    if p <= criteria[2]:
        return "***"
    elif p <= criteria[1]:
        return "**"
    elif p <= criteria[0]:
        return "*"
    else:
        return "ns"

def list_members_of_other_list(ls1, ls2):
    return [elem for elem in ls1 if elem in ls2]

def add_arguments(parser, args):
    if "output" in args:
        parser.add_argument("-o", "--output", help="path to output file")
    if "style" in args:
        parser.add_argument("--style", default="paper", help="format style (paper, presentation)")
    if "legend_style" in args:
        parser.add_argument("--legend_style", default="default")
    if "width" in args:
        parser.add_argument("--width", type=float, default=None)
    if "height" in args:
        parser.add_argument("--height", type=float, default=None)

