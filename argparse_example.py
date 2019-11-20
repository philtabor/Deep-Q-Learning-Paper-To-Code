import argparse

parser = argparse.ArgumentParser(description='')

# type can be int, str, float, bool, etc.
# this argument is optional
parser.add_argument('-argument', type=dtype, default=x, help='help string')

# this argument is not optional
parser.add_argument('argument', type=dtype, default=x, help='help string')

# parse the args.
args = parser.parse_args()

# access parameters like this
variable = args.argument
