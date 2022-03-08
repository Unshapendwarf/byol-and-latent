import argparse

parser = argparse.ArgumentParser(description='EDSR and MDSR')

# Hardware specifications
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--tenbrd-enable', action='store_true', help='enable tensorboard')
# parser.add_argument('--n_GPUs', type=int, default=1,
#                     help='number of GPUs')

# Data specifications
# parser.add_argument('--dir_data', type=str, default='/home/hong/dataset/',
#                     help='dataset directory')


# Training specifications
parser.add_argument('--epochs', type=int, default=400,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=512,
                    help='input batch size for training')

parser.add_argument('--train-path', type=str, default = "./tensors/byol_encoder_out/train.pt",
                    help='train_tensor_path')
parser.add_argument('--test-path', type=str, default = "./tensors/byol_encoder_out/test.pt",
                    help='test_tensor_path')
parser.add_argument('--model-save-dir', type=str, default = "./runs/")
parser.add_argument('--out-save-dir', type=str, default = "./tensors/latent_out/")

parser.add_argument('--hid1', type=int, default=512, help='hidden layer 1 size')
parser.add_argument('--hid2', type=int, default=512, help='hidden layer 2 size')
parser.add_argument('--hid3', type=int, default=256, help='hidden layer 3 size')
parser.add_argument('--hid4', type=int, default=256, help='hidden layer 4 size')
parser.add_argument('--hid5', type=int, default=128, help='hidden layer 5 size')
parser.add_argument('--hid6', type=int, default=128, help='hidden layer 6 size')
parser.add_argument('--hid7', type=int, default=64, help='hidden layer 7 size')

args = parser.parse_args()

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

