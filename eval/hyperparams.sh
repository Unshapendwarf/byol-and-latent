#! /bin/bash

# byol encoder output dim = 512
# latent vector output dim = 64

# 2^9 = 512, 2^8 = 256, 2^7 = 128, 2^6 = 64, 2^5 = 32, 2^4 = 16

# <fixed spec> model_input: 512, model_out: 32

# 1117 hidden layer 1
# python load_and_train.py --tenbrd_enable --hid1 256
# python load_and_train.py --tenbrd_enable --hid1 128
# python load_and_train.py --tenbrd_enable --hid1 64

# 1117 hidden layer 2
# python load_and_train.py --tenbrd_enable --hid1 512 --hid2 256
# python load_and_train.py --tenbrd_enable --hid1 256 --hid2 128
# python load_and_train.py --tenbrd_enable --hid1 256 --hid2 64
# python load_and_train.py --tenbrd_enable --hid1 128 --hid2 64

# 1117 hidden layer 3
# python load_and_train.py --tenbrd_enable --hid1 512 --hid2 256 --hid3 128
# python load_and_train.py --tenbrd_enable --hid1 512 --hid2 256 --hid3 256
# python load_and_train.py --tenbrd_enable --hid1 256 --hid2 256 --hid3 128
# python load_and_train.py --tenbrd_enable --hid1 256 --hid2 256 --hid3 64
# python load_and_train.py --tenbrd_enable --hid1 256 --hid2 128 --hid3 64
# python load_and_train.py --tenbrd_enable --hid1 128 --hid2 128 --hid3 64

# # 1117 hidden layer 4
python load_and_train.py --tenbrd_enable --hid1 512 --hid2 256 --hid3 256 --hid4 128
# python load_and_train.py --tenbrd_enable --hid1 512 --hid2 256 --hid3 128 --hid4 128
# python load_and_train.py --tenbrd_enable --hid1 512 --hid2 256 --hid3 128 --hid4 64
# python load_and_train.py --tenbrd_enable --hid1 256 --hid2 256 --hid3 128 --hid4 128
# python load_and_train.py --tenbrd_enable --hid1 256 --hid2 256 --hid3 128 --hid4 64
# python load_and_train.py --tenbrd_enable --hid1 256 --hid2 128 --hid3 64 --hid4 64
# python load_and_train.py --tenbrd_enable --hid1 128 --hid2 128 --hid3 64 --hid4 64

# # 1118 hidden layer 1 32
# python load_and_train.py --tenbrd_enable --hid1 256
# python load_and_train.py --tenbrd_enable --hid1 128
# python load_and_train.py --tenbrd_enable --hid1 64

# 1118 hidden layer 2 32
# python load_and_train.py --tenbrd_enable --hid1 512 --hid2 256
# python load_and_train.py --tenbrd_enable --hid1 256 --hid2 128
# python load_and_train.py --tenbrd_enable --hid1 256 --hid2 64
# python load_and_train.py --tenbrd_enable --hid1 128 --hid2 64

# 1118 hidden layer 3 32
# python load_and_train.py --tenbrd_enable --hid1 512 --hid2 256 --hid3 256
# python load_and_train.py --tenbrd_enable --hid1 512 --hid2 256 --hid3 128
# python load_and_train.py --tenbrd_enable --hid1 512 --hid2 256 --hid3 64
# python load_and_train.py --tenbrd_enable --hid1 256 --hid2 128 --hid3 64
# python load_and_train.py --tenbrd_enable --hid1 256 --hid2 64 --hid3 64
# python load_and_train.py --tenbrd_enable --hid1 128 --hid2 64 --hid3 32

# # 1118 hidden layer 4 32
# python load_and_train.py --tenbrd_enable --hid1 512 --hid2 256 --hid3 128 --hid4 64
# python load_and_train.py --tenbrd_enable --hid1 256 --hid2 256 --hid3 128 --hid4 64
# python load_and_train.py --tenbrd_enable --hid1 256 --hid2 128 --hid3 64 --hid4 64
# python load_and_train.py --tenbrd_enable --hid1 256 --hid2 128 --hid3 64 --hid4 32
# python load_and_train.py --tenbrd_enable --hid1 128 --hid2 128 --hid3 64 --hid4 64
# python load_and_train.py --tenbrd_enable --hid1 128 --hid2 64 --hid3 64 --hid4 32

# # # 1118 hidden layer 4 16
# python load_and_train.py --tenbrd_enable --hid1 256 --hid2 128 --hid3 64 --hid4 64
# python load_and_train.py --tenbrd_enable --hid1 256 --hid2 128 --hid3 64 --hid4 32
# python load_and_train.py --tenbrd_enable --hid1 128 --hid2 128 --hid3 64 --hid4 32
# python load_and_train.py --tenbrd_enable --hid1 128 --hid2 64 --hid3 64 --hid4 32
# # python load_and_train.py --tenbrd_enable --hid1 512 --hid2 256 --hid3 128 --hid4 64
# # python load_and_train.py --tenbrd_enable --hid1 256 --hid2 256 --hid3 128 --hid4 64