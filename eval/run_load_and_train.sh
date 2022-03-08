#! /bin/bash

# byol encoder output dim = 512
# latent vector output dim = 16
# 2^9 = 512, 2^8 = 256, 2^7 = 128, 2^6 = 64, 2^5 = 32, 2^4 = 16


python load_and_train.py \
                        --hid1 128 \
                        --hid2 128 \
                        --hid3 64 \
                        --hid4 32 \
                        --tenbrd-enable \
                        --model-save-dir ./runs/