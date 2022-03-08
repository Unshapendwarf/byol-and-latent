import sys
# import time
# import os

import torch
import numpy as np

from sklearn import preprocessing
from torch import cuda
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

sys.path.append('../')
from mymodels.our_network import MyOne


def create_data_loaders_from_arrays(X1, y, batch_size):
    print("batch size: %d \n" % batch_size)
    train = torch.utils.data.TensorDataset(X1, y)
    train_loader = DataLoader(train, batch_size=1, shuffle=False)
    return train_loader

def dataload(data_path):
    loaded = torch.load(data_path)

    x1 = loaded['x1']
    y = loaded['y']

    print("data shape: ", x1.shape, y.shape)

    scaler = preprocessing.StandardScaler()
    scaler.fit(x1)

    x1 = scaler.transform(x1).astype(np.float32)
    my_loader = create_data_loaders_from_arrays(torch.from_numpy(x1), y, 64)

    return my_loader


class Latent_generator():
    def __init__(self, pretrain_model, data_loader, device):
        self.model = pretrain_model
        self.data_loader = data_loader
        self.device = device

    def generate(self, save_dir):
        model = self.model
        model = model.to(self.device)

        with torch.no_grad():
            self.model.eval()
            for i, (x1, y) in enumerate(tqdm(self.data_loader)):
                element = x1[0]
                element = element.to(self.device)
                y = y.to(self.device)

                out = model(element)
                
                lhs = int(y[0].item() / 1000)
                rhs = int(y[0].item() % 1000)
                orgn_name = str(lhs)+"_"+str(rhs)

                torch.save(out, save_dir +orgn_name+".pt")


if __name__ == "__main__":
    curr_dir = "/home/hong/dir4/byol-and-latent/latent_eval/"

    #device setting
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    # model parameter setting
    byol_encoder_out_dim = 512
    latent_out_dim = 16
    h1, h2, h3, h4 = 128, 128, 64, 32

    mymo = MyOne(byol_encoder_out_dim, h1, h2, h3, h4, latent_out_dim)
    mymo.load_state_dict(torch.load(curr_dir + "runs/512_16.pt"))  # input 1*512 and output 1*16
    
    # target psnr_sum
    target_psnr_sum = 60

    # dataloader setting
    # my_data_loader = dataload(curr_dir + "tensors/" + str(target_psnr_sum) + "db/byol_encoder_out/x2.pt")
    my_data_loader = dataload(curr_dir + "tensors/" + "final/byol_encoder_out/x2.pt")

    # latent generate
    # latent_out_dir = "tensors/"+str(target_psnr_sum)+"db/"
    latent_out_dir = "tensors/final/"
    latent_gen = Latent_generator(mymo, my_data_loader, device)
    latent_gen.generate( curr_dir + latent_out_dir + "latent_out/")