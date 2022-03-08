import sys
import time
import os

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from sklearn import preprocessing
from torch import cuda
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from options import args


sys.path.append('../')
# from mymodels.resnet_base_network import ResNet18
# from mydata.imageloader import MyDataset, psnrDataUnit
from mymodels.our_network import MyOne


def create_data_loaders_from_arrays(X1_train, X2_train, y_train, X1_test, X2_test, y_test, arg_batch_size):
    print("batch size: %d \n" % arg_batch_size)
    train = torch.utils.data.TensorDataset(X1_train, X2_train, y_train)
    train_loader = DataLoader(train, batch_size=arg_batch_size, shuffle=True)

    test = torch.utils.data.TensorDataset(X1_test, X2_test, y_test)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)
    return train_loader, test_loader

def dataload(args):
    train_load_path = args.train_path
    test_load_path = args.test_path

    loaded1 = torch.load(train_load_path)
    loaded2 = torch.load(test_load_path)

    x1_train = loaded1['x1']
    x2_train = loaded1['x2']
    y_train = loaded1['y']
    x1_test = loaded2['x1']
    x2_test = loaded2['x2']
    y_test = loaded2['y']

    print("Training data shape:", x1_train.shape, x2_train.shape, y_train.shape)
    print("Testing data shape:", x1_test.shape, x2_test.shape, y_test.shape)

    scaler = preprocessing.StandardScaler()
    scaler.fit(x1_train)
    
    x1_train = scaler.transform(x1_train).astype(np.float32)
    x2_train = scaler.transform(x2_train).astype(np.float32)
    x1_test = scaler.transform(x1_test).astype(np.float32)
    x2_test = scaler.transform(x2_test).astype(np.float32)
    train_loader, test_loader = create_data_loaders_from_arrays(torch.from_numpy(x1_train), torch.from_numpy(x2_train), y_train, 
                                                                torch.from_numpy(x1_test), torch.from_numpy(x2_test), y_test, args.batch_size)

    return train_loader, test_loader

class Trainer():
    def __init__(self, args, my_model, my_loss, my_optimizer, my_writer, train_loader, test_loader, device):
        self.args = args
        self.model = my_model
        self.loss = my_loss
        self.optimizer = my_optimizer
        self.sumwriter = my_writer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def train(self):
        epoch = self.args.epochs
        model = self.model
        model = model.to(self.device)
        optimizer = self.optimizer

        model.train()
        for epoch in tqdm(range(epoch)):
            for x1, x2, y in self.train_loader:

                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                y = y.to(self.device)
                
                # zero the parameter gradients
                optimizer.zero_grad() 
                
                out1 = model(x1)
                out2 = model(x2)

                logits = torch.cdist(out1.unsqueeze(1), out2.unsqueeze(1))
                loss = self.loss(logits.squeeze(1), y.unsqueeze(1))

                loss.backward()
                optimizer.step()
                if self.args.tenbrd_enable:
                    self.sumwriter.add_scalar("Loss/train", loss, epoch)
        if self.args.tenbrd_enable:
            self.sumwriter.close()
    def savemodel(self, save_dir):
        # model save
        # torch.save(self.model, save_dir)
        torch.save(self.model.state_dict(), save_dir + "trained_model.pt")

    def test(self):
        test_result_diff = []
        now = time.localtime()

        model = self.model
        model = model.to(self.device)

        with torch.no_grad():
            self.model.eval()
            for x1, x2, y in tqdm(self.test_loader):
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                y = y.to(self.device)    
                
                out1 = model(x1)
                out2 = model(x2)
                # predictions = torch.argmax(logits, dim=1)
                
                logits = torch.cdist(out1.unsqueeze(1), out2.unsqueeze(1))
                logits = logits.squeeze(1)

                ty = 1/float(y[0].item())
                
                if logits[0].item()==0:
                    oy = 100
                else: 
                    oy = 1 /logits[0].item()
                test_result_diff.append((ty, oy, abs(ty-oy)))

        with open("result_%.2f_%02d%02d_%02d%02d.txt" %(1.0, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min), 'w') as f:
            for item1, item2, item3 in test_result_diff:
                f.write("%s, %s, %s\n" % (item1, item2, item3)) # sr_sum, model_out, abs(diff)


if __name__ == "__main__":
    
    #device setting
    device = 'cuda' if cuda.is_available() else 'cpu'
    if args.cpu:
        device = 'cpu'
    else:
        torch.cuda.empty_cache()
    print(f"Training with: {device}")

    # model setting, hidden layer: 4
    byol_encoder_out_dim = 512
    latent_out_dim = 16
    h1, h2, h3, h4 = args.hid1, args.hid2, args.hid3, args.hid4

    mymo = MyOne(byol_encoder_out_dim, h1, h2, h3, h4, latent_out_dim)
    print("Model's state_dict:")
    for param_tensor in mymo.state_dict():
        print(param_tensor, "\t", mymo.state_dict()[param_tensor].size())

    # loss setting, L1
    mycriterion = torch.nn.L1Loss()

    # optimizer setting
    myoptimizer = torch.optim.Adam(mymo.parameters(), lr=3e-4)

    # tensorboard setting
    writer = False
    if args.tenbrd_enable:
        writer = SummaryWriter()

    #dataload
    train_ld, test_ld = dataload(args)
    t = Trainer(args, mymo, mycriterion, myoptimizer, writer, train_ld, test_ld, device)
    t.train()
    t.savemodel(args.model_save_dir)
    # t.test()