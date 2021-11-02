# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import sys
import yaml
from torchvision import transforms, datasets
import torchvision
import numpy as np
import os
from sklearn import preprocessing
from torch import cuda
from torch.utils.data.dataloader import DataLoader


# %%
sys.path.append('../')
from mymodels.resnet_base_network import ResNet18
from mydata.imageloader import MyDataset, psnrDataUnit


# %%
batch_size = 64
data_transforms = torchvision.transforms.Compose([transforms.ToTensor()])

device = 'cuda' if cuda.is_available() else 'cpu'
print(f"Training with: {device}")

# %%
config = yaml.load(open("../config/config.yaml", "r"), Loader=yaml.FullLoader)


# %%
encoder = ResNet18(**config['network'])
output_feature_dim = encoder.projetion.net[0].in_features
print(output_feature_dim)


# %%
#duplication
encoder2 = ResNet18(**config['network'])
output_feature_dim2 = encoder2.projetion.net[0].in_features


# %%
#load pre-trained parameters
#load_params = torch.load(os.path.join('/home/hong/dir1/PyTorch-BYOL/runs/Jul29_18-37-02_mango2/checkpoints/model.pth'),
#                         map_location=torch.device(torch.device(device)))
# epoch: 10
load_params = torch.load(os.path.join('/home/hong/dir1/PyTorch-BYOL/runs/Sep26_15-10-29_mango2/checkpoints/model.pth'),
                        map_location=torch.device(torch.device(device)))

#load_params = torch.load(os.path.join('/home/hong/dir1/PyTorch-BYOL/runs/Jul29_01-32-09_mango2/checkpoints/model.pth'),
#                         map_location=torch.device(torch.device(device)))\

# hong2
load_params2 = torch.load(os.path.join('/home/hong/dir1/PyTorch-BYOL/runs/using0928/checkpoints/model.pth'),
                        map_location=torch.device(torch.device(device)))

if 'online_network_state_dict' in load_params:
    encoder.load_state_dict(load_params['online_network_state_dict'])
    print("Parameters successfully loaded.")

# remove the projection head
encoder = torch.nn.Sequential(*list(encoder.children())[:-1])    
encoder = encoder.to(device)


# %%
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)


# %%
# Neural Network Class
class MyOne(torch.nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, H3)
        self.linear4 = torch.nn.Linear(H3, D_out)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.nn.functional.relu(self.linear3(x))
        x = self.linear4(x)
        return x    


# %%
# logreg = LogisticRegression(output_feature_dim*2, 10)
# logreg = logreg.to(device)
mymo = MyOne(output_feature_dim*2, 125, 65, 24, 1)
mymo = mymo.to(device)
# 모델의 state_dict 출력
print("Model's state_dict:")
for param_tensor in mymo.state_dict():
    print(param_tensor, "\t", mymo.state_dict()[param_tensor].size())


# %%
def get_features_from_encoder(encoder, loader):
    
    x_train = []
    y_train = []

    # get the features from the pre-trained model
    for i, (x, y) in enumerate(loader):
        print(x.shape)
        with torch.no_grad():
            feature_vector = encoder(x)
            x_train.extend(feature_vector)
            y_train.extend(y.numpy())

            
    x_train = torch.stack(x_train)
    y_train = torch.tensor(y_train)
    return x_train, y_train


# %%
def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test):

    train = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

    test = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False)
    return train_loader, test_loader


#####################################################
# %%
#generating index dict
import random
import glob

indexDict = dict()
file_path = "/home/hong/dir1/PyTorch-BYOL/writing2.txt"
# open index file
f = open(file_path, 'r')
lines = f.readlines()
for line in lines:
    listed = line.split(" ")  # split by spaces
    if listed[0] not in indexDict:
        # dict has img1_name
        indexDict[listed[0]] = list()
    tmp_unit = psnrDataUnit(listed[1], listed[2], listed[3])
    indexDict[listed[0]].append(tmp_unit)
f.close()
# check the loading is valid or not
indexDictKeys = list(indexDict.keys())
random.shuffle(indexDictKeys) 


# %%
image_path = '/mnt/URP_DS/HR' # without last slash
test_data_path = '/mnt/URP_DS/HR'  #without last slash

img1_idx = 0
for img1_idx in range (0, 2):

    train_img_psnr_list = []
    for p_unit in indexDict[indexDictKeys[img1_idx]]:
        train_img_psnr_list.append((image_path + '/'+p_unit.getimg2()+'.png', p_unit.getonetwo()))
    # train_image_paths= np.array(train_image_paths).flatten().tolist()
    # random.shuffle(train_image_paths)  # ??? necessary??

    print('train_image_path example: ', train_img_psnr_list[img1_idx])
    print(len(train_img_psnr_list))

    # split train valid from train paths (80,20)
    train_img_psnr_list, valid_img_psnr_list = train_img_psnr_list[:int(0.8*len(train_img_psnr_list))], train_img_psnr_list[int(0.8*len(train_img_psnr_list)):]

    # create the test_image_paths
    test_img_psnr_list = []
    for p_unit in indexDict[indexDictKeys[img1_idx]]:
        test_img_psnr_list.append((image_path + '/'+p_unit.getimg2()+'.png', p_unit.getonetwo()))


    # %%
    # device = 'cpu'
    

    train_dataset = MyDataset(train_img_psnr_list, transform=data_transforms)
    test_dataset = MyDataset(test_img_psnr_list, transform=data_transforms)
    # test_dataset = MyDataset(valid_img_psnr_list, transform=data_transforms)
    single_dataset = MyDataset([('/mnt/URP_DS/HR/'+indexDictKeys[img1_idx]+'.png', 40)], transform=data_transforms)  # 40-> meaningless


    # %%
    print("Input shape:", train_dataset[0][0].shape)


    # %%
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=True, pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=True, pin_memory=True)
                            
    single_loader = DataLoader(single_dataset, batch_size=1,
                            num_workers=0, drop_last=False, shuffle=True, pin_memory=True)


    # %%
    encoder.eval()
    x_train, y_train = get_features_from_encoder(encoder, train_loader)
    x_test, y_test = get_features_from_encoder(encoder, test_loader)

    if len(x_train.shape) > 2:
        print(x_train.shape)
        x_train = torch.mean(x_train, dim=[2, 3])
        x_test = torch.mean(x_test, dim=[2, 3])
        
    print("Training data shape:", x_train.shape, y_train.shape)
    print("Testing data shape:", x_test.shape, y_test.shape)


    # %%
    x_single, y_single = get_features_from_encoder(encoder, single_loader)
    single_list = []
    for i in range (0,x_train.shape[0]):
        single_list.append(x_single)
    x_single1 = torch.stack(single_list, 1)
    x_single1 = x_single1.squeeze()
    print(x_single1.shape)

    x_train = torch.cat((x_train, x_single1), 1)

    single_list2 = []
    for i in range (0,x_test.shape[0]):
        single_list2.append(x_single)
    x_single2 = torch.stack(single_list2, 1)
    x_single2 = x_single2.squeeze()
    print(x_single2.shape)
    x_test = torch.cat((x_test, x_single2), 1)

    print(x_train.shape, x_test.shape)


    # %%
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)


    # %%
    train_loader, test_loader = create_data_loaders_from_arrays(torch.from_numpy(x_train), y_train, torch.from_numpy(x_test), y_test)


    # %%
    running_loss_history = []
    running_logit_history = []

    optimizer = torch.optim.Adam(mymo.parameters(), lr=3e-4)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.L1Loss()
    eval_every_n_epochs = 10

    # device = 'cuda' if cuda.is_available() else 'cpu'
    # print(f"Training with: {device}")

    for epoch in range(200):
    #     train_acc = []
        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()        
            
            logits = mymo(x)
            running_logit_history.append(1/logits[0].item())
            # predictions = torch.argmax(logits, dim=1)
            
            loss = criterion(logits, y)
            running_loss_history.append(loss)
            
            loss.backward()
            optimizer.step()
        
        # total = 0
        if epoch % eval_every_n_epochs == 0:
            # correct = 0
            validation_psnrsum_history = []
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                logits = mymo(x)
                validation_psnrsum_history.append(1/logits[0].item())
                
                # predictions = torch.argmax(logits, dim=1)
                
                # total += y.size(0)
                # correct += (predictions == y).sum().item()
            mymax = max(validation_psnrsum_history)
            mymin = min(validation_psnrsum_history)
            avg = sum(validation_psnrsum_history, 0.0)/len(validation_psnrsum_history)
            prediction = (mymax, mymin, avg)
            # acc = 100 * correct / total
            with open('val_result'+str(epoch)+'.txt', 'w') as f:
                for item in validation_psnrsum_history:
                    f.write("%s\n" % item)


# %%


