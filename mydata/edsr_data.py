from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import glob
import numpy
import random

import pandas as pd

#######################################################
#               Define Transforms
#######################################################
#To define an augmentation pipeline, you need to create an instance of the Compose class.
#As an argument to the Compose class, you need to pass a list of augmentations you want to apply. 
#A call to Compose will return a transform function that will perform image augmentation.
#(https://albumentations.ai/docs/getting_started/image_augmentation/)

train_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
        A.RandomCrop(height=256, width=256),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        A.CenterCrop(height=256, width=256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


####################################################
#       Create Train, Valid and Test sets
####################################################
train_data_path = 'images/train' 
test_data_path = 'images/test'
# train_data_path = '/mnt/URP_DS/HR' # without last slash
# test_data_path = '/mnt/URP_DS/HR_mid'  #without last slash

#1.
# get all the paths from train_data_path and append image paths and class to to respective lists
# eg. train path-> 'images/train/26.Pont_du_Gard/4321ee6695c23c7b.jpg'
# eg. class -> 26.Pont_du_Gard
train_image_paths = [] #to store image paths in list

train_image_paths.append(glob.glob(train_data_path + '/*.png'))
train_image_paths = list(flatten(train_image_paths))
random.shuffle(train_image_paths)  # ??? necessary??

print('train_image_path example: ', train_image_paths[0])

#2.
# split train valid from train paths (80,20)
# if you don't need the validation set, delete it
train_image_paths, valid_image_paths = train_image_paths[:int(0.8*len(train_image_paths))], train_image_paths[int(0.8*len(train_image_paths)):] 

#3.
# create the test_image_paths
test_image_paths = []
test_image_paths.append(glob.glob(test_data_path + '/*.png'))
test_image_paths = list(flatten(test_image_paths))

print("Train size: {}\nValid size: {}\nTest size: {}".format(len(train_image_paths), len(valid_image_paths), len(test_image_paths)))

############################################

def load_file(opt):
    data_path = []
    f = open("{}.txt".format(opt), 'r')
    while True:
        line = f.readline()
        if not line:break
        data_path.append(line[:-1])
    f.close()
    return data_path

class MyDataset(Dataset):
    def __init__(self, opt_data):
        super(MyDataset, self).__init__()
        """
        opt_data : 'train', 'validation'
        """
        # self.file_list = load_file('/mnt/URP_DS')
        # y = pd.read_csv('audio_data/train_answer.csv', index_col=0)
        # change this one to another value

        self.y = y.values
        
    def __getitem__(self, index):
        x = np.load(self.file_list[index])
        self.x_data = torch.from_numpy(x).float()
        self.y_data = torch.from_numpy(self.y[index]).float()
        return self.x_data, self.y_data

    def __len__(self):
        return len(self.y)
        

# not necessaries under
if __name__ == "__main__":
    train_x = torch.rand(500)
    train_y = torch.rand(500)
    tr_dataset = MyDataset(train_x)
    # tr_dataset = MyDataset(train_x, train_y)


##########################################################
# def flatten( mylist ):
#     np.arrray().flatten().tolist()

# image_path = '/mnt/URP_DS/HR' # without last slash
# test_data_path = '/mnt/URP_DS/HR'  #without last slash
# cross_result_csv_path = '/home/hong/dir1/PyTorch-BYOL/mydata/base_DS.csv'

# #1.
# train_image_paths = [] #to store image paths in list
# train_image_paths.append(glob.glob(image_path + '/*.png'))
# train_image_paths= np.array(train_image_paths).flatten().tolist()
# random.shuffle(train_image_paths)  # ??? necessary??

# print('train_image_path example: ', train_image_paths[0])

# #2.
# # split train valid from train paths (80,20)
# # if you don't need the validation set, delete it
# train_image_paths, valid_image_paths = train_image_paths[:int(0.8*len(train_image_paths))], train_image_paths[int(0.8*len(train_image_paths)):] 

# #3.
# # create the test_image_paths
# test_image_paths = []
# test_image_paths.append(glob.glob(test_data_path + '/*.png'))
# test_image_paths= np.array(test_image_paths).flatten().tolist()

# print("Train size: {}\nValid size: {}\nTest size: {}".format(len(train_image_paths), len(valid_image_paths), len(test_image_paths)))

