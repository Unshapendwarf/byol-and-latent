from torch.utils.data import Dataset, DataLoader

# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import cv2
import glob
import numpy as np

class psnrDataUnit:
    def __init__(self, name1, name2, psnr1, psnr2):
        self.img1name = name1
        self.img2name = name2 #image2 name in string
        self.psnr1 = float(psnr1) # psnr1 with float 
        self.psnr2 = float(psnr2) # psnr2 with float
    
    def getimg1(self):
        return self.img1name

    def getimg2(self):
        return self.img2name

    def getpsnr1(self):
        return self.psnr1

    def getpsnr2(self):
        return self.psnr2

    def getsrsum(self):
        return self.psnr1+self.psnr2


class MyDataset(Dataset):
    def __init__(self, images_psnrs, transform=False):
        '''
        MyDataset params
        images_psnrs(list): [(image1_path, image2_path, psnr_sum), ...]
        '''
        super(MyDataset, self).__init__()
        self.images_psnrs = images_psnrs
        self.transform = transform

        
    def __getitem__(self, index):
        image1_path, image2_path, psnr_sum = self.images_psnrs[index]
        image1 = cv2.imread(image1_path + ".png")
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.imread(image2_path + ".png")
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        invsVal = 1/float(psnr_sum) 
        return image1, image2, invsVal

    def __len__(self):
        return len(self.images_psnrs)

if __name__ == "__main__":
    print("imageloader started")
