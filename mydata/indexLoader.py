from torch.utils.data import Dataset
import cv2

class myDataUnit:
    def __init__(self, filepath, name):
        self.filepath = filepath
        self.imgname = name

    def getpath(self):
        return self.filepath
        
    def getname(self):
        return self.imgname


class nolabel_dataset(Dataset):
    def __init__(self, imageunits, transform=False):
        super(nolabel_dataset, self).__init__()
        self.imageunits = imageunits
        self.transform = transform

    def __getitem__(self, index):
        image_path, floated_name = self.imageunits[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            image = self.transform(image)
        return image, floated_name

    def __len__(self):
        return len(self.imageunits)


if __name__=="__main__":
    print("testIndex")

