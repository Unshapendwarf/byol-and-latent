import random

class psnrDataUnit:
    def __init__(self, name2, psnr1, psnr2):
        self.img2name = name2 #image2 name in string
        self.psnr1 = float(psnr1) # psnr1 with float 
        self.psnr2 = float(psnr2) # psnr2 with float
    
    def getimg2(self):
        return self.img2name;

    def getpsnr1(self):
        return self.psnr1

    def getpsnr2(self):
        return self.psnr2

    def getonetwo(self):
        return self.psnr1+self.psnr2

def testIndex(file_path):
    indexDict = dict()
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
    print(len(indexDict[indexDictKeys[0]]))
    print(indexDict[indexDictKeys[0]][0].getimg2())
    print(len(indexDictKeys))

if __name__=="__main__":
    print("testIndex")
    csv_data_path = "/home/hong/dir1/PyTorch-BYOL/writing2.txt"
    testIndex(csv_data_path)
# close index file

