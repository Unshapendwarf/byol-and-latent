import random
import os

target_num = 1000
write_name = "gallery_user_{}.txt".format(target_num)

x1_path = "/home/hong/dataset/gallery_dataset/users_1_40/copy_user_10/all/"
# x1_path = "/home/hong/dataset/gallery_dataset/users_1_40/copy_user_10/cropped_x1/user_"+str(1)

x1_list = os.listdir(x1_path)

train_path = "/home/hong/dataset/gallery_dataset/users_1_40/copy_user_10/all/"
# train_path = "/home/hong/dataset/gallery_dataset/users_1_40/copy_user_10/cropped_x1/user_"+str(target_num)
train_list = os.listdir(train_path)

center_list_jpg = [ file for file in x1_list if file.endswith("_055.jpg")]

t1_list_jpg = [ file for file in train_list if file.endswith("_001.jpg")]
t5_list_jpg = [ file for file in train_list if file.endswith("_005.jpg")]
t10_list_jpg = [ file for file in train_list if file.endswith("_010.jpg")]
t50_list_jpg = [ file for file in train_list if file.endswith("_050.jpg")]
t51_list_jpg = [ file for file in train_list if file.endswith("_051.jpg")]
t91_list_jpg = [ file for file in train_list if file.endswith("_091.jpg")]
t96_list_jpg = [ file for file in train_list if file.endswith("_096.jpg")]
t100_list_jpg = [ file for file in train_list if file.endswith("_100.jpg")]
# t9_list_jpg = [ file for file in train_list if file.endswith("9.jpg")]
'''
0 1 2 3 4 5 6 7 8 9 10
1 1 2 3 4 5 6 7 8 9 10
2 1 2 3 4 5 6 7 8 9 10
3 1 2 3 4 5 6 7 8 9 10
4 1 2 3 4 5 6 7 8 9 10

5 1 2 3 4 5 6 7 8 9 10
6 1 2 3 4 5 6 7 8 9 10
7 1 2 3 4 5 6 7 8 9 10
8 1 2 3 4 5 6 7 8 9 10
9 1 2 3 4 5 6 7 8 9 10
'''


f = open("cross"+write_name, "w")
nonstring = "40"

for i in range(100):
    center = random.choice(center_list_jpg)
    
    for ind in range(200):
        j = random.choice(t1_list_jpg)
        data = "{} {} {} {}\n".format(center, j, nonstring, nonstring)
        f.write(data)
    # for ind in range(200):
    #     j = random.choice(t10_list_jpg)
    #     data = "{} {} {} {}\n".format(center, j, nonstring, nonstring)
    #     f.write(data)
    for ind in range(200):
        j = random.choice(t50_list_jpg)
        data = "{} {} {} {}\n".format(center, j, nonstring, nonstring)
        f.write(data)
    for ind in range(200):
        j = random.choice(t51_list_jpg)
        data = "{} {} {} {}\n".format(center, j, nonstring, nonstring)
        f.write(data)
    for ind in range(200):
        j = random.choice(t91_list_jpg)
        data = "{} {} {} {}\n".format(center, j, nonstring, nonstring)
        f.write(data)

    for ind in range(200):
        j = random.choice(t100_list_jpg)
        data = "{} {} {} {}\n".format(center, j, nonstring, nonstring)
        f.write(data)
    # for j in t9_list_jpg:
    #     data = "{} {} {} {}\n".format(i, j, nonstring, nonstring)
    #     f.write(data)

f.close()