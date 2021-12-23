import os
from PIL import Image
import pillow_avif
import cv2

file_dir = "/home/hong/dir1/final_eval/LR/"
file_list = os.listdir(file_dir)
file_list = [ file for file in file_list if file.endswith(".AVIF")]

save_dir = "/home/hong/dir1/final_eval/lr_png/"
png_dir = "/home/hong/dir1/final_eval/HR/"

write_file = open("result_1_psnr.txt", "w")

for file in file_list:
    file_name = file[:-5]
    file_path = file_dir + file_name
    save_path = save_dir +file_name

    avifimg = Image.open(file_path+".AVIF")
    avifimg.save(save_path +".png", 'PNG')
    
    # png_path = png_dir + file_name
    # img1 = cv2.imread(save_path + ".png")
    # img2 = cv2.imread(png_path + ".png")
    # psnr = cv2.PSNR(img1, img2)
    # # print(img1.shape, img2.shape)
    # linestr = "{:<8} {}\n".format(file_name, psnr)
    # # print(linestr)
    # write_file.write(linestr)

write_file.close()