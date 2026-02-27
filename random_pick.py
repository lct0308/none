import os
from random import sample

test_txt = "/data1/pbw/Project/test_seq/crowded.txt" #crowded/final_test
image_filenames = []
with open(test_txt, 'r') as list_file:
    for file in list_file:
        file = file.split()
        img_line = file[0]
        image_filenames.append(img_line)

random_num = 558 # 1102, 1048, 1083
image_filenames = sample(image_filenames, random_num)

save_path = f"/data1/pbw/Project/test_seq/crowded_{random_num}.txt"
for image in image_filenames:
    with open(save_path, 'a') as f:
        f.write(image + '\n')
        f.close()