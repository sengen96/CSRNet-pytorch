# A
# train       = 300
# train w val = 240  (60 left for val?)
# test        = 182

# B
# train       = 400
# train w val = 320 (80 left for val?)
# test        = 316

# file to modify
file_path = './json_old/part_A_train_with_val.json'
f = open(file_path, "r")
chunk = str(f.read())
print(chunk)

# modification
chunk = chunk.replace("/home/leeyh/Downloads/Shanghai", "./ShanghaiTech_Crowd_Counting_Dataset")

# file to write modified to
fp = './part_A_train_with_val.json'
f = open(fp, "w")
f.write(chunk)
f.close()

# print chunk out for checking
repl = ["[", "]", "'", " "]
for char in repl:
    chunk = chunk.replace(char, "")
dirs = chunk.split(",")
for dir in dirs:
    print(dir)
print(len(dirs))



'''
import os


import os
# part B
dataset = './ShanghaiTech_Crowd_Counting_Dataset/part_A_final'
# test/train set
test_set = dataset + '/train_data'
images_dir = test_set+ '/images'
images = os.listdir(images_dir)
fp = './part_A_train.json'
f = open(fp, "w")
for i, image in enumerate(images):
    img_dir = images_dir + '/' + image
    # print(img_dir)
    if i == 0:
        f.write('["' + img_dir + '", ')
    elif i == (len(images) - 1):
        f.write('"' + img_dir + '"]')
    elif i!=0 and i!=(len(images) - 1):
        f.write('"' + img_dir + '", ')

'''
