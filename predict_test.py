'''
predict recursively all images in a file
'''

from matplotlib import cm as c
from model import CSRNet
import torch
from matplotlib import pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import h5py
import os
import cv2
import time


# initialize model
PATH = './pretrained_models/PartAmodel_best.pth.tar'
model = CSRNet().cuda()
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['state_dict'])



transform=transforms.Compose([
                      transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                  ])

imgs_path = './ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data/images'
imgs = os.listdir(imgs_path)
# could use while(True): for live video feed

start1 = time.time()
for image in imgs:
    start = time.time()

    img = transform(Image.open(os.path.join(imgs_path, image)).convert('RGB')).cuda()

    # og
    # img = transform(Image.open('./ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data/images/IMG_1.jpg').convert('RGB')).cuda()

    with torch.no_grad():
        output = model(img.unsqueeze(0))

    print("Predicted Count : ",int(output.detach().cpu().sum().numpy()))
    print("%.3f" % (time.time() - start))
    # temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
    # plt.imshow(temp,cmap = c.jet)
    # plt.show()

print("FPS avg: %.1f" % (len(imgs)/(time.time()-start1)))
'''
    # show target result
    image = image.replace("jpg", "h5")
    temp = h5py.File(os.path.join('./ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data/ground_truth/', image), 'r')
    temp_1 = np.asarray(temp['density'])
    # plt.imshow(temp_1,cmap = c.jet)
    print("Original Count : ",int(np.sum(temp_1)) + 1)
    # plt.show()
    #
    # print("Original Image")
    # plt.imshow(plt.imread('./ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data/images/IMG_1.jpg'))
    # plt.show()
'''