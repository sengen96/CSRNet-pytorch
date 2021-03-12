'''
predict a single image
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
PATH = './pretrained_models/partBmodel_best.pth.tar'
model = CSRNet().cuda()
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['state_dict'])



transform=transforms.Compose([
                      transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                  ])

img_path = './ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data/images/IMG_27.jpg'

image = Image.open(img_path)
image = image.resize((1366,768))
# image.show()

img = transform(image.convert('RGB')).cuda()

with torch.no_grad():
    output = model(img.unsqueeze(0))

count = int(output.detach().cpu().sum().numpy())
print("Predicted Count : ", count)

'''
temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
# print(np.array(image).shape[0], np.array(image).shape[1])
temp = cv2.resize(temp, (np.array(image).shape[1], np.array(image).shape[0]))

plt.imshow(temp)
plt.imshow(image, cmap = c.jet)
plt.imshow(temp, cmap=c.jet, alpha=1.0)
plt.show()
'''
cv_img = np.array(image)
cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
temp = cv2.resize(temp, (cv_img.shape[1], cv_img.shape[0]))
# scaling + convert to unint8 for image reading
scale = 255/temp.max()
temp = (temp*scale).astype(np.uint8)
temp = cv2.applyColorMap(temp, cv2.COLORMAP_JET)
overlay = cv2.addWeighted(cv_img, 0.5, temp, 0.5, 0.0)
cv2.imshow('overlay', overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()




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