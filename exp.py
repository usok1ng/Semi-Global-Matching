import os
import cv2
import matplotlib.pyplot as plt

img_path = 'input'
img_list = os.listdir(img_path)

for img_name in img_list:
    pre_img = os.path.join(img_path, img_name)
    img = cv2.imread(pre_img, cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    print(img[0][1])
    plt.imshow(img, cmap = 'gray')

plt.show()