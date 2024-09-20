import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('related_data/lenaNoise.png')
show('orginal',img)

# ? 均值滤波
# todo cv2.blur()均值滤波
blur = cv2.blur(img, (3, 3))
show('blur', blur)

# ? 方框滤波 和均值滤波基本一样 多一个参数
# todo cv2.boxFilter() -1表示输入和输出图像结果再颜色通道数上一致 
# ! normalize=True 加归一化操作(即加和后除以9)，和均值滤波一致 
# ! normalize=False 不归一化(即加和后不除以9) 可能会越界255 直接赋值255
boxTrue = cv2.boxFilter(img, -1, (3,3), normalize=True)
show('box True', boxTrue)

boxFalse = cv2.boxFilter(img, -1, (3,3), normalize=False)
show('box False', boxFalse)

Mix = np.hstack((img, boxTrue, boxFalse))
show('result', Mix)