import cv2
# import pandas as pd
# import math
import numpy as np


def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ? Sobel 圆
img = cv2.imread('related_data/pie.png', cv2.IMREAD_GRAYSCALE)
show('ORGINAL', img)

# todo cv2.Sobel(src, ddepth, dx, dy, ksize)
# todo ddepth图像深度(default = -1) 输出深度=输入深度
# todo dx dy 算x、y方向的梯度 ksize Sobel算子大小
# ! 右-左 因为梯度取值0~255 若为负数会直接被截断成0 使用cv2.CV_64F 可以使得位数变多 可以表示负数
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
# * 圆左半边缘 白-黑 255-0=255 圆右半边缘 黑-白 0-255=-255
show('sobelx', sobelx)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
# print(sobelx)
#np.set_printoptions(threshold=math.inf)
# outputfile = open('orgin.txt', 'w')
# # 显示所有列
# pd.set_option('display.max_columns', None)
# # 显示所有行
# pd.set_option('display.max_rows', None)
# print(sobelx, file=outputfile)
# outputfile.close()
# todo cv2.convertScaleAbs() 线性变换 可以先缩放再计算绝对值（此处无缩放）
sobelx = cv2.convertScaleAbs(sobelx)
# ! 圆只有黑和白不需要归一化 但信用卡数字识别是灰度图需要归一化
# show('sobelx', sobelx)
# (minVal, maxVal) = (np.min(sobelx), np.max(sobelx))
#  归一化！
# sobelx = (255 * ((sobelx - minVal) / (maxVal - minVal)))
# print(sobelx)
show('sobelx', sobelx)
