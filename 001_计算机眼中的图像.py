import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('related_data/cat.jpg')
# *三通道 三组矩阵
print(img)

# ? 基础显示
cv2.imshow('cat', img)
# todo 等待时间 0在键盘上任意操作关闭窗口 x等待x ms 再关闭
# cv2.waitKey(1000)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ? 自定义函数 减少操作
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ? 图像大小 通道数
print(img.shape)

# ? 直接以灰度图像打开
img = cv2.imread('related_data/cat.jpg', cv2.IMREAD_GRAYSCALE)
# * 单通道 一组矩阵
print(img)

# ? 保存
cv2.imwrite('related_data/mycat.png',img)

# ? 输出
# * 图像格式numpy.ndarray
print(type(img))
# * 图像大小 像素值
print(img.size)
# * 数据类型
print(img.dtype)