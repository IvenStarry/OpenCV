import cv2
import numpy as np

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('related_data/dog.jpg')
show('original', img)
print('orgin = ', img.shape)

# ? 高斯金字塔
# todo cv2.pyrDown() cv2.pyrUp()高斯金字塔 向下/向上取样
down = cv2.pyrDown(img)
show('down', down)
print('down = ', down.shape)

up1 = cv2.pyrUp(img)
show('up1', up1)
print('up1 = ', up1.shape)

up2 = cv2.pyrUp(up1)
show('up2', up2)
print('up2 = ', up2.shape)

# * 先上采样后下采样 得到的图像和原来不一致，会更模糊
up = cv2.pyrUp(img)
print('up = ', up.shape)
up_down =cv2.pyrDown(up)
show('up_down', down)
print('up_down = ', up_down.shape)
Mix = np.hstack((img, up_down))
show('Mix', Mix)

# ? 拉普拉斯金字塔
# ! up_down down_up 先后顺序和图像大小关系
# 原始图像（245, 247, 3) 先up (490, 494, 3) 再down (245, 247, 3)
# 原始图像（245, 247, 3) 先down (123, 124, 3) 再up (246, 248, 3)
# * 原始图像长高为奇数 up翻倍为偶数 down减去偶数层是奇数
# * 原始图像长高为奇数 down减去偶数层还是奇数 up翻倍为偶数
# * 因此 先down后up不能直接减去原图像 大小不一致 可以resize原图像或者down_up图像
# down = cv2.pyrDown(img)
# print('down = ', down.shape)
# down_up = cv2.pyrUp(down)
# print('up&down = ', down_up.shape)
# res = img - down_up
# show('result', res)

img = cv2.resize(img, (246, 248))
down = cv2.pyrDown(img)
print('down = ', down.shape)
down_up = cv2.pyrUp(down)
print('up&down = ', down_up.shape)
res = img - down_up
show('result', res)