import cv2
import matplotlib.pyplot as plt
import numpy as np

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('related_data/lena.jpg', 0)
# todo cv2.calcHist()计算直方图 输入 原图像 通道 掩膜 bin区间大小 像素值范围 
# todo 这个函数的参数必须用[ ]括起来 
hist = cv2.calcHist([img], [0], None, [256], [0,256])
print(hist.shape)

# ? plt.hist() 由原图像直接输出灰度直方图
# ! 读取的图像是一个二维数组 需要降维！
# todo img.ravel()将二维数组降成一位数组
# * bin值为16 划分成16个子集
plt.hist(img.ravel(), 16)
plt.show()

# ? plt.plot() 由cv2.calcHist()得到数组 输出RGB直方图
img = cv2.imread('related_data/lena.jpg')
color = ('b', 'g', 'r')
# todo enumerate()将一个可遍历iterable数据对象(如list列表、tuple元组或str字符串)组合为一个索引序列
# todo 同时列出数据和数据下标，一般用在for循环当中 返回 索引 数值
print(list(enumerate(color)))
for i,col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0,255])
    plt.plot(histr, color = col)
    plt.xlim([0,256])
plt.show()

# ? 使用mask
# * img.shape[:2]就是图像高和宽 img.shape(263, 263, 3) [:2]取前两个
img = cv2.imread('related_data/lena.jpg', 0)
mask = np.zeros(img.shape[:2], np.uint8)
# * 先高后宽
mask[50:200, 50:200] = 255
# show('mask', mask)
# // print(mask.shape)
# // print(img.shape)
# todo 与操作 bitwise_and(src1, src2, dst=None, mask=None)
# todo src1 src2 参与运算的两个图像 dst 输出数组 mask掩膜
# openCV 位操作 https://blog.csdn.net/m0_51545690/article/details/123956698?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170515555116800227474815%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170515555116800227474815&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-123956698-null-null.142^v99^pc_search_result_base2&utm_term=cv2.bitwise_and&spm=1018.2226.3001.4187
mask_img = cv2.bitwise_and(img, img, mask=mask)

hist_full = cv2.calcHist([img], [0], None, [256], [0,255])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0,255])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(mask_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0, 255])
plt.show()