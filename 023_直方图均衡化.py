import cv2
import matplotlib.pyplot as plt
import numpy as np

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ? 直方图均衡化
img = cv2.imread('related_data/lena.jpg', 0)
# todo cv2.equalizeHist()均衡化函数 原理看md
equ = cv2.equalizeHist(img)
plt.subplot(121),plt.hist(img.ravel(), 256),plt.title('img')
plt.subplot(122),plt.hist(equ.ravel(), 256),plt.title('equ')
plt.show()

res = np.hstack((img, equ))
show('result', res)

# 有的图像均衡化会丢失细节 最好分块均衡化 可能效果会更好 如自适应均衡化
img = cv2.imread('related_data/clahe.jpg', 0)
# todo cv2.equalizeHist()均衡化函数 原理看md
equ = cv2.equalizeHist(img)
plt.subplot(121),plt.hist(img.ravel(), 256),plt.title('img')
plt.subplot(122),plt.hist(equ.ravel(), 256),plt.title('equ')
plt.show()

res = np.hstack((img, equ))
show('result', res)

# ? 自适应均衡化
# todo cv2.createCLAHE(clipLimit,tileGridSize) 噪声影响若过大则多余重新分配 可以看md或网址
# clipLimit:限制对比度的阈值，默认为40，直方图中像素值出现次数大于该阈值，多余的次数会被重新分配
# tileGridSize:图像会被划分的size， 如tileGridSize=(8,8),默认为(8,8)
# https://blog.csdn.net/m0_45805664/article/details/107615211?ops_request_misc=&request_id=&biz_id=102&utm_term=cv2.createCLAHE&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-4-107615211.142^v99^pc_search_result_base2&spm=1018.2226.3001.4187
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# todo 使用apply()将自适应均衡化操作作用于原图
res_clahe = clahe.apply(img)
res = np.hstack((img, equ, res_clahe))
show('Adaptive equalization', res)