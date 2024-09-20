import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('related_data/lena.jpg', 0)

# ? 傅里叶变换
# * opencv执行傅里叶必须转换为np.float32格式
img_float32 = np.float32(img)

# todo cv2.dft()执行傅里叶变换 # src：输入图像，可以为实数矩阵或者复数矩阵 flags：转换标志
# （如DFT_COMPLEX_OUTPUT，对一维或二维实数数组正变换，输出一个同样尺寸的复数矩阵）
# （DFT_REAL_OUTPUT，对一维或二维复数数组反变换，通常输出同样尺寸的复矩阵）
# 返回结果：是双通道的，第一个的结果是虚数部分，第二个通道的结果是实数部分
# https://blog.csdn.net/qq_45832961/article/details/124175063
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
# todo np.fft.fftshift() 将低频部分转换到中间位置
dft_shift = np.fft.fftshift(dft)
# todo cv2.magnitude() 将复数结果转换为幅值 res=sqrt(im^2 + re^2)转为成灰度图可以表达的形式
# * 因此结果值比较小不宜观察 通过映射公式放大细节 
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('input'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('magnitude_spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
#越靠近中心的就是低频部分 朝两边发散的是高频