import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('related_data/lena.jpg', 0)
img_float32 = np.float32(img)
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# ? 低通滤波器 
rows, cols = img.shape
# 中心位置
crow, ccol = int(rows / 2), int(cols / 2)

# * 为了划分低频和高频的位置
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow - 30 : crow + 30, ccol-30 : ccol +30] = 1

# IDFT 逆变化
# * 获取低通掩膜
fshift = dft_shift * mask
# todo np.fft.ifftshift() 把原来放在中间的低频 重新还原
f_ishift = np.fft.ifftshift(fshift)
# todo cv2.idft() 傅里叶逆变换
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('input'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('res'), plt.xticks([]), plt.yticks([])
plt.show()

# ? 高通滤波器
mask = np.ones((rows, cols, 2), np.uint8)
mask[crow - 30 : crow + 30, ccol-30 : ccol +30] = 0

# * 获取高通掩膜
fshift = dft_shift * mask
# todo np.fft.ifftshift() 把原来放在中间的低频 重新还原
f_ishift = np.fft.ifftshift(fshift)
# todo cv2.idft() 傅里叶逆变换
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('input'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('res'), plt.xticks([]), plt.yticks([])
plt.show()