import cv2
import numpy as np

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('related_data/iven.jpg')
show('img', img)

# ? 礼帽 原始图像-开运算处理图像
kernel = np.ones((2, 2), np.uint8)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# ? 黑帽 闭运算-原始图像
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

img = cv2.resize(img, (500,200))
tophat = cv2.resize(tophat, (500,200))
blackhat = cv2.resize(blackhat, (500,200))

result = np.vstack((img, tophat, blackhat))
show('tophat blackhat', result)