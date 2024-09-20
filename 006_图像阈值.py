import cv2
import matplotlib.pyplot as plt

img = cv2.imread('related_data/cat.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# todo thresh,result=cv2.threshold (src, thresh, maxval, type)
# thresh 阈值 maxval最大灰度值 type二值化类型
ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['Origin', 'binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

# // print(range(6))
# * range(6)  从0-6 不包括6
for i in range(6):
    images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    # todo plt.xticks([]) plt.yticks([]) 传入空列表,不显示x轴刻度
    plt.xticks([])
    plt.yticks([])

plt.show()
