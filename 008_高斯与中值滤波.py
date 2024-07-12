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
blur = cv2.blur(img, (3, 3))

# ? 方框滤波 和均值滤波基本一样 多一个参数
boxTrue = cv2.boxFilter(img, -1, (3,3), normalize=True)
boxFalse = cv2.boxFilter(img, -1, (3,3), normalize=False)
Mix = np.hstack((img, boxTrue, boxFalse))

# ? 高斯滤波
# todo cv2.GaussianBlur() 第三个值为标准差sigma
# https://blog.csdn.net/sunjintaoxxx/article/details/121420594?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170480771716800225526947%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170480771716800225526947&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-121420594-null-null.142^v99^pc_search_result_base2&utm_term=cv2.GaussianBlur&spm=1018.2226.3001.4187
gaussian = cv2.GaussianBlur(img, (5, 5), 1)
show('guassian', gaussian)

# ? 中值滤波
# todo cv2.medianBlur() 5代表5*5的卷积核
median = cv2.medianBlur(img, 5)
show('median', median)

# ! cv2.imread() 读取的是BGR格式 使用matlab输出图像使用cv2.cvtColor()转换RGB格式
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
boxTrue = cv2.cvtColor(boxTrue, cv2.COLOR_BGR2RGB)
boxFalse = cv2.cvtColor(boxFalse, cv2.COLOR_BGR2RGB)
gaussian = cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)
median = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)

plt.subplot(231), plt.imshow(img), plt.title('orgin')
plt.subplot(232), plt.imshow(blur), plt.title('blur')
plt.subplot(233), plt.imshow(boxTrue), plt.title('boxTrue')
plt.subplot(234), plt.imshow(boxFalse), plt.title('boxFalse')
plt.subplot(235), plt.imshow(gaussian), plt.title('gaussian')
plt.subplot(236), plt.imshow(median), plt.title('median')
plt.show()