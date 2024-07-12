import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('related_data/lena.png', cv2.IMREAD_GRAYSCALE)

# cv2.Canny() img minval maxval
v1 = cv2.Canny(img, 80, 150)
v2 = cv2.Canny(img, 50, 100)

res = np.hstack((v1, v2))
show('result', res)

img = cv2.imread('related_data/car.png', cv2.IMREAD_GRAYSCALE)
v1 = cv2.Canny(img, 120, 250)
v2 = cv2.Canny(img, 50, 100)

res = np.hstack((v1, v2))
show('result', res)

plt.subplot(121), plt.title('car min=120,max=250'), plt.imshow(v1, 'gray')
plt.subplot(122), plt.title('car min=50,max=100'), plt.imshow(v2, 'gray')
plt.show()