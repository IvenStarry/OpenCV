import cv2
import numpy as np

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('related_data/iven.jpg')
show('img', img)

# ? 开运算 先腐蚀 再膨胀
# todo cv2.morphologyEx() 形态学操作
kernel = np.ones((2, 2), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
show('opening', opening)

# ? 闭运算 先膨胀 再膨胀
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
show('closing', closing)