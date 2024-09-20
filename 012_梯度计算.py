import cv2
import numpy as np

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

pie = cv2.imread('related_data/pie.png')
show("pie", pie)

# ? 梯度运算 膨胀-腐蚀
kernel = np.ones((7, 7), np.uint8)
dilate = cv2.dilate(pie, kernel, iterations=5)
erosion = cv2.erode(pie, kernel, iterations=5)

res = np.hstack((dilate, erosion))
show('res', res)

gradient = cv2.morphologyEx(pie, cv2.MORPH_GRADIENT, kernel)
show('gradient', gradient)