import cv2
import numpy as np

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ? Sobel 圆
img = cv2.imread('related_data/pie.png', cv2.IMREAD_GRAYSCALE)
show('ORGINAL', img)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
# * 圆左半边缘 白-黑 255-0=255 圆右半边缘 黑-白 0-255=-255
show('sobelx', sobelx)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
show('sobelx', sobelx)

sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)

show('sobely', sobely)

# 分别计算x和y 在求和
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
show('sobelxy', sobelxy)

# *不建议直接用Sobel算xy方向
sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
sobelxy = cv2.convertScaleAbs(sobelxy)
show('sobelxy', sobelxy)

# ? Sobel lena
img = cv2.imread('related_data/lena.png', cv2.IMREAD_GRAYSCALE)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
show('sobelx', sobelx)

sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
show('sobely', sobely)

# 分别计算x和y 在求和
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
show('sobelxy', sobelxy)

# *不建议直接用Sobel算xy方向 可能融合效果不好 有重影等现象 建议上一种分开计算
sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
sobelxy = cv2.convertScaleAbs(sobelxy)
show('sobelxy', sobelxy)