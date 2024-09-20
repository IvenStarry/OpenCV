import cv2
import numpy as np

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('related_data/iven.jpg')
show('img', img)

# ? 腐蚀操作
# todo np.ones()创建初值全为1的数组 默认类型为float64
# * int8就是用8个比特位来保存整数，第一位用来表示符号，索引int8的整数范围是-127到127；
# * uint8表示无符号整数，没有符号位，8个比特位全部用来表示整数，所以数据范围是0到255。
kernel = np.ones((2, 2), np.uint8)
# // print(kernel)
# todo cv2.erode腐蚀操作 img原图像 kernel算子大小 iteration迭代次数
# * 迭代次数越多 白色区域腐蚀越多
erosion = cv2.erode(img, kernel, iterations=1)
show('erosion', erosion)