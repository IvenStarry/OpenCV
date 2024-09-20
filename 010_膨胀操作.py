import cv2
import numpy as np

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('related_data/iven.jpg')
show('img', img)

# todo np.ones()创建初值全为1的数组 默认类型为float64
# * int8就是用8个比特位来保存整数，第一位用来表示符号，索引int8的整数范围是-127到127；
# * uint8表示无符号整数，没有符号位，8个比特位全部用来表示整数，所以数据范围是0到255。
kernel = np.ones((2, 2), np.uint8)
# // print(kernel)
# ? 膨胀操作
# todo cv2.dilate膨胀操作 内容同上
dilate = cv2.dilate(img, kernel, iterations=1)
show('dilate', dilate)

pie = cv2.imread('related_data/pie.png')
kernel = np.ones((30, 30), np.uint8)
dilate1 = cv2.dilate(pie, kernel, iterations=1)
dilate2 = cv2.dilate(pie, kernel, iterations=2)
dilate3 = cv2.dilate(pie, kernel, iterations=3)
# todo np.hstack()将参数元组的元素数组按水平方向进行叠加
# todo vstack() 函数数值 dstack()第三维度叠加
# https://blog.csdn.net/helloword111222/article/details/120577720?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170472175216800192214776%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170472175216800192214776&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-5-120577720-null-null.142^v99^pc_search_result_base2&utm_term=dstack&spm=1018.2226.3001.4187
res1 = np.hstack((dilate1, dilate2, dilate3))
show('result h', res1)

# vstack
res2 = np.vstack((dilate1, dilate2, dilate3))
show('result v', res2)
print(res2)

# ! 图像的三维数组表示
#https://blog.csdn.net/weixin_48306625/article/details/107532968?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170472448916800213064883%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170472448916800213064883&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-107532968-null-null.142^v99^pc_search_result_base2&utm_term=%E4%B8%89%E7%BB%B4%E5%9B%BE%E5%83%8F%E6%95%B0%E7%BB%84&spm=1018.2226.3001.4187
# dstack
a = [[10, 11, 12], [13, 14, 15], [13, 14, 15]]
b = [[1, 2, 3], [4, 5, 6], [4, 5, 6]]
d = [[20, 21, 22], [23, 24, 25], [23, 24, 25]]
c = np.dstack((a, b, d))
print('a = ', a, '\nb = ', b, '\nd = ', d, '\nc = ', c)
print(c.shape)
# print(type(c))
print(c[:, :, 2])
# // show('Mix', c)
# result:
# a =  [[10, 11, 12], [13, 14, 15], [13, 14, 15]]
# b =  [[1, 2, 3], [4, 5, 6], [4, 5, 6]]
# d =  [[20, 21, 22], [23, 24, 25], [23, 24, 25]]
# c [[[10  1 20]
#   [11  2 21]
#   [12  3 22]]

#  [[13  4 23]
#   [14  5 24]
#   [15  6 25]]

#  [[13  4 23]
#   [14  5 24]
#   [15  6 25]]]

# ! 第三通道
# [[20 21 22]
#  [23 24 25]
#  [23 24 25]]