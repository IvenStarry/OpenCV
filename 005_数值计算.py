import cv2
import matplotlib.pyplot as plt

img_cat = cv2.imread('related_data/cat.jpg')
img_dog = cv2.imread('related_data/dog.jpg')

# * 所有像素全部加10
img_cat2 = img_cat + 10
print(img_cat[:5, :, 0])
# todo 矩阵直接相加 结果超过255 自动对256取余
print(img_cat2[:5, :, 0])
print((img_cat + img_cat2)[:5, :, 0])
# todo cv2.add()矩阵相加超过255 直接赋值255
print((cv2.add(img_cat, img_cat2))[:5, :, 0])

# 图像大小不一致不可以直接相加 operands could not be broadcast together with shapes (245,247,3) (235,236,3)
# *img.shape[x, y, z] x行数 y列数 z通道数
#// Mix = img_dog + img_cat

# * dsize = (width, height) 区别于shape
img_dog1 = cv2.resize(img_dog, (236, 235))
# // 直接相加 Mix = img_dog + img_cat
# todo cv2.addWeighted(src1, alpha, src2, beta, gamma,dst,dtype) alpha beta 第1、2张图像的像素权重 gamma融合后再加的像素值(double型) dst 输出图像  
# todo dtype：目标图像的数组深度，即单个像素值的位数，默认为None。灰度图像，每个像素颜色占用1个字节8位，则称图像深度为8位，而RGB的彩色图像占用3字节，图像深度为24位
# todo 图像深度  https://blog.csdn.net/LaoYuanPython/article/details/109569968?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170468859316800180659546%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170468859316800180659546&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-109569968-null-null.142^v99^pc_search_result_base2&utm_term=%E5%9B%BE%E5%83%8F%E7%9A%84%E6%B7%B1%E5%BA%A6&spm=1018.2226.3001.4187
# *数据单位 bit byte word 字节
Mix1 = cv2.addWeighted(img_cat, 0.4, img_dog1, 0.6, 0)
plt.imshow(Mix1)
plt.title('Mix1')
plt.show()
# // plt.imshow(img_cat)
# // plt.show()
# *不指定新图像大小 给出拉伸倍数
img_dog2 = cv2.resize(img_dog, (0, 0), fx=3, fy=1)
cv2.imshow('resize dog', img_dog2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ! cv2.imshow()和plt.imshow()的区别
# 什么时候使用plt.show()，什么时候用cv2.imshow()?
# 如果需要展示读入的图像，或者展示对读入图像进行一系列操作后的图像时，使用cv2.imshow()
# 如果不需要展示原始图像，而是绘制一张新的图像，使用plt.imshow()
# 其实两者都可以，但要注意的是opencv是BGR通道，plt默认RGB通道，若使用cv2.imread()读入图像，用plt.imshow()展示原始图像或者展示对读入图像进行一系列操作后的图像时，需要进行通道转换。
# 在展示灰度图像时，cv2.imshow(‘gray’, gray)
# plt.imshow(gray,cmap=‘gray’), plt.title(‘gray’)
# 原文链接：https://blog.csdn.net/qq_38132105/article/details/105555514