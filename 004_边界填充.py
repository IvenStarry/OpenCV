import cv2
import matplotlib.pyplot as plt

img = cv2.imread('related_data/dog.jpg')
# 设置填充的边缘大小
top_size, bottom_size, left_size, right_size = ( 20, 20, 20, 20)

# todo copyMakeBorder() 边界填充函数 区分类型
replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size , right_size, borderType= cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size , right_size, borderType= cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size , right_size, borderType= cv2.BORDER_REFLECT_101)
warp = cv2.copyMakeBorder(img, top_size, bottom_size, left_size , right_size, borderType= cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size , right_size, borderType= cv2.BORDER_CONSTANT, value=0)

plt.subplot(231), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
# * 复制法 直接复制最边缘像素
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
# * 反射法1 fedcba | abcdefgh | hgfedc （a h为边界 倒序扩充）
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
# * 反射法2 fedcb | abcdefgh | gfedc
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
# * 外包装法 abcdefgh | abcdefgh | abcdefg
plt.subplot(235), plt.imshow(warp, 'gray'), plt.title('WRAP')
# * 常量法 直接输入一个常量填充 value = ?
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')

# 显示
plt.show()