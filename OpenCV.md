# OpenCV 

github：https://github.com/IvenStarry  

学习视频网站：B站咕泡唐宇迪
 https://www.bilibili.com/video/BV1EG4y1B7Wz/?spm_id_from=333.999.0.0&vd_source=6fd71d34326a08965fdb07842b0124a7

VSCode 快捷键 https://zhuanlan.zhihu.com/p/66826924

## 第一章 图像基本操作
### 计算机眼中的图像
```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('related_data/cat.jpg')
# *三通道 三组矩阵
print(img)

# ? 基础显示
cv2.imshow('cat', img)
# todo 等待时间 0在键盘上任意操作关闭窗口 x等待x ms 再关闭
# cv2.waitKey(1000)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ? 自定义函数 减少操作
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ? 图像大小 通道数
print(img.shape)

# ? 直接以灰度图像打开
img = cv2.imread('related_data/cat.jpg', cv2.IMREAD_GRAYSCALE)
# * 单通道 一组矩阵
print(img)

# ? 保存
cv2.imwrite('related_data/mycat.png',img)

# ? 输出
# * 图像格式numpy.ndarray
print(type(img))
# * 图像大小 像素值
print(img.size)
# * 数据类型
print(img.dtype)
```

### 视频的读取与处理
```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

vc = cv2.VideoCapture('related_data/002_test.mp4')

# ? 检测打开视频是否正确
if vc.isOpened():
    # todo vc.read()打开一帧 open是一个bool类型判断有无读取到文件 frame为这一帧图像
    open, frame = vc.read()
else:
    open = False
    print('False')

cv2.namedWindow('result', cv2.WINDOW_NORMAL)
# todo 调整窗口大小
cv2.resizeWindow('result', 800, 450)
# cv2.setWindowProperty('result', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#当open为True
while open:
    ret, frame = vc.read()
    # *判断是否读完
    if frame is None:
        break
    # * 判断
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('result', gray)
        # todo ASCII码,0xFF == 27代表退出键 点esc直接退出循环 当窗口等待时间5ms且按退出键直接退出循环
        # * cv2.waitKey(delay)参数：1、delay≤0：一直等待按键；2、delay取正整数：等待按键的时间，比如cv2.waitKey(25)，就是等待25（milliseconds）；（视频中一帧数据显示（停留）的时间）
        # * cv2.waitKey(delay)返回值：1、等待期间有按键：返回按键的ASCII码（比如：Esc的ASCII码为27）；2、等待期间没有按键：返回 -1；
        # ! 关键 cv2.waitKey(5) & 0xFF == 27
        # todo 无论如何系统都会等待5ms 等待用户键盘输入 0xFF意思是16进制的FF也就是八位2进制数1111 1111
        # todo 系统中按键对应的ASCII码值并不一定仅仅只有8位，同一按键对应的ASCII并不一定相同（但是后8位一定相同）
        # todo 在其他按键如numclock激活的时候有些按键的ascii值就会超过8位 加入& 0xFF的意义是得到键入的值永远只看后八位 可以有效排除其他键位干扰
        # https://zhuanlan.zhihu.com/p/38443324  https://blog.csdn.net/hao5119266/article/details/104173400
        # todo 0x代表16进制数,0xff表示的数二进制1111 1111 占一个字节.和其进行&操作的数,最低8位,不会发生变化
        # todo 有个数字 0x1234,如果只想将低8位写入到内存中 0x1234&0xff
        # todo 0x1234 表示为二进制 0001001000110100  0xff 表示为二进制 11111111 两个数做与操作，显然将0xff补充到16位，就是高位补0 
        # todo 此时0xff 为 0000000011111111 位与操作&两个位都为1时，结果才为1
        # todo 0x1234 & 0xff 结果0000000000110100
        # https://blog.csdn.net/i6223671/article/details/88924481
        # ASCII码 https://blog.csdn.net/zisongjia/article/details/102465628?ops_request_misc=&request_id=&biz_id=102&utm_term=ascii%E5%A4%9A%E5%B0%91%E4%BD%8D&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-102465628.142^v99^pc_search_result_base2&spm=1018.2226.3001.4187
        # 位操作 https://blog.csdn.net/hzf0701/article/details/117359478?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170463977516800182743959%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170463977516800182743959&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-117359478-null-null.142^v99^pc_search_result_base2&utm_term=%E4%BD%8D%E4%B8%8E%E8%BF%90%E7%AE%97&spm=1018.2226.3001.4187
        if cv2.waitKey(5) & 0xFF == 27:
            break
# 关闭视频文件
vc.release()
# 销毁窗口
cv2.destroyAllWindows()

```

### ROI区域
```python
import cv2

# ? 自定义函数 减少操作
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('related_data/cat.jpg')
# * ROI截取
cat = img[40:180, 40:180]
cv_show('cat',cat)

# todo split() 颜色通道提取 B G R 格式
b, g, r = cv2.split(img)
print('b = ', b,'\n g = ', g,'\n r = ', r)
print('b.shape =', b.shape)

# ! 矩阵操作
# todo b[:,1]的意思是这一行所有数组的第一个元素(b[x,y]表示x行中的数组中的第y个元素，那么:代表所有的意思)
# todo b[1,:]表示序号为1的数组的所有元素
# todo a3[:2,1:,:2] :2表示前两个都要，1：表示除了第1个不要，剩下的都要
# https://blog.csdn.net/weixin_40938312/article/details/131614320?ops_request_misc=&request_id=&biz_id=102&utm_term=img%5B:,%20:,%200%5D&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-131614320.142^v99^pc_search_result_base2&spm=1018.2226.3001.4187
# https://zhuanlan.zhihu.com/p/497136322

# ? 只保留R通道
# todo img.copy()函数复制信息
R_img = img.copy()
R_img[:, :, 0] = 0
R_img[:, :, 1] = 0
cv_show('R channel', R_img)

# ? 只保留G通道
# todo img.copy()函数复制信息
G_img = img.copy()
G_img[:, :, 0] = 0
G_img[:, :, 2] = 0
cv_show('G channel', G_img)

# ? 只保留B通道
# todo img.copy()函数复制信息
B_img = img.copy()
B_img[:, :, 1] = 0
B_img[:, :, 2] = 0
cv_show('B channel', B_img)
```


### 边界填充
```python
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
```

### 数值计算
```python
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
```

## 第二章 阈值与平滑处理
### 图像阈值
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121554584.png)
```python
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
```

### 图像平滑处理
|对位做内积|
均值滤波 与方框函数基本一致
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121555136.png)
```python
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
# todo cv2.blur()均值滤波
blur = cv2.blur(img, (3, 3))
show('blur', blur)

# ? 方框滤波 和均值滤波基本一样 多一个参数
# todo cv2.boxFilter() -1表示输入和输出图像结果再颜色通道数上一致 
# ! normalize=True 加归一化操作(即加和后除以9)，和均值滤波一致 
# ! normalize=False 不归一化(即加和后不除以9) 可能会越界255 直接赋值255
boxTrue = cv2.boxFilter(img, -1, (3,3), normalize=True)
show('box True', boxTrue)

boxFalse = cv2.boxFilter(img, -1, (3,3), normalize=False)
show('box False', boxFalse)

Mix = np.hstack((img, boxTrue, boxFalse))
show('result', Mix)
```

### 高斯与中值滤波
高斯滤波  距离越近的权重大 越远的权重小
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121555954.png)
中值滤波 从小到大排序找中间值
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121555907.png)
```python
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
```

## 第三章 图像形态学操作
### 腐蚀操作
```python
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
```

### 膨胀操作
```python
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
```

### 开运算和闭运算
```python
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
```

### 梯度计算
```python
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
```

### 礼帽和黑帽
```python
import cv2
import numpy as np

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('related_data/iven.jpg')
show('img', img)

# ? 礼帽 原始图像-开运算处理图像
kernel = np.ones((2, 2), np.uint8)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# ? 黑帽 闭运算-原始图像
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

img = cv2.resize(img, (500,200))
tophat = cv2.resize(tophat, (500,200))
blackhat = cv2.resize(blackhat, (500,200))

result = np.vstack((img, tophat, blackhat))
show('tophat blackhat', result)
```

## 第四章 图像梯度计算
### Sobel算子
|对位做内积|
Sobel算子 简洁实用
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121611188.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121611414.png)
```python
import cv2
# import pandas as pd
# import math
import numpy as np


def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ? Sobel 圆
img = cv2.imread('related_data/pie.png', cv2.IMREAD_GRAYSCALE)
show('ORGINAL', img)

# todo cv2.Sobel(src, ddepth, dx, dy, ksize)
# todo ddepth图像深度(default = -1) 输出深度=输入深度
# todo dx dy 算x、y方向的梯度 ksize Sobel算子大小
# ! 右-左 因为梯度取值0~255 若为负数会直接被截断成0 使用cv2.CV_64F 可以使得位数变多 可以表示负数
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
# * 圆左半边缘 白-黑 255-0=255 圆右半边缘 黑-白 0-255=-255
show('sobelx', sobelx)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
# print(sobelx)
#np.set_printoptions(threshold=math.inf)
# outputfile = open('orgin.txt', 'w')
# # 显示所有列
# pd.set_option('display.max_columns', None)
# # 显示所有行
# pd.set_option('display.max_rows', None)
# print(sobelx, file=outputfile)
# outputfile.close()
# todo cv2.convertScaleAbs() 线性变换 可以先缩放再计算绝对值（此处无缩放）
sobelx = cv2.convertScaleAbs(sobelx)
# ! 圆只有黑和白不需要归一化 但信用卡数字识别是灰度图需要归一化
# show('sobelx', sobelx)
# (minVal, maxVal) = (np.min(sobelx), np.max(sobelx))
#  归一化！
# sobelx = (255 * ((sobelx - minVal) / (maxVal - minVal)))
# print(sobelx)
show('sobelx', sobelx)
```
### 梯度计算方法
```python
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
```

### scharr与laplacian算子
Scharr算子 对细节刻画更强
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121612302.png)
Laplacian算子 不分XY X对噪声点敏感 但不一定是好事 一般运用在特殊场合
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121612424.png)
```python
import cv2
import numpy as np

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ?
img = cv2.imread('related_data/lena.png', cv2.IMREAD_GRAYSCALE)
show("orgin",img)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
# cv2.Scharr()不能选择卷积核大小 默认3*3
scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
scharry = cv2.convertScaleAbs(scharry)
scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)

# cv2.Laplacian()无需考虑xy方向 看示意图
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

res = np.hstack((sobelxy, scharrxy, laplacian))
show('result', res)
```

## 第五章 边缘检测
### Canny边缘检测流程
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121618962.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121618367.png)
方向和梯度 Sobel
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121618791.png)
### 非极大值抑制
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121619487.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121619307.png)
即取离梯度方向近的点
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121619644.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121619476.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121620168.png)
### 边缘检测效果
```python
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
```

## 第六章 图像金字塔与轮廓检测
### 卷积操作
https://www.bilibili.com/video/BV1VV411478E/?spm_id_from=333.337.search-card.all.click&vd_source=6fd71d34326a08965fdb07842b0124a7 
### 图像金字塔定义
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121634058.png)
高斯金字塔
向下采样 800*800->400*400->200*200->
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121634013.png)
向上采样 200*200->400*400->800*800->
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121634207.png)
拉普拉斯金字塔
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121635454.png)
### 金字塔制作方法
```python
import cv2
import numpy as np

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('related_data/dog.jpg')
show('original', img)
print('orgin = ', img.shape)

# ? 高斯金字塔
# todo cv2.pyrDown() cv2.pyrUp()高斯金字塔 向下/向上取样
down = cv2.pyrDown(img)
show('down', down)
print('down = ', down.shape)

up1 = cv2.pyrUp(img)
show('up1', up1)
print('up1 = ', up1.shape)

up2 = cv2.pyrUp(up1)
show('up2', up2)
print('up2 = ', up2.shape)

# * 先上采样后下采样 得到的图像和原来不一致，会更模糊
up = cv2.pyrUp(img)
print('up = ', up.shape)
up_down =cv2.pyrDown(up)
show('up_down', down)
print('up_down = ', up_down.shape)
Mix = np.hstack((img, up_down))
show('Mix', Mix)

# ? 拉普拉斯金字塔
# ! up_down down_up 先后顺序和图像大小关系
# 原始图像（245, 247, 3) 先up (490, 494, 3) 再down (245, 247, 3)
# 原始图像（245, 247, 3) 先down (123, 124, 3) 再up (246, 248, 3)
# * 原始图像长高为奇数 up翻倍为偶数 down减去偶数层是奇数
# * 原始图像长高为奇数 down减去偶数层还是奇数 up翻倍为偶数
# * 因此 先down后up不能直接减去原图像 大小不一致 可以resize原图像或者down_up图像
# down = cv2.pyrDown(img)
# print('down = ', down.shape)
# down_up = cv2.pyrUp(down)
# print('up&down = ', down_up.shape)
# res = img - down_up
# show('result', res)

img = cv2.resize(img, (246, 248))
down = cv2.pyrDown(img)
print('down = ', down.shape)
down_up = cv2.pyrUp(down)
print('up&down = ', down_up.shape)
res = img - down_up
show('result', res)
```
### 轮廓检测与轮廓特征
轮廓检测
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121636889.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121636748.png)
轮廓近似
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121637622.png)
判断弧AB离直线AB最远的点C到直线AB的距离是否小于阈值T
若小于T，则直线AB可以近似弧线AB
若大于T，则不可近似，由C点分成AC，CB两端弧，重复上述操作
```python
import cv2

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ? 提取轮廓
img = cv2.imread('related_data/contours.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
show('thresh', thresh)

# todo cv2.findContours()函数返回 二值图像，轮廓信息（点），层级信息 输入 二值图，轮廓搜索模式，轮廓逼近方法
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# * 若直接在原图上画图 画画信息会保留在原图像上 不利于继续操作
# * 若用imgcopy = img，img变化，imgcopy跟着变化，因此使用copy()
imgcopy = img.copy()
# todo cv2.drawContours()输入 原图像 轮廓信息 轮廓索引（-1等于全部） 颜色模式 线条厚度
res = cv2.drawContours(imgcopy, contours, -1, (0, 0, 255), 2)
show('res', res)
# res = cv2.drawContours(img, contours, -1, (0, 0 ,255), 2)
# show('res', res)
# show('img', img)

imgcopy = img.copy()
part = cv2.drawContours(imgcopy, contours, 0, (0, 0, 255), 2)
show('part', part)

# ? 轮廓特征
cnt = contours[0]
# todo cv2.contourArea() 计算轮廓面积 必须指定具体哪一个轮廓
print(cv2.contourArea(cnt))
# todo cv2.arcLength() 计算轮廓周长 True表示闭合
print(cv2.arcLength(cnt, True))
```

### 轮廓近似
```python
import cv2

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ? 轮廓近似
img = cv2.imread('related_data/contours2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,  cv2.CHAIN_APPROX_NONE)

cnt = contours[0]
imgcopy = img.copy()
res = cv2.drawContours(imgcopy, contours, -1, (0, 0, 255), 2)
show('res', res)

# * epsilon 就是阈值T 太大会使轮廓被近似成线 太小则无法近似，只能保存原来轮廓
epsilon = 0.1 * cv2.arcLength(cnt, True)
# todo cv2.approxPolyDP() 拟合函数 输入 轮廓信息 阈值 True闭合
approx = cv2.approxPolyDP(cnt, epsilon, True)

imgcopy = img.copy()
res = cv2.drawContours(imgcopy, [approx], -1, (0, 0, 255), 2)
show('res', res)

# ? 边界矩形
img = cv2.imread('related_data/contours.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[2]

# todo cv2.boundingRect() 返回外界矩形 左上角x，y坐标 矩形宽度高度 输入具体一个轮廓
x, y, w, h = cv2.boundingRect(cnt)
imgcopy = img.copy()
# todo cv2.rectangle() 画矩形 输入 原始图像 左上角坐标 右下角坐标 绘图颜色 线条厚度
imgcopy = cv2.rectangle(imgcopy, (x, y), (x + w, y + h), (0, 255, 0), 2)
show('rect', imgcopy)

# * cv2.contourArea()返回float型
area = cv2.contourArea(cnt)
rect_area = w * h
# // print(rect_area)
# // print(float(rect_area))
# // print(area)
# // print(float(area))
extent = area / float(rect_area)
print('轮廓面积与边界矩形比值：', extent)

# ? 边界圆
# todo cv2.minEnclosingCircle()最小外接圆 返回 中心点信息 半径 
(x, y), radius = cv2.minEnclosingCircle(cnt)
# // print((x, y))
center = (int(x), int(y))
radius = int(radius)
imgcopy = cv2.circle(imgcopy, center, radius, (255, 0, 0), 2)
show('circle + rect', imgcopy)
```

### 模板匹配
模板匹配
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121638492.png)
具体公式
https://blog.csdn.net/qq_52852138/article/details/121463428?ops_request_misc=
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121638693.png)

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# todo imread参数
# cv2.IMREAD_COLOR(1)：始终将图像转换为 3 通道BGR彩色图像，默认方式
# cv2.IMREAD_GRAYSCALE(0)：始终将图像转换为单通道灰度图像
# cv2.IMREAD_UNCHANGED(-1)：按原样返回加载的图像（使用Alpha通道）
# cv2.IMREAD_ANYDEPTH(2)：在输入具有相应深度时返回16位/ 32位图像，否则将其转换为8位
# cv2.IMREAD_ANYCOLOR(4)：以任何可能的颜色格式读取图像
# ! 若要指定中文路径
# imgFile = "../images/测试图01.png"  # 带有中文的文件路径和文件名
# # imread() 不支持中文路径和文件名，读取失败，但不会报错!
# # img = cv2.imread(imgFile, flags=1)
# # 使用 imdecode 可以读取带有中文的文件路径和文件名
# img = cv2.imdecode(np.fromfile(imgFile, dtype=np.uint8), -1)

# ? 匹配单个对象
img = cv2.imread('related_data/lena.jpg', 0)
template = cv2.imread('related_data/face.jpg', 0)
# todo a3[:2,1:,:2] :2表示前两个都要，1：表示除了第1个不要，剩下的都要
h, w = template.shape[:2]
# print('hello  ', template.shape)
# print(h,'  ',w)
# show('img', img)
# # img.shape (263, 263)
# print(img.shape)
# ! template.shape (110, 85)  height, width
# print(template.shape)

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
# todo cv2.matchTemplate()模板匹配函数 原始图像A*B 模板a*b 输出矩阵(A-a+1)*(B-b+1)
res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
# (154, 179)  
# print(res.shape)
# print(res) 

# todo cv2.minMaxLoc() 搜索矩阵最小值最大值 返回值和位置（左上角的点）
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# print(min_val)
# print(min_loc)
# print(max_val)
# print(max_loc)

for i in methods:
    imgcopy = img.copy()
    # print(i)
    # todo 使用eval()函数，将字符串还原为数字类型，和int()函数的作用类似 即转化为cv类型对应的数字
    method = eval(i)
    # print(method)
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # * cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED 值越小越拟合 其他越大越拟合
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    # * 在gray图像上画图 ，在0-255之间选择颜色,选rgb类型会自动划分
    cv2.rectangle(imgcopy, top_left, bottom_right, 255, 2)
    
    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(imgcopy, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.suptitle(i)
    plt.show()

# ? 匹配多个对象
img_rgb = cv2.imread('related_data/mario.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('related_data/mario_coin.jpg', 0)
h, w=template.shape[:2]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
# * 取阈值大于0.8的坐标
threshold = 0.8
# todo numpy.where(condition, x, y) 判断condition返回True的
# condition是一个布尔数组或布尔条件表达式，用于指定需要满足的条件。
# x和y分别是满足条件和不满足条件时的替代值。它们可以是标量、向量或数组。
# 函数返回一个与condition大小相同的数组，其中满足条件的元素用x替代，不满足条件的元素用y替代
loc = np.where(res > threshold)
# ! 二维数组返回的索引
# array = np.array([[0, 1, 2, 3, 5, 4, 5], [0, 1, 2, 3, 4, 5, 5]])
#  array_ = np.where(array == np.max(array))  找等于5的点
# 值(array([0, 0, 1, 1], dtype=int64), array([4, 6, 5, 6], dtype=int64))
# * 代表5的索引为[0,4] [0,6]......
#  np.where()用法 https://blog.csdn.net/u011699626/article/details/112058004?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170489996416800185850737%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170489996416800185850737&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-112058004-null-null.142^v99^pc_search_result_base2&utm_term=np.where&spm=1018.2226.3001.4187
#  https://www.bilibili.com/video/BV1Lc411S75F/?spm_id_from=333.337.search-card.all.click&vd_source=6fd71d34326a08965fdb07842b0124a7
# print(loc)
# print('--------------------------------------------------------')
# print(res)



# ! [::-1]的作用是对列表进行翻转 且不能是[:,:,-1]
# ! 取翻转是因为cv操作一个点的位置是(y,x)或(height,width) 而np.where得到的索引是数组x，y
# 翻转 https://blog.csdn.net/qq_40714949/article/details/127037956?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170490283216800227477955%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170490283216800227477955&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-127037956-null-null.142^v99^pc_search_result_base2&utm_term=%5B%3A%20%3A%20-1%5D&spm=1018.2226.3001.4187

# todo zip( )函数 将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
# a = [1, 3, 5, 7, 9]
# b = [2, 4, 6, 8, 10]
# print(zip(a, b))
# print(list(zip(a, b)))
# # <zip object at 0x12844128>
# # [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]

# todo zip(* ) 相当于解压操作（逆向zip()）将数组释放
# a = [[1, 3], [2, 4], [3, 5], [4, 6]]
# b = [[2, 4, 6, 8, 10], [1, 3, 5, 7, 9]]
# print(list(zip(*a)))
# # [(1, 2, 3, 4), (3, 4, 5, 6)]
# print(list(zip(*b)))
# # [(2, 1), (4, 3), (6, 5), (8, 7), (10, 9)]

# ! 因此 zip(loc[1],loc[0]) 和 zip(*loc[: : -1])结果一致 
# * 第一个相当于示例1          第二个相当于示例2的b
# // print(list(zip(loc[1],loc[0])))
# // print('--------------------------------------------------------')
# // print(list(zip(*loc[: : -1])))
# // print('--------------------------------------------------------')
# // print(list(zip(loc[: : -1])))
# // print(list(zip(*loc[: : ])))
for i in zip(*loc[: : -1]):
    bottom_right = (i[0] + w, i[1] + h)
    cv2.rectangle(img_rgb, i, bottom_right, (255, 0, 0), 1)

img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.xticks([]), plt.yticks([])
plt.title('result')
plt.show()
```


## 第七章 直方图与傅里叶变换
### 直方图绘制
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121649639.png)
```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('related_data/lena.jpg', 0)
# todo cv2.calcHist()计算直方图 输入 原图像 通道 掩膜 bin区间大小 像素值范围 
# todo 这个函数的参数必须用[ ]括起来 
hist = cv2.calcHist([img], [0], None, [256], [0,256])
print(hist.shape)

# ? plt.hist() 由原图像直接输出灰度直方图
# ! 读取的图像是一个二维数组 需要降维！
# todo img.ravel()将二维数组降成一位数组
# * bin值为16 划分成16个子集
plt.hist(img.ravel(), 16)
plt.show()

# ? plt.plot() 由cv2.calcHist()得到数组 输出RGB直方图
img = cv2.imread('related_data/lena.jpg')
color = ('b', 'g', 'r')
# todo enumerate()将一个可遍历iterable数据对象(如list列表、tuple元组或str字符串)组合为一个索引序列
# todo 同时列出数据和数据下标，一般用在for循环当中 返回 索引 数值
print(list(enumerate(color)))
for i,col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0,255])
    plt.plot(histr, color = col)
    plt.xlim([0,256])
plt.show()

# ? 使用mask
# * img.shape[:2]就是图像高和宽 img.shape(263, 263, 3) [:2]取前两个
img = cv2.imread('related_data/lena.jpg', 0)
mask = np.zeros(img.shape[:2], np.uint8)
# * 先高后宽
mask[50:200, 50:200] = 255
# show('mask', mask)
# // print(mask.shape)
# // print(img.shape)
# todo 与操作 bitwise_and(src1, src2, dst=None, mask=None)
# todo src1 src2 参与运算的两个图像 dst 输出数组 mask掩膜
# openCV 位操作 https://blog.csdn.net/m0_51545690/article/details/123956698?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170515555116800227474815%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170515555116800227474815&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-123956698-null-null.142^v99^pc_search_result_base2&utm_term=cv2.bitwise_and&spm=1018.2226.3001.4187
mask_img = cv2.bitwise_and(img, img, mask=mask)

hist_full = cv2.calcHist([img], [0], None, [256], [0,255])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0,255])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(mask_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0, 255])
plt.show()
```
### 直方图均衡化
直方图均衡化
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121650056.png)
自适应直方图均衡化
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121650076.png)
```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ? 直方图均衡化
img = cv2.imread('related_data/lena.jpg', 0)
# todo cv2.equalizeHist()均衡化函数 原理看md
equ = cv2.equalizeHist(img)
plt.subplot(121),plt.hist(img.ravel(), 256),plt.title('img')
plt.subplot(122),plt.hist(equ.ravel(), 256),plt.title('equ')
plt.show()

res = np.hstack((img, equ))
show('result', res)

# 有的图像均衡化会丢失细节 最好分块均衡化 可能效果会更好 如自适应均衡化
img = cv2.imread('related_data/clahe.jpg', 0)
# todo cv2.equalizeHist()均衡化函数 原理看md
equ = cv2.equalizeHist(img)
plt.subplot(121),plt.hist(img.ravel(), 256),plt.title('img')
plt.subplot(122),plt.hist(equ.ravel(), 256),plt.title('equ')
plt.show()

res = np.hstack((img, equ))
show('result', res)

# ? 自适应均衡化
# todo cv2.createCLAHE(clipLimit,tileGridSize) 噪声影响若过大则多余重新分配 可以看md或网址
# clipLimit:限制对比度的阈值，默认为40，直方图中像素值出现次数大于该阈值，多余的次数会被重新分配
# tileGridSize:图像会被划分的size， 如tileGridSize=(8,8),默认为(8,8)
# https://blog.csdn.net/m0_45805664/article/details/107615211?ops_request_misc=&request_id=&biz_id=102&utm_term=cv2.createCLAHE&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-4-107615211.142^v99^pc_search_result_base2&spm=1018.2226.3001.4187
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# todo 使用apply()将自适应均衡化操作作用于原图
res_clahe = clahe.apply(img)
res = np.hstack((img, equ, res_clahe))
show('Adaptive equalization', res)
```

### 傅里叶变换
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121650187.png)
所有周期函数可以用一些正弦波堆叠
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121651401.png)
时域频域转换 看侧面图
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121651827.png)
转化到频域中会使图像层次分明 使图像处理方法简单高效 提取低频高频区域更简单
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('related_data/lena.jpg', 0)

# ? 傅里叶变换
# * opencv执行傅里叶必须转换为np.float32格式
img_float32 = np.float32(img)

# todo cv2.dft()执行傅里叶变换 # src：输入图像，可以为实数矩阵或者复数矩阵 flags：转换标志
# （如DFT_COMPLEX_OUTPUT，对一维或二维实数数组正变换，输出一个同样尺寸的复数矩阵）
# （DFT_REAL_OUTPUT，对一维或二维复数数组反变换，通常输出同样尺寸的复矩阵）
# 返回结果：是双通道的，第一个的结果是虚数部分，第二个通道的结果是实数部分
# https://blog.csdn.net/qq_45832961/article/details/124175063
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
# todo np.fft.fftshift() 将低频部分转换到中间位置
dft_shift = np.fft.fftshift(dft)
# todo cv2.magnitude() 将复数结果转换为幅值 res=sqrt(im^2 + re^2)转为成灰度图可以表达的形式
# * 因此结果值比较小不宜观察 通过映射公式放大细节 
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('input'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('magnitude_spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
#越靠近中心的就是低频部分 朝两边发散的是高频
```

### 低通与高通滤波
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('related_data/lena.jpg', 0)
img_float32 = np.float32(img)
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# ? 低通滤波器 
rows, cols = img.shape
# 中心位置
crow, ccol = int(rows / 2), int(cols / 2)

# * 为了划分低频和高频的位置
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow - 30 : crow + 30, ccol-30 : ccol +30] = 1

# IDFT 逆变化
# * 获取低通掩膜
fshift = dft_shift * mask
# todo np.fft.ifftshift() 把原来放在中间的低频 重新还原
f_ishift = np.fft.ifftshift(fshift)
# todo cv2.idft() 傅里叶逆变换
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('input'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('res'), plt.xticks([]), plt.yticks([])
plt.show()

# ? 高通滤波器
mask = np.ones((rows, cols, 2), np.uint8)
mask[crow - 30 : crow + 30, ccol-30 : ccol +30] = 0

# * 获取高通掩膜
fshift = dft_shift * mask
# todo np.fft.ifftshift() 把原来放在中间的低频 重新还原
f_ishift = np.fft.ifftshift(fshift)
# todo cv2.idft() 傅里叶逆变换
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('input'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('res'), plt.xticks([]), plt.yticks([])
plt.show()
```

## 项目实战 信用卡数字识别
Bank Card Recogition
cv2.THRESH|OSTU 方法找到最佳的阈值
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121725564.png)

### 信用卡数字识别
```python
import numpy as np
import argparse
import cv2
function=__import__("026_function")

# ! import不止可以导入模块 还可以导入同工程目录下同名的其他py文件 import myutils
# ! 导入模块不支持“.”点号 若中间有空格 可以用__import__
# todo from 文件名 import 类（文件名为要导入的类所存在的文件名） 在包含主程序的文件中运用from语句导入我们想要调用的类
# import 用法 https://blog.csdn.net/m0_64365419/article/details/125953971?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170541205516800188552059%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=170541205516800188552059&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-2-125953971-null-null.142^v99^pc_search_result_base2&utm_term=Python%E5%AF%BC%E5%85%A5%E6%96%87%E4%BB%B6%E5%90%8D%E4%B8%AD%E5%B8%A6%E7%A9%BA%E6%A0%BC%E7%9A%84%E6%A8%A1%E5%9D%97&spm=1018.2226.3001.4187

# ? 配置参数
# ! 配置参数在lauch.json 中加入图片相应的路径
# todo argparse.ArgumentParser() 创建一个解析对象；
# todo add_argument向该对象中添加你要关注的命令行参数和选项，每一个add_argument方法对应一个你要关注的参数或选项；
# * name or flags 一个命名或者一个选项字符串的列表，例如 foo 或 -f, --foo。
# * required 此命令行选项是否可省略 required=True 即必选
# * help 一个此选项作用的简单描述
# https://blog.csdn.net/weixin_45388452/article/details/125401449?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170532969416800182778928%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170532969416800182778928&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-125401449-null-null.142^v99^pc_search_result_base2&utm_term=add_argument&spm=1018.2226.3001.4187
# todo 最后调用parse_args()方法进行解析；解析成功之后即可使用。
# the following arguments are required
# https://blog.csdn.net/PSpiritV/article/details/122997870?ops_request_misc=&request_id=&biz_id=102&utm_term=the%20following%20arguments%20are%20re&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-122997870.142^v99^pc_search_result_base2&spm=1018.2226.3001.4187
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-t", "--template", required=True, help="path to template OCR-A image")
args = vars(ap.parse_args())

# ? 最后匹配银行卡类型
FIRST_NUMBER = {
	"3": "America Express",
	"4": "Visa",
	"5": "MasterCard",
	"6": "Discover Card"
}

def show(name,img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# todo args[]可以直接读取图片
TemplateImg = cv2.imread(args["template"])
show('TemplateImg', TemplateImg)

# * 灰度图
ref = cv2.cvtColor(TemplateImg, cv2.COLOR_BGR2GRAY)
show('gray',ref)

# * 二值图
# ! threshold返回两个值 第一个是阈值 第二个是结果图像 [1]就是返回第二个结果
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
show('threhold',ref)

# * 只找最外层轮廓
refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(TemplateImg, refCnts, -1, (0, 0, 255), 3)
show('Contours', TemplateImg)

# todo 计算轮廓个数尽量用len(refCnts) 不要用np.array(refCnts).shape
# print (len(refCnts))

# ? 使用function.py中的函数排序轮廓
refCnts ,boundingCoxes = function.sort_contours(refCnts, method='left-to-right')

# ? 给每个数字分配一个模板
# * 创建空字典
digits = {}
for (i, c) in enumerate(refCnts):
	(x, y, w, h) = cv2.boundingRect(c)
	roi = ref[y:y + h, x:x + w]
	roi = cv2.resize(roi, (57, 88))

	digits[i] = roi

# ? 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# ? 待匹配图像
TestImg = cv2.imread(args["image"])
show('original', TestImg)
# ! function.py
TestImg = function.resize(TestImg, width=300)
gray = cv2.cvtColor(TestImg, cv2.COLOR_BGR2GRAY)
show('gray', gray)

# todo 礼帽操作突出更明亮的区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
show('TopHat', tophat)

# ? 梯度计算Sobel
# * ksize=-1 即3*3卷积核
gradx = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradx = np.absolute(gradx)
show('gradx', gradx)
(minVal, maxVal) = (np.min(gradx), np.max(gradx))
# ! 归一化！
gradx = (255 * ((gradx - minVal) / (maxVal - minVal)))
# todo astype()：把dataframe中的任何列转换成其他类型
gradx = gradx.astype("uint8")
# print(np.array(gradx).shape)
show('gradx', gradx)

# ? 闭操作1 找四个一组的数字矩形框
gradx = cv2.morphologyEx(gradx, cv2.MORPH_CLOSE, rectKernel)
show('Close gradx', gradx)
# todo #THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0 示意图看md
thresh = cv2.threshold(gradx, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
show('thresh', thresh)

# ? 闭操作2 填充内部空洞
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel) #再来一个闭操作
show('Close thresh',thresh)

# ? 找轮廓 画轮廓
threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts
TestImgcopy = TestImg.copy()
cv2.drawContours(TestImgcopy, cnts, -1, (0, 255, 0), 3)
show('Contours', TestImgcopy)
locs = []

# ? 筛选轮廓
# for (i, c) in enumerate(cnts):
# 	# 计算矩形
# 	(x, y, w, h) = cv2.boundingRect(c)
for i in range(len(cnts)):
	(x, y, w, h) = cv2.boundingRect(cnts[i])
	ar = w / float(h)

	if ar > 2.5 and ar < 4.0:
		if (w > 40 and w < 55) and (h > 10 and h < 20):
			# todo append() 函数可以向列表末尾添加元素
			locs.append((x, y, w, h))

# ? 轮廓排序
loc = sorted(locs, key = lambda x:x[0], reverse=False)
output = []

# ? 遍历四个一组的数字组合
for (i, (Gx, Gy, Gw, Gh)) in enumerate(loc):
	# * 创建结果字典
	GroupOutput = []
    
	# * 给原来的的数字组合轮廓上下左右拓宽5
	group = gray[Gy - 5: Gy + Gh +5, Gx - 5: Gx + Gw + 5]
	show('group', group)

	# * OTSU 自动找阈值
	group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	# show('group thresh', group)

	digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# ! contours.sort_contours
	digitCnts = function.sort_contours(digitCnts, method="left-to-right")[0]
    
	for c in digitCnts:
		# 找到当前数值的轮廓，resize成合适的的大小
		(x, y, w, h) = cv2.boundingRect(c)
		roi = group[y:y + h, x:x + w]
		roi = cv2.resize(roi, (57, 88))
		# show('roi',roi)

		# * 匹配得分
		scores = []
		# todo items()以列表返回可遍历的(键, 值) 元组数组
		for(digit, digitROI) in digits.items():
			result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
			# too many values to unpack 错误原因：输出变量个数不一致 我.打成,了
			(_, score, _, _) = cv2.minMaxLoc(result)
			scores.append(score)
		# todo np.argmax()返回最大值
		GroupOutput.append(str(np.argmax(scores)))
		cv2.rectangle(TestImg, (Gx - 5, Gy - 5), (Gx + Gw + 5 , Gy + Gh + 5), (0, 0, 255), 1)
		# todo cv2.putText() 图片 文字 文字位置 字体的类型 大小 颜色 粗细
		cv2.putText(TestImg, "".join(GroupOutput), (Gx, Gy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    # todo extend用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
	# ! extend 与 append 区别
	# * append()用于在列表末尾添加新的对象，列表只占一个索引位，在原有列表上增加。
	# * extend()向列表尾部追加一个列表，将列表中的每个元素都追加进来，在原有列表上增加。
	# https://blog.csdn.net/wzk4869/article/details/125689590?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170559202716800192263479%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170559202716800192263479&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-125689590-null-null.142^v99^pc_search_result_base2&utm_term=python%E4%B8%ADappend%E5%92%8Cextend%E5%8C%BA%E5%88%AB&spm=1018.2226.3001.4187
	output.extend(GroupOutput)
	# print(output)

# todo format是字符串内嵌的一个方法，用于格式化字符串 以大括号{}来标明被替换的字符串
# https://blog.csdn.net/qq_42855293/article/details/118480087?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170559201016800227422648%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170559201016800227422648&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-118480087-null-null.142^v99^pc_search_result_base2&utm_term=format%E5%9C%A8python%E4%B8%AD%E7%9A%84%E7%94%A8%E6%B3%95&spm=1018.2226.3001.4187
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
# todo 'sep'.join(seq) 连接字符串数组 
# todo sep：分隔符。可以为空 seq：要连接的元素序列、字符串、元组、字典
print("Credit Card #: {}".format("".join(output)))
show('result', TestImg)
```


### function
```python
import cv2

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    
    # ? 返回每个轮廓的x y h w
    # todo for c in s: #for in 循环表示将s的每一个元素赋值给c
    # todo for in 输出的是数组的index下标，而for of 输出的是数组的每一项的值
    # https://blog.csdn.net/qq_43796489/article/details/119566594
    # todo 列表生成式 函数或x for x in range(0,101) for循环遍历出来的值，放入列表中
    # e.g. numbers =[x for x in range(0,101)]
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    # // print(cv2.boundingRect(c) for c in cnts)
    # // print(list(boundingBoxes))
    # // print(list(zip(cnts, boundingBoxes)))

    # ? 排序操作
    # todo sorted(iterable, cmp=None, key=None, reverse=False)
    # 【iterable】 可迭代对象。
    # 【cmp】 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。（一般省略）
    # 【key】主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
    # 常用的用来作为参数key的函数有 lambda函数和operator.itemgetter()
    # 尤其是列表元素为多维数据时，需要key来选取按哪一位数据来进行排序
    # 【reverse】 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
    # todo key=lambda x:len(x) 元素: 元素[字段索引]
    # 想对元素第二个字段排序，则key=lambda y: y[1]，想对元素第一个字段排序，则key=lambda y: y[0]
    # y可以是任意字母 x a b......
    # * 这里的b: b[1][i] 代表由cnts, boundingBoxes组合的zip对象 的第二个元素（最小外接矩形)的第一个元素即为x值 最小外接矩形左上角坐标
    # todo 如果zip()没有可迭代的元素，则它将返回一个空的迭代器，如果每个迭代器的元素个数不一致，则返回的列表长度与最短的一致。
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i],
                                        reverse=reverse))
    return cnts, boundingBoxes

def resize(img, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = img.shape[:2]
    if width is None and height is None:
        return img
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(img, dim, interpolation=inter)
    return resized
```

### ocr_template_match
```python
# 导入工具包
from imutils import contours
import numpy as np
import argparse
import cv2
myutils=__import__("026_function")

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-t", "--template", required=True,
	help="path to template OCR-A image")
args = vars(ap.parse_args())

# 指定信用卡类型
FIRST_NUMBER = {
	"3": "American Express",
	"4": "Visa",
	"5": "MasterCard",
	"6": "Discover Card"
}
# 绘图展示
def cv_show(name,img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
# 读取一个模板图像
img = cv2.imread(args["template"])
cv_show('img',img)
# 灰度图
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show('ref',ref)
# 二值图像
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('ref',ref)

# 计算轮廓
#cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图）,cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
#返回的list中每个元素都是图像中的一个轮廓

refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,refCnts,-1,(0,0,255),3) 
cv_show('img',img)
# print (np.array(refCnts).shape)
refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0] #排序，从左到右，从上到下
digits = {}

# 遍历每一个轮廓
for (i, c) in enumerate(refCnts):
	# 计算外接矩形并且resize成合适大小
	(x, y, w, h) = cv2.boundingRect(c)
	roi = ref[y:y + h, x:x + w]
	roi = cv2.resize(roi, (57, 88))

	# 每一个数字对应每一个模板
	digits[i] = roi

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

#读取输入图像，预处理
image = cv2.imread(args["image"])
cv_show('image',image)
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show('gray',gray)

#礼帽操作，突出更明亮的区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel) 
cv_show('tophat',tophat) 
# 
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, #ksize=-1相当于用3*3的
	ksize=-1)


gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

print (np.array(gradX).shape)
cv_show('gradX',gradX)

#通过闭操作（先膨胀，再腐蚀）将数字连在一起
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel) 
cv_show('gradX',gradX)
#THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
thresh = cv2.threshold(gradX, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] 
cv_show('thresh',thresh)

#再来一个闭操作

thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel) #再来一个闭操作
cv_show('thresh',thresh)

# 计算轮廓

threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3) 
cv_show('img',cur_img)
locs = []

# 遍历轮廓
for (i, c) in enumerate(cnts):
	# 计算矩形
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)

	# 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
	if ar > 2.5 and ar < 4.0:

		if (w > 40 and w < 55) and (h > 10 and h < 20):
			#符合的留下来
			locs.append((x, y, w, h))

# 将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda x:x[0])
output = []

# 遍历每一个轮廓中的数字
for (i, (gX, gY, gW, gH)) in enumerate(locs):
	# initialize the list of group digits
	groupOutput = []

	# 根据坐标提取每一个组
	group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
	cv_show('group',group)
	# 预处理
	group = cv2.threshold(group, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	cv_show('group',group)
	# 计算每一组的轮廓
	digitCnts,hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	digitCnts = contours.sort_contours(digitCnts,
		method="left-to-right")[0]

	# 计算每一组中的每一个数值
	for c in digitCnts:
		# 找到当前数值的轮廓，resize成合适的的大小
		(x, y, w, h) = cv2.boundingRect(c)
		roi = group[y:y + h, x:x + w]
		roi = cv2.resize(roi, (57, 88))
		cv_show('roi',roi)

		# 计算匹配得分
		scores = []

		# 在模板中计算每一个得分
		for (digit, digitROI) in digits.items():
			# 模板匹配
			result = cv2.matchTemplate(roi, digitROI,
				cv2.TM_CCOEFF)
			(_, score, _, _) = cv2.minMaxLoc(result)
			scores.append(score)

		# 得到最合适的数字
		groupOutput.append(str(np.argmax(scores)))

	# 画出来
	cv2.rectangle(image, (gX - 5, gY - 5),
		(gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
	cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
		cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

	# 得到结果
	output.extend(groupOutput)

# 打印结果
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.waitKey(0)
```

### myutils
```python
import cv2

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts] #用一个最小的矩形，把找到的形状包起来x,y,h,w
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
```

### 最终识别结果
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407121726915.png)