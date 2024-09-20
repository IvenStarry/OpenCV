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