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