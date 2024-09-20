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