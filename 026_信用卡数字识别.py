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