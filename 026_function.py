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