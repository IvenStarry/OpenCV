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