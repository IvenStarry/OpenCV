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
