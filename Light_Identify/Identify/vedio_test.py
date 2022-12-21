import numpy as np
import cv2 as cv

# 捕获本地视频,请自行修改自己存放视频的路径
cap = cv.VideoCapture("light.mp4")
fps = 20
size = (1280, 720)

# 指定VideoWrite 的fourCC视频编码
fourcc = cv.VideoWriter_fourcc(*'DIVX')

# 指定输出文件，fourCC视频编码，FPS帧率，画面大小
out = cv.VideoWriter('cope.avi', fourcc, fps, size)

# 检查是否导入视频成功
if not cap.isOpened():
    print("视频无法打开")
    exit()

# 获取视频的宽，高信息,    cap.get()，传入的参数可以是0-18的整数
print('WIDTH', cap.get(3))
print('HEIGHT', cap.get(4))

while True:
    # 捕获视频帧，返回ret，frame
    # ret的true与false反应是否捕获成功，frame是画面
    ret, frame = cap.read()
    if ret:
        img = cv.resize(frame, size)
        # videoWriter.write(img)
    else:
        print("视频播放完毕")
        break

    # 处理帧， 将画面转化为灰度图
    # gray1 = cv.cvtColor(frame, cv.COLOR_BGRA2GRAY)

    # 对画面帧进进行处理，这里对画面进行翻转
    # gray2 = cv.flip(gray1, 0)

    # 将处理后的视频逐帧地显示
    cv.imshow('frame_window', img)

    # 将处理后的画面逐帧地保存到output文件中
    out.write(img)

    # 获取按键动作，如果按下q，则退出循环
    # 25毫秒是恰好的，如果太小，播放速度会很快，如果太小，播放速度会很慢
    if cv.waitKey(25) == ord('q'):
        break

# cap.release()
out.release()  # 可以实现预览
cv.destroyAllWindows()

