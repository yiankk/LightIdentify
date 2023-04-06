import cv2
import numpy as np

"""
功能：读取一张图片，显示出来，转化为HSV色彩空间
     并通过滑块调节HSV阈值，实时显示
"""

image = cv2.imread('E:\\pythonProject\\CV_EX\\Light_Identify\\light\\12.jpg') # 根据路径读取一张图片
image = cv2.GaussianBlur(image, (5, 5), 1)
cv2.imshow("BGR", image) # 显示图片
size = image.shape
width = size[1]
height = size[0]
hsv_low = np.array([0, 0, 0])
hsv_high = np.array([0, 0, 0])

# 下面几个函数，写得有点冗余


def h_low(value):
    hsv_low[0] = value


def h_high(value):
    hsv_high[0] = value


def s_low(value):
    hsv_low[1] = value


def s_high(value):
    hsv_high[1] = value


def v_low(value):
    hsv_low[2] = value


def v_high(value):
    hsv_high[2] = value


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# 可以自己设定初始值，最大值255不需要调节
cv2.createTrackbar('H low', 'image', 0, 255, h_low)
cv2.createTrackbar('H high', 'image', 10, 255, h_high)
cv2.createTrackbar('S low', 'image', 60, 255, s_low)
cv2.createTrackbar('S high', 'image', 255, 255, s_high)
cv2.createTrackbar('V low', 'image', 60, 255, v_low)
cv2.createTrackbar('V high', 'image', 255, 255, v_high)


dst = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # BGR转HSV
dst = cv2.inRange(dst, hsv_low, hsv_high) # 通过HSV的高低阈值，提取图像部分区域
cv2.imshow('dst', dst)
kernel = np.ones((3,3), np.uint8)
open = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)  # 开运算
contours, hierarchy = cv2.findContours(open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i, contour in enumerate(contours):  # 从轮廓列表中选出设计矩形 (需改进)
    x, y, w, h = cv2.boundingRect(contour)

    img = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('img', img)
cv2.waitKey(0)

while True:
    dst = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # BGR转HSV
    dst = cv2.inRange(dst, hsv_low, hsv_high)  # 通过HSV的高低阈值，提取图像部分区域
    cv2.imshow('dst', dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()