import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

image = cv.imread('E:\\pythonProject\\CV_EX\\Light_Identify\\light\\26.jpg') # 根据路径读取一张图片


def hsv_cope(image):
    color_dist = {
        'green': {'Lower': np.array([45, 145, 145]), 'Upper': np.array([90, 255, 255])},
        'red': {'Lower': np.array([0, 145, 146]), 'Upper': np.array([8, 255, 255])},
        'deep_red':  {'Lower': np.array([160, 143, 146]), 'Upper': np.array([180, 255, 255])},
    }
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    inRange_hsv = cv.inRange(hsv, color_dist['green']['Lower'], color_dist['green']['Upper'])
    red_0 = cv.inRange(hsv, color_dist['red']['Lower'], color_dist['red']['Upper'])
    red_1 = cv.inRange(hsv, color_dist['deep_red']['Lower'], color_dist['deep_red']['Upper'])
    # 拼接两个区间
    red_goal = red_0 + red_1

    kernel = np.ones((3, 3), np.uint8)
    open_green = cv.morphologyEx(inRange_hsv, cv.MORPH_OPEN, kernel)  # 开运算
    contours_green, hierarchy_green = cv.findContours(open_green, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    open_red = cv.morphologyEx(red_goal, cv.MORPH_OPEN, kernel)  # 开运算
    contours_red, hierarchy_red = cv.findContours(open_red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    hsv_list_green = []
    hsv_record_green = 0
    for i, contour in enumerate(contours_green):  # 从轮廓列表中选出设计矩形 (需改进)
        x, y, w, h = cv.boundingRect(contour)
        hsv_list1 = [x, y, w, h]
        hsv_list_green.append(hsv_list1)
        hsv_record_green += 1
        cv.rectangle(image, (hsv_list_green[i][0], hsv_list_green[i][1]),
                     (hsv_list_green[i][0] + hsv_list_green[i][2], hsv_list_green[i][1] + hsv_list_green[i][3]),
                     (0, 255, 0), 2)

    hsv_list_red = []
    hsv_record_red = 0
    for j, contour in enumerate(contours_red):
        x, y, w, h = cv.boundingRect(contour)
        hsv_list2 = [x, y, w, h]
        hsv_list_red.append(hsv_list2)
        hsv_record_red += 1
        cv.rectangle(image, (hsv_list_red[j][0], hsv_list_red[j][1]),
                     (hsv_list_red[j][0] + hsv_list_red[j][2], hsv_list_red[j][1] + hsv_list_red[j][3]),
                     (0, 255, 0), 2)


    cv.imshow('1', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # return hsv_list_green, hsv_record_green, hsv_list_red, hsv_record_red
hsv_cope(image)


