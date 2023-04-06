import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
import math

# 原始图片路径
original_path = 'E:\\pythonProject\\CV_EX\\Light_Identify\\light\\'

# 批处理保存路径
# canny框选路径
pro1_path = 'E:\\pythonProject\\CV_EX\\Light_Identify\\canny\\'
# 黑色筛选框选路径
pro2_path = 'E:\\pythonProject\\CV_EX\\Light_Identify\\black\\'
# canny和黑色框合成路径
pro3_path = 'E:\\pythonProject\\CV_EX\\Light_Identify\\canny_black\\'
# 颜色框选路径
pro4_path = 'E:\\pythonProject\\CV_EX\\Light_Identify\\hsv\\'
# 最终成品路径
pro5_path = 'E:\\pythonProject\\CV_EX\\Light_Identify\\final\\'

original_list = os.listdir(original_path)  # 存储着每张图片的名字
original_list.sort(key=lambda x: int(x.split('.')[0]))
pro1_list = os.listdir(pro1_path)
pro1_list.sort(key=lambda x: int(x.split('.')[0]))
pro2_list = os.listdir(pro2_path)
pro2_list.sort(key=lambda x: int(x.split('.')[0]))


# 二值化,canny等相关运算对原图像处理获取一次物体框轮廓
def img_processing1(img, width, high):   # canny相关处理
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    aussian = cv.GaussianBlur(gray, (5, 5), 1)  # 高斯滤波
    ret1, th1 = cv.threshold(aussian, 100, 255, cv.THRESH_BINARY_INV)
    # 开运算
    kernel = np.ones((6, 3), np.uint8)
    open = cv.morphologyEx(th1, cv.MORPH_OPEN, kernel)
    # Sobel算子
    xgrad = cv.Sobel(open, cv.CV_16SC1, 1, 0)
    ygrad = cv.Sobel(open, cv.CV_16SC1, 0, 1)
    # canny边缘检测
    canny = cv.Canny(xgrad, ygrad, 80, 150)
    contours, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    list_1 = []
    record_1 = 0
    for i, contour in enumerate(contours):  # 从轮廓列表中选出设计矩形
        x, y, w, h = cv.boundingRect(contour)
        if h > 0.05*high or w > 0.05*width:
            list_11 = [x, y, w, h]
            list_1.append(list_11)
            record_1 += 1
    return list_1, record_1


# 通过黑色筛选,矩形判断对原图像处理获取第二次物体框轮廓
def img_processing2(img, width, high):
    image = cv.GaussianBlur(img, (5, 5), 1)  # 高斯滤波
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    inRange_hsv = cv.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 100]))
    kernel = np.ones((2, 1), np.uint8)
    open = cv.morphologyEx(inRange_hsv, cv.MORPH_OPEN, kernel)
    contours, hierarchy = cv.findContours(open, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)  # 寻找轮廓点
    list_2 = []
    record_2 = 0
    for obj in contours:
        area = cv.contourArea(obj)  # 计算轮廓内区域的面积
        if area > 0.0001 * width * high:
            # cv.drawContours(dst1, obj, -1, (255, 0, 0), 4)  # 绘制轮廓线
            perimeter = cv.arcLength(obj, True)  # 计算轮廓周长
            approx = cv.approxPolyDP(obj, 0.02 * perimeter, True)  # 获取轮廓角点坐标
            CornerNum = len(approx)  # 轮廓角点的数量
            x, y, w, h = cv.boundingRect(approx)  # 获取坐标值和宽度、高度
            # 轮廓对象分类
            if CornerNum == 4:
                if w != h:

                    # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绘制边界框
                    list2 = [x, y, w, h]
                    list_2.append(list2)
                    record_2 += 1
    return list_2, record_2


# 通过hsv针对红色和绿色来选出轮廓用于和之前获取的框操作进行筛选判断
def hsv_cope(image, width, high):
    color_dist = {
        'green': {'Lower': np.array([45, 145, 145]), 'Upper': np.array([90, 255, 255])},
        'red': {'Lower': np.array([0, 145, 146]), 'Upper': np.array([8, 255, 255])},
        'deep_red': {'Lower': np.array([160, 143, 146]), 'Upper': np.array([180, 255, 255])}
    }
    image = cv.GaussianBlur(image, (5, 5), 1)  # 高斯滤波
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
        if w > 0.025 * width or h > high * 0.025:
            hsv_list1 = [x, y, w, h]
            hsv_list_green.append(hsv_list1)
            hsv_record_green += 1

    hsv_list_red = []
    hsv_record_red = 0
    for j, contour in enumerate(contours_red):
        x, y, w, h = cv.boundingRect(contour)
        if w > 0.025 * width or h > high * 0.025:
            hsv_list2 = [x, y, w, h]
            hsv_list_red.append(hsv_list2)
            hsv_record_red += 1

    return hsv_list_green, hsv_record_green, hsv_list_red, hsv_record_red


# 将canny获取的物体框与黑色,矩形筛选获取的物体框进行合成让物体框更加全面
def compare(list_1,record_1, list_2, record_2):
    count = 0
    if record_2 > 0:

        for i in range(record_1):
            for j in range(record_2):
                if (abs(2*list_1[i][0]+list_1[i][2]-2*list_2[j][0]-list_2[j][2]) < list_1[i][2]+list_2[j][2]
                and abs(2*list_1[i][1]+list_1[i][3]-2*list_2[j][1]-list_2[j][3]) < list_1[i][3]+list_2[j][3]):
                    list_1[i] = 0
                    count += 1
                    break
    list_new = list_1 + list_2
    for i in range(count):
        list_new.remove(0)

    return list_new, len(list_new)


# 将物体框与hsv操作获取的轮廓进行包含判断筛选出红绿灯画出框
def rect(img, record, list1, hsv_record_green, hsv_list_green, hsv_record_red, hsv_list_red):
    for i in range(record):
        for j in range(hsv_record_green):
            if (list1[i][0] < hsv_list_green[j][0] and list1[i][1] < hsv_list_green[j][1]
                    and list1[i][0] + list1[i][2] > hsv_list_green[j][0] + hsv_list_green[j][2]
                    and list1[i][1] + list1[i][3] > hsv_list_green[j][1] + hsv_list_green[j][3]):
                cv.rectangle(img, (list1[i][0], list1[i][1]),
                             (list1[i][0] + list1[i][2], list1[i][1] + list1[i][3]), (0, 255, 0), 2)
                cv.putText(img, "Greenlight", (list1[i][0], list1[i][1] - 20), 0, 0.65, (0, 255, 53), 2)
                break

    for i in range(record):
        for j in range(hsv_record_red):
            if (list1[i][0] < hsv_list_red[j][0] and list1[i][1] < hsv_list_red[j][1]
                    and list1[i][0] + list1[i][2] > hsv_list_red[j][0] + hsv_list_red[j][2]
                    and list1[i][1] + list1[i][3] > hsv_list_red[j][1] + hsv_list_red[j][3]):
                cv.rectangle(img, (list1[i][0], list1[i][1]),
                             (list1[i][0] + list1[i][2], list1[i][1] + list1[i][3]), (66, 43, 255), 2)
                cv.putText(img, "Redlight", (list1[i][0], list1[i][1] - 20), 0, 0.65, (66, 43, 255), 2)
                break


# 菜单
def menu():
    print("-------测试环节--------")
    print("---1.图片批量处理测试---")
    print("---2.图片单张处理测试---")
    print("---3.视频处理---------")
    print("---4.待定-------------")


if __name__ == '__main__':
    menu()
    number = input("请输入数字选项: ")
    if number == '1':
        for k in range(len(original_list)):
            img = cv.imread(original_path + original_list[k])
            size = img.shape
            width = size[1]
            high = size[0]
            list_1, record_1 = img_processing1(img, width, high)
            for i in range(record_1):
                cv.rectangle(img, (list_1[i][0], list_1[i][1]),
                             (list_1[i][0] + list_1[i][2], list_1[i][1] + list_1[i][3]), (0, 255, 0), 2)
            save_path = pro1_path + str(k + 1) + '.jpg'
            cv.imwrite(save_path, img)

            img1 = cv.imread(original_path + original_list[k])
            hsv_list_green, hsv_record_green, hsv_list_red, hsv_record_red = hsv_cope(img1, width, high)
            for i in range(hsv_record_red):
                cv.rectangle(img1, (hsv_list_red[i][0], hsv_list_red[i][1]),
                             (hsv_list_red[i][0] + hsv_list_red[i][2], hsv_list_red[i][1] + hsv_list_red[i][3]),
                             (0, 255, 0), 2)
            for i in range(hsv_record_green):
                cv.rectangle(img1, (hsv_list_green[i][0], hsv_list_green[i][1]),
                             (hsv_list_green[i][0] + hsv_list_green[i][2], hsv_list_green[i][1] + hsv_list_green[i][3]),
                             (0, 255, 0), 2)
            save_path = pro4_path + str(k + 1) + '.jpg'
            cv.imwrite(save_path, img1)

            img2 = cv.imread(original_path + original_list[k])
            list_2, record_2 = img_processing2(img2, width, high)
            for i in range(record_2):
                cv.rectangle(img2, (list_2[i][0], list_2[i][1]),
                             (list_2[i][0] + list_2[i][2], list_2[i][1] + list_2[i][3]), (0, 255, 0), 2)
            save_path = pro2_path + str(k + 1) + '.jpg'
            cv.imwrite(save_path, img2)

            img3 = cv.imread(original_path + original_list[k])
            list_new, record_new = compare(list_1, record_1, list_2, record_2)
            for i in range(record_new):
                cv.rectangle(img3, (list_new[i][0], list_new[i][1]),
                             (list_new[i][0] + list_new[i][2], list_new[i][1] + list_new[i][3]), (0, 255, 0), 2)
            save_path = pro3_path + str(k + 1) + '.jpg'
            cv.imwrite(save_path, img3)

            img4 = cv.imread(original_path + original_list[k])
            rect(img4, record_new, list_new, hsv_record_green, hsv_list_green, hsv_record_red, hsv_list_red)
            save_path = pro5_path + str(k + 1) + '.jpg'
            cv.imwrite(save_path, img4)

    if number == '2':
        number1 = input("请输入修改图片名称数字: ")
        img = cv.imread(original_path + f'{number1}.jpg')
        # img = cv.resize(img, (1024, 760))
        img1 = cv.imread(original_path + f'{number1}.jpg')
        # img1 = cv.resize(img1, (1024, 760))
        size = img.shape
        width = size[1]
        high = size[0]
        list_1, record_1 = img_processing1(img, width, high)
        hsv_list_green, hsv_record_green, hsv_list_red, hsv_record_red = hsv_cope(img, width, high)
        # dilate = cv.dilate(canny, np.ones(shape=[5, 5], dtype=np.uint8), iterations=1)
        # close = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel) # 闭运算
        # open = cv.morphologyEx(dilate, cv.MORPH_OPEN, kernel)  # 开运算
        # cv.drawContours(img, contours, -1, (0, 0, 255), 2)
        list_2, record_2 = img_processing2(img, width, high)
        list_new, record_new = compare(list_1, record_1, list_2, record_2)
        rect(img, record_new, list_new, hsv_record_green, hsv_list_green, hsv_record_red, hsv_list_red)

        cv.imshow('rectangle', img)
        # cv.imshow('threshold', img1)
        # cv.imshow('open', open)
        # cv.imshow('canny', canny)

        cv.waitKey(0)
        cv.destroyAllWindows()
    if number == '3':
        cap = cv.VideoCapture("E:\\pythonProject\\CV_EX\\Light_Identify\\vedio\\light2.mp4")
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

            list_1, record_1 = img_processing1(img, cap.get(3), cap.get(4))
            hsv_list_green, hsv_record_green, hsv_list_red, hsv_record_red = hsv_cope(img, cap.get(3), cap.get(4))
            # dilate = cv.dilate(canny, np.ones(shape=[5, 5], dtype=np.uint8), iterations=1)
            # close = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel) # 闭运算
            # open = cv.morphologyEx(dilate, cv.MORPH_OPEN, kernel)  # 开运算
            list_2, record_2 = img_processing2(img, cap.get(3), cap.get(4))
            list_new, record_new = compare(list_1, record_1, list_2, record_2)
            img = cv.resize(frame, size)
            rect(img, record_new, list_new, hsv_record_green, hsv_list_green, hsv_record_red, hsv_list_red)

            # 将处理后的视频逐帧地显示
            cv.imshow('frame_window', img)
            # 将处理后的画面逐帧地保存到output文件中
            out.write(img)
            # 获取按键动作，如果按下q，则退出循环
            if cv.waitKey(25) == ord('q'):
                break

        # cap.release()
        out.release()  # 可以实现预览
        cv.destroyAllWindows()
    else:
        print("待定")