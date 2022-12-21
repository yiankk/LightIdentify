import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

# 原始图片路径
original_path = 'E:\\pythonProject\\CV_EX\\Light_Identify\\light\\'

# 批处理保存路径
# 灰度,降噪图路径
pro1_path = 'E:\\pythonProject\\CV_EX\\Light_Identify\\light_cope1\\'
# 轮廓绘制图路径
pro2_path = 'E:\\pythonProject\\CV_EX\\Light_Identify\\light_cope2\\'
# 矩形绘制图路径
pro3_path = 'E:\\pythonProject\\CV_EX\\Light_Identify\\light_cope3\\'


original_list = os.listdir(original_path)  # 存储着每张图片的名字
original_list.sort(key=lambda x: int(x.split('.')[0]))
pro1_list = os.listdir(pro1_path)
pro1_list.sort(key=lambda x: int(x.split('.')[0]))
pro2_list = os.listdir(pro2_path)
pro2_list.sort(key=lambda x: int(x.split('.')[0]))


def hsv_cope(image, weight, height):
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

    kernel = np.ones((6, 3), np.uint8)
    open_green = cv.morphologyEx(inRange_hsv, cv.MORPH_OPEN, kernel)  # 开运算
    contours_green, hierarchy_green = cv.findContours(open_green, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    open_red = cv.morphologyEx(red_goal, cv.MORPH_OPEN, kernel)  # 开运算
    contours_red, hierarchy_red = cv.findContours(open_red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    hsv_list_green = []
    hsv_record_green = 0
    for i, contour in enumerate(contours_green):  # 从轮廓列表中选出设计矩形 (需改进)
        x, y, w, h = cv.boundingRect(contour)
        # if w > 0.025 * weight or h > height * 0.025:
        hsv_list1 = [x, y, w, h]
        hsv_list_green.append(hsv_list1)
        hsv_record_green += 1

    hsv_list_red = []
    hsv_record_red = 0
    for j, contour in enumerate(contours_red):
        x, y, w, h = cv.boundingRect(contour)
        # if w > 0.025 * weight or h > height * 0.025:
        hsv_list2 = [x, y, w, h]
        hsv_list_red.append(hsv_list2)
        hsv_record_red += 1

    return hsv_list_green, hsv_record_green, hsv_list_red, hsv_record_red


# 对图片集进行初步的转灰,降噪
def img_processing1(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    aussian = cv.GaussianBlur(gray, (5, 5), 1)  # 高斯滤波
    return aussian

    # resize = cv.resize(aussian, (500, 500))  # 整体裁剪


# 对图片进行处理绘制出轮廓
def img_processing2(img, aussian):
    # img_original = cv.imread(original_path + original_list[i])
    # img = cv.imread(pro1_path + pro1_list[i])
    # 二值化阈值处理
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
    # img_contours = cv.drawContours(img, contours, -1, (0, 0, 255), 2)
    return contours, hierarchy, th1, open, canny
    # cv.drawContours(img, contours, -1, (0, 0, 255), 2)


def img_processing3(contours, weight, height):
    list1 = []
    record = 0
    for i, contour in enumerate(contours):  # 从轮廓列表中选出设计矩形
        x, y, w, h = cv.boundingRect(contour)
        if h > 0.05*height or w > 0.05*weight:
            list2 = [x, y, w, h]
            list1.append(list2)
            record += 1
    return list1, record


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
            img = cv.resize(img, (1024, 760))
            size = img.shape
            weight = size[1]
            height = size[0]
            aussian = img_processing1(img)
            hsv_list_green, hsv_record_green, hsv_list_red, hsv_record_red = hsv_cope(img, weight, height)
            contours, hierarchy, th1, open, canny = img_processing2(img, aussian)

            list1, record = img_processing3(contours, weight, height)
            img = cv.imread(original_path + original_list[k])
            img = cv.resize(img, (1024, 760))
            rect(img, record, list1, hsv_record_green, hsv_list_green, hsv_record_red, hsv_list_red)

            save_path = pro1_path + str(k + 1) + '.jpg'
            cv.imwrite(save_path, aussian)
            # save_path = pro2_path + str(k + 1) + '.jpg'
            # cv.imwrite(save_path, img_contours)
            save_path = pro3_path + str(k + 1) + '.jpg'
            cv.imwrite(save_path, img)

    if number == '2':
        number1 = input("请输入修改图片名称数字: ")
        img = cv.imread(original_path + f'{number1}.jpg')
        img = cv.resize(img, (1024, 760))
        size = img.shape
        weight = size[1]
        height = size[0]
        hsv_list_green, hsv_record_green, hsv_list_red, hsv_record_red = hsv_cope(img, weight, height)
        aussian = img_processing1(img)
        # dilate = cv.dilate(canny, np.ones(shape=[5, 5], dtype=np.uint8), iterations=1)
        # close = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel) # 闭运算
        # open = cv.morphologyEx(dilate, cv.MORPH_OPEN, kernel)  # 开运算
        contours, hierarchy,th1, open, canny = img_processing2(img, aussian)
        # cv.drawContours(img, contours, -1, (0, 0, 255), 2)
        img = cv.imread(original_path + f'{number1}.jpg')
        img = cv.resize(img, (1024, 760))
        list1, record = img_processing3(contours, weight, height)
        rect(img, record, list1, hsv_record_green, hsv_list_green, hsv_record_red, hsv_list_red)

        cv.imshow('rectangle', img)
        # cv.imshow('threshold', th1)
        # cv.imshow('open', open)
        # cv.imshow('canny', canny)

        cv.waitKey(0)
        cv.destroyAllWindows()
    if number == '3':
        cap = cv.VideoCapture("light2.mp4")
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

            hsv_list_green, hsv_record_green, hsv_list_red, hsv_record_red = hsv_cope(img, cap.get(3), cap.get(4))
            aussian = img_processing1(img)
            # dilate = cv.dilate(canny, np.ones(shape=[5, 5], dtype=np.uint8), iterations=1)
            # close = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel) # 闭运算
            # open = cv.morphologyEx(dilate, cv.MORPH_OPEN, kernel)  # 开运算
            contours, hierarchy, th1, open, canny = img_processing2(img, aussian)
            list1, record = img_processing3(contours, cap.get(3), cap.get(4))
            rect(img, record, list1, hsv_record_green, hsv_list_green, hsv_record_red, hsv_list_red)

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


    






























