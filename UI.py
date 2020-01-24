#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 前端库
from tkinter import *
from tkinter import ttk
import cv2 as cv
from PIL import Image, ImageTk
import os
from tkinter.simpledialog import askstring
# 后端库
import cv2
import numpy as np
from utils import file_processing, image_processing
import face_recognition
import winsound
from playsound import playsound
import copy
import time
from tkinter.messagebox import askquestion, showinfo
import threading

# 后端申明
resize_width = 160
resize_height = 160
# 存放facenet预训练模型的路径
model_path = 'models/20180408-102900'
# 存放人脸特征数据库的路径
npy_dataset_path = 'dataset/emb/faceEmbedding.npy'
filename = 'dataset/emb/name.txt'
# 加载facenet
face_net = face_recognition.facenetEmbedding(model_path)
face_detect = face_recognition.Facedetection()

# 数据库声明
model_path = 'models/20180408-102900'  # 存放facenet预训练模型的路径
dataset_path = 'dataset/images/photo_database'  # 用于建立特征库的照片
out_emb_path = 'dataset/emb/faceEmbedding.npy'  # 特征库输出地址
out_filename = 'dataset/emb/name.txt'  # 标签保存地址

# 前端声明
camera_switch = False
count = 1

# 临时变量
tempimagepath = r"dataset\images\photo_database\\"

# 摄像机设置
# 0是代表摄像头编号，只有一个的话默认为0
capture = cv.VideoCapture(0)


def getframe():
    ref, frame = capture.read()
    cv.imwrite(tempimagepath, frame)


def closecamera():
    capture.release()


# 界面相关
window_width = 1024
window_height = 578
image_width = int(window_width * 0.4)  # 画布的大小
image_height = int(window_height * 0.4)
imagepos_x = int(window_width * 0.01)  # 画布的坐标
imagepos_y = int(window_height * 0.1)
but1pos_x = 50  # 拍照按钮坐标
but1pos_y = 350
top = Tk()
top.wm_title("多媒体信息识别技术大作业")
top.geometry(str(window_width) + 'x' + str(window_height))
top.resizable(0, 0)


# 线程
def thread_it(func, *args):
    # 创建线程
    t = threading.Thread(target=func, args=args)
    # 守护
    t.setDaemon(True)
    # 启动
    t.start()


"""
对摄像头发回的照片进行处理并返回，同时将原生照片返回
"""


def tkImage(isFrame):
    ref, frame = capture.read()
    frame = cv.flip(frame, 1)  # 翻转 0:上下颠倒 大于0水平颠倒
    if isFrame is True:
        return frame
    else:
        cvimage = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
        pilImage1 = Image.fromarray(cvimage)
        # 第一个参数为指定宽和高的元组，第二个参数指定过滤器类型NEAREST、BILINER、BICUBIC、ANTALIAS，其中ANTALIAS为最高品质。
        pilImage2 = pilImage1.resize((image_width, image_height), Image.ANTIALIAS)
        tkImage = ImageTk.PhotoImage(image=pilImage2)
        return tkImage


# 开启和关闭摄像头
def button1():
    global camera_switch
    global count
    camera_switch = bool(1 - camera_switch)  # 摄像头开关
    while True:
        if camera_switch is False:
            closecamera()
            bg_image1 = Image.open(r"UI\img\SSPU_2.png")
            bg_image2 = bg_image1.resize((image_width, image_height), Image.ANTIALIAS)
            bg_im = ImageTk.PhotoImage(bg_image2)
            canvas.create_image(0, 0, anchor='nw', image=bg_im)  # 显示背景
        else:
            if capture.isOpened() is False:  # 判断摄像头是否处于开启状态
                capture.open(0)
            picture = tkImage(isFrame=False)
            canvas.create_image(0, 0, anchor='nw', image=picture)
        top.update()  # 显示图片后面加上这两句
        top.after(100)


# 拍照
def button2():
    ref, frame = capture.read()
    name = askstring(title="输入姓名", prompt="请输入姓名")
    path = tempimagepath + name + ".jpg"
    print(cv.imencode('.jpg', frame)[1].tofile(path))


# 签到
def button3():
    while True:
        face_recognition_image()


# 建立人脸数据库
def button4():
    success = create_face_embedding(model_path, dataset_path, out_emb_path, out_filename)
    if len(success) >= 0:
        showinfo("提示", "特征库已经制作完毕")


# 清空所有条目
def button5(tree):
    # 误删除
    a1 = askquestion(title="删除所有！", message="是否删除所有？")
    print(a1)
    if a1 == "yes":
        x = tree.get_children()
        for item in x:
            tree.delete(item)
    else:
        pass


# 删除单条
def delitrem(event):
    item = treeview.selection()
    a1 = askquestion(title="删除！", message="确定删除？")
    if a1 == "yes":
        print(treeview.item(item, "value"))
        print(treeview.get_children())
        treeview.delete(item)
    else:
        pass


# 控件定义
canvas = Canvas(top, bg='white', width=image_width, height=image_height)  # 绘制画布
bg_image1 = Image.open(r"UI\img\SSPU_2.png")
bg_image2 = bg_image1.resize((image_width, image_height), Image.ANTIALIAS)
bg_im = ImageTk.PhotoImage(bg_image2)
canvas.create_image(0, 0, anchor='nw', image=bg_im)  # 显示背景
b1 = Button(top, text='打开/关闭摄像头', width=15, height=2, command=button1)
b2 = Button(top, text="拍照", width=15, height=2, command=button2)
b3 = Button(top, text="人脸识别", width=15, height=2, command=lambda: thread_it(button3))
b4 = Button(top, text="建立人脸特征库", width=15, height=2, command=lambda: thread_it(button4))
b5 = Button(top, text="删除所有", width=10, height=1, command=lambda: button5(treeview))

# 控件位置设置
canvas.place(x=imagepos_x, y=imagepos_y)
b2.place(x=but1pos_x + 110, y=but1pos_y)
b1.place(x=but1pos_x - 50, y=but1pos_y)
b3.place(x=but1pos_x + 260, y=but1pos_y)
b4.place(x=but1pos_x - 50, y=but1pos_y + 70)
b5.place(x=but1pos_x + 877, y=but1pos_y + 190)

# ######################签到信息显示######################
columns = ("序号", "姓名", "签到时间")
treeview = ttk.Treeview(top, height=18, show="headings", columns=columns)  # 表格
# y滚动条
vsb = ttk.Scrollbar(top, orient="vertical", command=treeview.yview)
vsb.pack(side='right', fill='y', pady=20)
treeview.configure(yscrollcommand=vsb.set)
treeview.pack(side="right", fill="y")
treeview.place(x=500, y=20, height=510)

treeview.column("序号", width=100, anchor='center')  # 表示列,不显示
treeview.column("姓名", width=200, anchor='center')
treeview.column("签到时间", width=200, anchor='center')

treeview.heading("序号", text="序号")  # 显示表头
treeview.heading("姓名", text="姓名")
treeview.heading("签到时间", text="签到时间")
# 双击左键删除某一条记录
treeview.bind("<Double-1>", delitrem)


# ########################################后端#########################################################

def face_recognition_image():
    """
    Args:
        model_path (): facenet预训练模型的路径
        dataset_path (): 存放人脸特征数据库的路径
        filename (): 存放每张图片信息的txt文档路径
    """
    global model_path, dataset_path, filename
    # 加载人脸特征数据库的数据
    dataset_emb, names_list = load_dataset(dataset_path, filename)
    bboxes, landmarks, image = photo_calculation_and_processing()
    if bboxes == [] or landmarks == []:  # bboxes保存人脸框信息，左上角横纵坐标、右下角横纵坐标
        print("-----no face")
        exit(0)
    print("-----image have {} faces".format(len(bboxes)))
    face_images = image_processing.get_bboxes_image(image, bboxes, resize_height,
                                                    resize_width)  # 将照片调整为160*160，输入到facenet中
    face_images = image_processing.get_prewhiten_images(face_images)
    pred_emb = face_net.get_embedding(face_images)  # 生成face_images的人脸特征向量
    pred_name, pred_score = compare_embadding(pred_emb, dataset_emb, names_list)


def load_dataset(dataset_path, filename):
    '''
    加载人脸数据库
    :param dataset_path: embedding.npy文件（faceEmbedding.npy）
    :param filename: labels文件路径路径（name.txt）
    :return:
    '''
    embeddings = np.load(npy_dataset_path)  # 读取.npy的二进制文件——人脸特征表
    # 读取name.txt中的信息，names_list是一个名字列表
    names_list = file_processing.read_data(filename, split=None, convertNum=False)
    return embeddings, names_list


def compare_embadding(pred_emb, dataset_emb, names_list, threshold=0.8):
    # 为bounding_box 匹配标签
    pred_num = len(pred_emb)  # 图片中的人脸数
    dataset_num = len(dataset_emb)  # 人脸特征库中的人脸数
    pred_name = []
    pred_score = []
    # 将从照片中提取出的特征向量与库中的特征向量作比较
    for i in range(pred_num):
        dist_list = []
        for j in range(dataset_num):
            dist = np.sqrt(np.sum(np.square(np.subtract(pred_emb[i, :], dataset_emb[j, :]))))  # 计算需要欧式距离，并求和
            dist_list.append(dist)  # 与每张脸的欧式距离
        min_value = min(dist_list)
        pred_score.append(min_value)
        print(min_value)
        if (min_value > threshold):  # 阈值设置，超过阈值说明该人脸不再数据库中
            pred_name.append('unknow')
            playsound(r"music\fail.mp3")
        else:
            pred_name.append(names_list[dist_list.index(min_value)])  # 通过最小值索引到特征库中对应的名字
            last = len(treeview.get_children())
            treeview.insert('', last + 1,
                            values=(str(last + 1), str(pred_name[0]),
                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))  # 将名字加入签到表格
            frame = Image.open(
                r"dataset\images\photo_database\\" +
                str(pred_name[i]) + "1.jpg")
            bg_image = frame.resize((image_width, image_height), Image.ANTIALIAS)
            bg_im = ImageTk.PhotoImage(bg_image)
            time1 = time.time()
            # time.sleep(10)
            playsound(r"music\success.mp3")
            count = 300
            while count != 0:
                count = count - 1
                canvas.create_image(0, 0, anchor='nw', image=bg_im)  # 显示背景
            print("欧式距离：", min_value, "小于阈值", threshold)
    return pred_name, pred_score


"""
    对摄像头发回来的照片进行计算和处理，
    需要计算得出人脸得分、人脸边框、人脸特征值
    ，处理后的照片交给facenet做识别
"""


def photo_calculation_and_processing():
    c = 0

    while True:
        draw = tkImage(True)
        frame = copy.deepcopy(draw)
        # 人脸显示
        cvimage = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
        pilImage = Image.fromarray(cvimage)
        # 第一个参数为指定宽和高的元组，第二个参数指定过滤器类型NEAREST、BILINER、BICUBIC、ANTALIAS，其中ANTALIAS为最高品质。
        pilImage = pilImage.resize((image_width, image_height), Image.ANTIALIAS)
        picture = ImageTk.PhotoImage(image=pilImage)
        canvas.create_image(0, 0, anchor='nw', image=picture)
        # 人脸检测
        showimage = draw
        image = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)  # 调整需要识别的图片，主要是转换格式
        target, bboxes, landmarks = face_detect.detect_face(image, True)  # 人脸检测,获得人脸得分、人脸框坐标、人脸特征点数据
        # 人脸得分
        # print("target:", target)
        bboxes = face_detect.get_square_bboxes(bboxes, fixed="height")  # 将图片调整为等高的
        highestscore = max(target)
        highestscore_index = target.index(highestscore)
        # for b in bboxes:  # 把找到的所有人脸框按行赋给b,bboxes是一个列表，里面放了元组，每一个元组是一个人脸框
        if highestscore >= 0.999995:
            print("已检测到人脸")
            cv2.imwrite(r'C:\Users\13906\Desktop\20190814\photos\\' + str(c) + '.jpg',
                        showimage)  # save as jpg     # 把采集到的数据放入文件夹,可作为以后的训练集
            winsound.Beep(600, 100)
            return [bboxes[highestscore_index]], [landmarks[highestscore_index]], draw  # 把数据传给facenet，其中draw是未被处理过的图片


####################################数据库###################################################
def get_face_embedding(model_path, files_list, names_list):
    '''
    获得embedding数据
    :param files_list: 图像列表
    :param names_list: 与files_list一一的名称列表
    :return:
    '''
    # 转换颜色空间RGB or BGR
    colorSpace = "RGB"
    embeddings = []  # 用于保存人脸特征数据库
    label_list = []  # 保存人脸label的名称，与embeddings一一对应
    for image_path, name in zip(files_list, names_list):
        print("processing image :{}".format(image_path))
        image = image_processing.read_image_gbk(image_path, colorSpace=colorSpace)
        # 进行人脸检测，获得bounding_box
        bboxes, landmarks = face_detect.detect_face(image, False)
        # 返回人脸框和5个关键点
        bboxes = face_detect.get_square_bboxes(bboxes)
        if bboxes == [] or landmarks == []:
            print("-----no face")
            continue
        if len(bboxes) >= 2 or len(landmarks) >= 2:
            print("-----image have {} faces".format(len(bboxes)))
            continue
        # 获得人脸区域
        face_images = image_processing.get_bboxes_image(image, bboxes, resize_height, resize_width)
        # 人脸预处理，归一化
        face_images = image_processing.get_prewhiten_images(face_images, normalization=True)
        # 获得人脸特征
        pred_emb = face_net.get_embedding(face_images)
        embeddings.append(pred_emb)
        # 可以选择保存image_list或者names_list作为人脸的标签
        # 测试时建议保存image_list，这样方便知道被检测人脸与哪一张图片相似
        # label_list.append(image_path)
        label_list.append(name)
    return embeddings, label_list


def create_face_embedding(model_path, dataset_path, out_emb_path, out_filename):
    '''

    :param model_path: faceNet模型路径
    :param dataset_path: 人脸数据库路径，每一类单独一个文件夹
    :param out_emb_path: 输出embeddings的路径
    :param out_filename: 输出与embeddings一一对应的标签
    :return: None
    '''
    # files_list存储每张图片路径的列表
    files_list, names_list = file_processing.gen_files_labels(dataset_path, postfix=['*.jpg'])
    embeddings, label_list = get_face_embedding(model_path, files_list, names_list)
    print("label_list:{}".format(label_list))
    print("have {} label".format(len(label_list)))

    embeddings = np.asarray(embeddings)
    np.save(out_emb_path, embeddings)
    file_processing.write_list_data(out_filename, label_list, mode='w')

    return label_list


def create_face_embedding_for_bzl(model_path, dataset_path, out_emb_path, out_filename):
    '''
    :param model_path: faceNet模型路径
    :param dataset_path: 人脸数据库路径，图片命名方式：张三_XXX_XXX.jpg,其中“张三”即为label
    :param out_emb_path: 输出embeddings的路径
    :param out_filename: 输出与embeddings一一对应的标签
    :return: None
    '''
    image_list = file_processing.get_images_list(dataset_path, postfix=['*.jpg', '*.png'])
    names_list = []
    for image_path in image_list:
        basename = os.path.basename(image_path)
        names = basename.split('_')[0]
        names_list.append(names)
    embeddings, label_list = get_face_embedding(model_path, image_list, names_list)
    print("label_list:{}".format(label_list))
    print("have {} label".format(len(label_list)))
    embeddings = np.asarray(embeddings)
    np.save(out_emb_path, embeddings)
    file_processing.write_data(out_filename, label_list, mode='w')


if __name__ == "__main__":
    top.mainloop()
