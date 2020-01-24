# 前端库
from tkinter import *
from tkinter import ttk
import cv2 as cv
from PIL import Image, ImageTk
from tkinter.simpledialog import askstring
import threading
from identify.identification import face_recognition_image

# 前端声明
camera_switch = False
count = 1

# 临时变量
tempimagepath = r"..\dataset\images\photo_database\\"

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
top.wm_title("人脸考勤系统")
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
bg_image1 = Image.open(r"img\SSPU_2.png")
bg_image2 = bg_image1.resize((image_width, image_height), Image.ANTIALIAS)
bg_im = ImageTk.PhotoImage(bg_image2)
canvas.create_image(0, 0, anchor='nw', image=bg_im)  # 显示背景
b1 = Button(top, text='打开/关闭摄像头', width=15, height=2, command=button1)
b2 = Button(top, text="拍照", width=15, height=2, command=button2)
b3 = Button(top, text="签到", width=15, height=2, command=lambda: thread_it(button3))
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

if __name__ == "__main__":
    top.mainloop()
