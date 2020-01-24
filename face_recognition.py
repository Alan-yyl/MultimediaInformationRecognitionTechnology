# -*-coding: utf-8 -*-
import facenet
import tensorflow as tf
import align.detect_face as detect_face
import numpy as np


class facenetEmbedding:
    def __init__(self, model_path):
        self.sess = tf.InteractiveSession()  # 可以理解为开始使用tensorflow
        self.sess.run(tf.global_variables_initializer())
        # Load the model加载facenet模型
        facenet.load_model(model_path)
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.tf_embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    def get_embedding(self, images):
        feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
        embedding = self.sess.run(self.tf_embeddings, feed_dict=feed_dict)
        return embedding

    def free(self):
        self.sess.close()


# 人脸检测类
class Facedetection:

    def __init__(self):
        # minimum size of face所认为的图片中需要识别的人脸的最小尺寸，minsize越大，生成的“金字塔”层数越少，resize和pnet的计算量越小。
        self.minsize = 30
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold阈值？？？
        self.factor = 0.709  # scale factor生成图片金字塔时使用的缩放因子即每次对边缩放的倍数
        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            #     # gpu_memory_fraction = 1.0
            #     # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            #     # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            sess = tf.Session()
            with sess.as_default():
                # 在使用mtcnn模型时必须先调用detect_facec的creat_mtcnn方法导入网络结构，
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)

    def detect_face(self, image, isscore, fixed=None):
        '''
        mtcnn人脸检测，
        PS：人脸检测获得bboxes并不一定是正方形的矩形框，参数fixed指定等宽或者等高的bboxes
        :param image:
        :param fixed:
        :parm ifscore:标记是否需要返回人脸得分
        :return:
        '''
        # bboxes为返回的人脸框，是一个n*5的数组，n表示检测到的人脸数，每一行的5个元素是一个行数组，表示一个人脸框的信息。
        # landmark为关键点信息，是一个10*n的数组，n表示检测到的人脸数，每一列的10元素为人脸关键点信息

        bboxes, landmarks = detect_face.detect_face(image, self.minsize, self.pnet, self.rnet, self.onet,
                                                    self.threshold, self.factor)
        landmarks_list = []
        landmarks = np.transpose(landmarks)  # 对landmarsk进行转置
        if isscore is True:
            if len(bboxes) != 0:
                scores = [b[4] for b in bboxes if b[4] != 0]
                print(scores)
            else:
                scores = [0]
        bboxes = bboxes.astype(int)  # 将bboxes中元素类型转为int
        bboxes = [b[:4] for b in bboxes]  # 将人脸的score值去除
        for landmark in landmarks:
            face_landmarks = [[landmark[j], landmark[j + 5]] for j in range(5)]
            landmarks_list.append(face_landmarks)  # face_landmarks依次存放每张照片左眼、右眼、鼻子、左嘴角、右嘴角的坐标
        if fixed is not None:
            bboxes, landmarks_list = self.get_square_bboxes(bboxes, landmarks_list, fixed)  # 调整为正方形
        if isscore is True:
            return scores, bboxes, landmarks_list
        else:
            return bboxes, landmarks_list

    def get_square_bboxes(self, bboxes, fixed="height"):
        '''
        获得等宽或者等高的bboxes
        :param bboxes:
        :param fixed: width or height
        :return:
        '''
        new_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            center_x, center_y = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            if fixed == "height":
                dd = h / 2
            elif fixed == 'width':
                dd = w / 2
            x11 = int(center_x - dd)
            y11 = int(center_y - dd)
            x22 = int(center_x + dd)
            y22 = int(center_y + dd)
            new_bbox = (x11, y11, x22, y22)
            new_bboxes.append(new_bbox)
        return new_bboxes

    def detection_face(img):
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            # gpu_memory_fraction = 1.0
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            sess = tf.Session()
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
                bboxes, landmarks = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        landmarks = np.transpose(landmarks)
        bboxes = bboxes.astype(int)
        bboxes = [b[:4] for b in bboxes]
        landmarks_list = []
        for landmark in landmarks:
            face_landmarks = [[landmark[j], landmark[j + 5]] for j in range(5)]
            landmarks_list.append(face_landmarks)
        return bboxes, landmarks_list
