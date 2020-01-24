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
    # 在图像上绘制人脸边框和识别的结果
    show_info = [n + ':' + str(s)[:5] for n, s in zip(pred_name, pred_score)]
    print(show_info)