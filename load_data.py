import os
import random
from cv2 import cv2
import numpy as np
from imutils import paths
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical


def load_images(path, image_size):
    """
    从文件中读取图片，并将图片随机乱序排序
    根据分类文件夹打上标签
    :param path: 需要读取的文件路径
    :param image_size:图像张量
    :return:图片数据和标签
    """
    data = []
    labels = []
    # 获得图像路径并随机选取
    imagePaths = sorted(list(paths.list_images(path)))
    lists = sorted(os.listdir(path + "/"))
    random.seed(42)
    random.shuffle(imagePaths)

    # 在输入图像上循环
    for imagePath in imagePaths:
        # 加载图像, 对其进行预处理, 并将其存储在数据列表中
        image = cv2.imread(imagePath)
        image = cv2.resize(image, image_size)
        image = img_to_array(image)
        data.append(image)

        # 从图像路径中提取类标签, 并更新
        # labels 列表
        label = int(lists.index(imagePath.split(os.path.sep)[-2]))
        labels.append(label)

    # 将原始像素强度缩放到范围 [0,1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # 将标签从整数转换为向量
    labels = to_categorical(labels, num_classes=len(os.listdir(path)))
    return data, labels


def load_test_image(path, image_size):
    """
    读取测试的图片
    :param model: 模型
    :param path: 图片的路径
    :param image_size: 图片的大小
    :return:
    """
    image_lists = []
    # 读取路径下文件夹中的图片
    for i in os.listdir(path):
        f_path = os.path.join(path, i)
        for j in os.listdir(f_path):
            b_path = os.path.join(f_path, j)
            for k in os.listdir(b_path):
                # 加载图片并做预处理
                image_path = os.path.join(b_path, k)
                image = cv2.imread(image_path)
                # orig = image.copy()
                image = cv2.resize(image, image_size)
                image = image.astype("float") / 255.0
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)
                # 将处理后的图片添加到图片列表中
                image_lists.append([image,i ,j, image_path])

    return image_lists