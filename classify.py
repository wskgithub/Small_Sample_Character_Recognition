import numpy as np
from cv2 import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model

def classify_image(model, image_lists, kind_lists):
    """
    对顶层图片进行顶层类型的分类
    :param model: 读取进来的模型
    :param image_lists:需做分类的图片列表
    :param kind_lists:类型名称列表
    :return:null
    """
    # 从图片列表中遍历每一张需要识别的图片
    all_image_num = len(image_lists)
    num_top_true = 0
    num_all_true = 0
    last_kind = ''
    new_model = load_model('models/' + image_lists[0][1] + '.h5')
    for image in image_lists:
        # 将图片送入模型中预测
        result = model.predict(image[0])[0]
        # 取出相似度最高的一项
        proba = np.max(result)
        # 获得识别出类型的标签
        label = kind_lists[int(np.where(result == proba)[0])]
        top_true, all_true = classify_bottom_kind(new_model, image, label)

        if top_true == True:
            num_top_true = num_top_true + 1
            if last_kind != label:
                # 加载对应模型
                new_model = load_model('models/' + label + '.h5')
                last_kind = label
            if all_true == True:
                num_all_true = num_all_true + 1

    print("分类图片总数:{}, 顶层类型分类正确:{}, 全部分类正确:{}, 顶层分类正确率:{}%, 全部分类正确率:{}%"
          .format(all_image_num, num_top_true, num_all_true, (num_top_true/all_image_num) * 100,
          (num_all_true/all_image_num) * 100))


def classify_bottom_kind(model, image, label):
    """
    根据上层分类结果对图片进行子类划分
    :param model: 子类模型
    :param image: 需进行分类的图片
    :param label: 标签
    :return: 分类是否正确的结果
    """
    top_true = False
    all_true = False

    # 将图片送入模型中预测
    result = model.predict(image[0])[0]
    # 取出相似度最高的一项
    proba = np.max(result)
    # 输出分类结果
    labels =label + ' -> character' + str(int((np.where(result == proba)[0]) + 1)).zfill(2)
    print("{}: {:.2f}%".format(labels, proba * 100))
    # 将结果写入分类日志文件
    with open("log.txt", "a") as f:
        f.write(labels + " -> " + str(proba * 100) + " -> path:" + image[3] + "\n")
    # 对分类结果进行判断，并设置相应标识
    if label==image[1]:
        top_true = True
        if image[2] == 'character' + str(int((np.where(result == proba)[0]) + 1)).zfill(2):
            all_true = True
    return top_true, all_true


def classify_one_image(image_path, image_size, kind_lists):
    """
    对单张图片进行分类
    :param image_path: 需分类图片路径
    :param image_size: 图片尺寸
    :param kind_lists: 上层类型列表
    :return: null
    """
    # 读取图片，并作预处理
    image = cv2.imread(image_path)
    image = cv2.resize(image, image_size)
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # 读取上层分类模型
    model = load_model('models/evaluation_top_model.h5')
    # 将图片送入模型中预测
    result = model.predict(image)
    # 取出相似度最高的一项
    proba = np.max(result)
    # 获得识别出类型的标签
    label = kind_lists[int(np.where(result == proba)[0])]
    # 读取子类分类模型
    new_model = load_model('models/' + label + '.h5')
    new_result = new_model.predict(image)
    new_proba = np.max(new_result)
    # 输出分类结果
    labels =label + ' -> character' + str(int((np.where(new_result == new_proba)[0]) + 1)).zfill(2)
    print("{}: {:.2f}%".format(labels, new_proba * 100))