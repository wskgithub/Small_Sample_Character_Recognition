import argparse
import os
import time

from classify import classify_image, classify_one_image
from get_list import get_file_list
from set_model import creat_model, reset_model
from load_data import load_images, load_test_image
from train_model import train, train_subclass
from keras.models import load_model

FILTERS = 32 # 卷积滤波器数量
IMAGE_SIZE = (105,105) # 图像缩放大小
KERNEL_SIZE = (3,3) # 卷积核大小
INPUT_SHAPE = (105,105,3) # 图像张量
POOL_SIZE = (2,2) # 池化缩小比例因素
NB_CLASSES = 0 # 分类数
EPOCHS = 50 # 循环的次数
KIND_LISTS = ['Angelic','Atemayar_Qelisayer', 'Atlantean', 'Aurek-Besh', 'Avesta', 'Ge_ez', 'Glagolitic',
              'Gurmukhi', 'Kannada', 'Keble', 'Malayalam', 'Manipuri', 'Mongolian', 'Old_Church_Slavonic_(Cyrillic)',
              'Oriya', 'Sylheti', 'Syriac_(Serto)', 'Tengwar', 'Tibetan', 'ULOG']

def args_parse():
    ap = argparse.ArgumentParser()
    # 功能选择
    """
    -f
    1:train background model
    2:train evaluation top model
    3:train evaluation model
    """
    ap.add_argument("-f", "--function", default=False, type=int,
                    help="Function Selection")

    """
    -c
    1:To classify pictures in batches
    2:Classify a single picture
    """
    # 图片分类模式选择
    ap.add_argument("-c", "--classify", default=False, type=int,
                    help="Test classify")

    # 数据集路径
    ap.add_argument("-tr", "--train_path", default=None, type=str,
                    help="path to train images")
    ap.add_argument("-ts", "--test_path", default=None, type=str,
                    help="path to test images")
    return ap.parse_args()


# 主函数
if __name__=='__main__':

    print('[INFO] start...')
    args = args_parse()

    if args.function == 1:
        # 利用参考数据集训练模型
        # 扩展参考数据集路径
        background_train_path = args.train_path
        # 参考测试数据集路径
        background_test_path = args.test_path
        # 读取参考训练数据和测试数据
        print("[INFO] loading background images...")
        x_train_background, y_train_background = load_images(background_train_path, IMAGE_SIZE)
        x_test_background, y_test_background = load_images(background_train_path, IMAGE_SIZE)
        # 初始化模型
        print("[INFO] initialize background model...")
        background_model = creat_model(FILTERS, KERNEL_SIZE, INPUT_SHAPE, POOL_SIZE,
                                            len(os.listdir(background_train_path)))
        # 训练模型
        print("[INFO] compiling background model...")
        train(background_model, x_train_background, y_train_background, x_test_background, y_test_background,
              len(y_train_background) // 10, EPOCHS, 'models/backgroud_model.h5')
    elif args.function == 2:
        # 使用迁移学习训练比赛集
        # 比赛训练数据集顶层类型(大类)
        evaluation_top_train_path = args.train_path
        # 比赛测试数据集
        evaluation_top_test_path = args.test_path
        # 读取比赛训练数据和测试数据
        print("[INFO] loading evaluation images...")
        x_train_top_evaluation, y_train_top_evaluation = load_images(evaluation_top_train_path, IMAGE_SIZE)
        x_test_top_evaluation, y_test_top_evaluation = load_images(evaluation_top_test_path, IMAGE_SIZE)
        # 加载并设置模型
        print('[INFO] loading model...')
        evaluation_model = reset_model('models/background_model.h5', len(os.listdir(evaluation_top_train_path)))
        # 训练模型
        print("[INFO] compiling evaluation model...")
        train(evaluation_model, x_train_top_evaluation, y_train_top_evaluation,
              x_test_top_evaluation, y_test_top_evaluation, len(y_train_top_evaluation) // 10,
              EPOCHS, 'models/evaluation_top_model.h5')

    elif args.function == 3:
        # 使用大类训练好的模型对每个大类中的小类进行分类器的训练
        # 比赛数据训练集路径(按小类划分)
        evaluation_train_path = args.train_path
        # 比赛数据测试集路径(按小类划分)
        evaluation_test_path = args.test_path
        # 获取训练集和测试集文件列表
        train_kind_lists = get_file_list(evaluation_train_path)
        test_kind_lists = get_file_list(evaluation_test_path)
        # 训练模型
        train_subclass(train_kind_lists, test_kind_lists, IMAGE_SIZE, EPOCHS)

    if args.classify == 1:
        # 对图片进行批量分类
        print('[INFO] start...')
        time_start = time.time()
        # 顶层类型模型的地址
        top_model_path = 'models/evaluation_top_model.h5'
        # 读取用于顶层分类的模型
        top_model = load_model(top_model_path)
        # 需分类的图片路径
        classify_image_path = args.test_path
        image_lists = load_test_image(classify_image_path, IMAGE_SIZE)
        # 开始送入模型分类
        classify_image(top_model, image_lists, KIND_LISTS)
        time_end = time.time()
        print('运行用时:{}'.format(time_end - time_start))
    elif args.classify == 2:
        # 对单张图片进行分类
        classify_one_image_path = args.test_path
        classify_one_image(classify_one_image_path, IMAGE_SIZE, KIND_LISTS)