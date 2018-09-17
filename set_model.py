from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential, load_model

def creat_model(filters, kernel_size, input_shape, pool_size, nb_classes):
    """
    初始化模型，构建卷积成和全连接层
    :param filters:卷积滤波器数量
    :param kernel_size:卷积核大小
    :param input_shape:图像张量
    :param pool_size:池化缩小比例因素
    :param nb_classes:分类数
    :return:初始化后的CNN模型
    """
    # 生成模型
    model = Sequential()

    #####特征层#####
    # 第一个卷积层
    model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation='relu',
                     input_shape=input_shape))
    # 池化
    model.add(MaxPooling2D(pool_size=pool_size))
    # 第二个卷积层
    model.add(Conv2D(filters=filters*2, kernel_size=kernel_size, activation='relu'))
    # 池化
    model.add(MaxPooling2D(pool_size=pool_size))
    # 第三个卷积层
    model.add(Conv2D(filters=filters*2*2, kernel_size=kernel_size, activation='relu'))
    # 第四个卷积层
    # model.add(Conv2D(filters=filters*2*2*2, kernel_size=kernel_size, activation='relu'))
    #池化
    model.add(MaxPooling2D(pool_size=pool_size))
    #####全链接层#####
    # 压缩维度
    model.add(Flatten())
    # 全链接层
    model.add(Dense(128, activation='relu'))
    # 模型平均，防止过拟合
    model.add(Dropout(0.5))
    # Softmax分类
    model.add(Dense(nb_classes, activation='softmax'))
    return model


def reset_model(model_name, nb_classes):
    """
    加载已训练好的模型
    冻结原模型的特征提取层
    重新设置全连接层
    :param model_name:模型名称
    :param nb_classes:分类数
    :return:重新设置后的模型
    """
    # 加载已训练好的模型
    model = load_model(model_name)
    # model.summary()
    # 锁定模型的特征层
    conv2d_1 = model.get_layer(name='conv2d_1')
    conv2d_1.trainable = False
    conv2d_2 = model.get_layer(name='conv2d_2')
    conv2d_2.trainable = False
    conv2d_3 = model.get_layer(name="conv2d_3")
    conv2d_3.trainable = False
    # 删除全连接层
    model.pop()
    model.pop()
    model.pop()
    # 重新创建全连接层
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    # model.add(Dense(nb_classes, activation='softmax', name='dense_2'))
    # model.summary()
    return model