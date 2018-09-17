import os

def get_file_list(path):
    """
    获取小类文件列表
    :param path:比赛数据集路径
    :return:各大类的文件路径列表
    """
    lists = []
    for file in os.listdir(path):
        lists.append(os.path.join(path, file))
    return lists