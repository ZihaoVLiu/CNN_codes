import os
import cv2
import time
import h5py
from PIL import Image
import numpy as np
import random

base_path = '/Users/zihaoliu/Desktop/Graduate_UW/SYDE660/CNN/data'

read_path_test = "/Users/zihaoliu/Desktop/Graduate_UW/SYDE660/CNN/data/test"
save_path_test = "/Users/zihaoliu/Desktop/Graduate_UW/SYDE660/CNN/data/test_resize"

read_path_train = "/Users/zihaoliu/Desktop/Graduate_UW/SYDE660/CNN/data/train"
save_path_train = "/Users/zihaoliu/Desktop/Graduate_UW/SYDE660/CNN/data/train_resize"

txt_train_file = "/Users/zihaoliu/Desktop/Graduate_UW/SYDE660/CNN/data/train_split_v3.txt"
txt_test_file = "/Users/zihaoliu/Desktop/Graduate_UW/SYDE660/CNN/data/test_split_v3.txt"

try_path_train = '{}/read_image_train'.format(base_path)
try_txt_train = '{}/try_txt_train.txt'.format(base_path)
try_path_test = '{}/read_image_test'.format(base_path)
try_txt_test = '{}/try_txt_test.txt'.format(base_path)

def get_class_info(txt_file, image_path):
    with open(txt_file) as file_obj:
        lines = file_obj.readlines()
        image_number = len(lines)
        image_name_list = os.listdir(image_path)
        dict_pneumonia = {}
        dict_normal = {}
        dict_covid = {}
        for index, line in enumerate(lines):
            line_split = line.split()
            image_name = line_split[1]
            image_label = line_split[2]
            if image_name in image_name_list:
                if image_label == 'pneumonia':
                    dict_pneumonia[index] = image_name
                elif image_label == 'normal':
                    dict_normal[index] = image_name
                elif image_label == 'COVID-19':
                    dict_covid[index] = image_name
    print("The number of pneumonia is %d" % len(dict_pneumonia))
    print("The number of normal is %d" % len(dict_normal))
    print("The number of COVID-19 is %d" % len(dict_covid))
    return dict_pneumonia, dict_normal, dict_covid


def get_image_path(dicts):
    dict_pneumonia, dict_normal, dict_covid = dicts
    list_pneumonia = list(dict_pneumonia.values())
    list_normal = list(dict_normal.values())
    list_covid = list(dict_covid.values())
    lists_data = list_pneumonia + list_normal + list_covid
    lists_label = [0] * len(list_pneumonia) + [1] * len(list_normal) + [2] * len(list_covid)
    return lists_data, lists_label

def random_image(lists):
    lists_data, lists_label = lists
    indexes = list(range(len(lists_label)))
    indexes = random.sample(indexes, len(lists_label))
    lists_data = np.array(lists_data)[indexes]
    lists_label = np.array(lists_label)[indexes]
    return lists_data.tolist(), lists_label.tolist()


def get_im_cv2(img_names, path, batch_size):
    imgs =  np.zeros((batch_size, 480, 480, 3))
    for index, item in enumerate(img_names):
        img_path = path + '/' + item
        img = cv2.imread(img_path)
        img = img / 255
        imgs[index] = img
    return imgs


def load_covidx(txt_train_file, save_path_train):
    dicts_train = get_class_info(txt_train_file, save_path_train)
    lists_train = get_image_path(dicts_train)
    lists_data_train, lists_label_train = random_image(lists_train)

    return lists_data_train, lists_label_train


def convert_to_one_hot_matrix(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def get_train_batch(lists_data, lists_label, batch_size, path):
    while True:
        for i in range(0, len(lists_data), batch_size):
            datas = get_im_cv2(lists_data[i: i + batch_size], path, batch_size)
            labels = np.array(lists_label[i: i + batch_size])
            labels = convert_to_one_hot_matrix(labels, 3).T
            yield (datas, labels)
            # yield ({'input': datas}, {'output': labels})

