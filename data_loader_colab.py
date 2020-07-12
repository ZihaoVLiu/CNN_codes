import os
import cv2
import time
import h5py
from PIL import Image
from data_augmentation import *
import matplotlib.pyplot as plt
import numpy as np
import random

base_path = '/Users/zihaoliu/Desktop/Graduate_UW/SYDE660/CNN/data'

# read_path_train = "{}/train".format(base_path)
save_path_train = "{}/train_resize".format(base_path)
gray_path_train = "{}/train_resize_gray".format(base_path)
h5_train = "{}/train_resize_gray_h5.h5".format(base_path)

# read_path_test = "{}/test".format(base_path)
save_path_test = "{}/test_resize".format(base_path)
gray_path_test = "{}/test_resize_gray".format(base_path)
h5_test = "{}/test_resize_gray_h5.h5".format(base_path)

txt_train_file = "{}/train_split_v3.txt".format(base_path)
txt_test_file = "{}/test_split_v3.txt".format(base_path)

covid_aug_path = "{}/covid_19_augmentation".format(base_path)

try_path_train = '{}/read_image_train'.format(base_path)
try_txt_train = '{}/try_txt_train.txt'.format(base_path)
try_path_test = '{}/read_image_test'.format(base_path)
try_txt_test = '{}/try_txt_test.txt'.format(base_path)


def rgb2gray(read_path, save_path):
    tic = time.time()
    for filename in os.listdir(read_path):
        if (filename == '.DS_Store'):
            continue
        print(filename)
        read_image_path = read_path + '/' + filename
        image = cv2.imread(read_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        save_image_path = save_path + '/' + filename
        cv2.imwrite(save_image_path, gray)
    toc = time.time()
    print('RGB to Gray time is %.2f.' % (toc - tic))


def get_class_info(txt_file, image_path):
    '''
    pair all the image name into different class dictionaries
    :param txt_file: .txt path
    :param image_path: .image directory path
    :return: 3 dictionaries
    '''
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


def get_image_path(dicts, sample_number=0, add_aug=False):
    '''
    pair all image names and classes into two lists in order
    :param dicts:
    :return: two lists, one is image name, another is class label
    '''
    dict_pneumonia, dict_normal, dict_covid = dicts
    list_pneumonia = list(dict_pneumonia.values())
    list_normal = list(dict_normal.values())
    list_covid = list(dict_covid.values())
    if bool(add_aug):
        aug_data, aug_label = get_augmentation_info(add_aug)
        list_covid += aug_data
        print("The total number of COVID-19 is %d" % len(list_covid))
        print('(There are', len(aug_data), 'augmentation images added.)')
    if not sample_number:
        lists_data = list_pneumonia + list_normal + list_covid
        lists_label = [0] * len(list_pneumonia) + [1] * len(list_normal) + [2] * len(list_covid)
    else:
        np.random.shuffle(list_pneumonia), np.random.shuffle(list_normal), np.random.shuffle(list_covid)
        lists_data = list_pneumonia[:sample_number] + list_normal[:sample_number] + list_covid[:sample_number]
        lists_label = [0] * sample_number + [1] * sample_number + [2] * sample_number
        print('%d images are selected from each of class.' % sample_number)
    return lists_data, lists_label

def random_image(lists):
    '''
    shuffle the lists
    :param lists:
    :return: two shuffled lists
    '''
    lists_data, lists_label = lists
    indexes = list(range(lists_label.size))
    np.random.shuffle(indexes)
    lists_data = lists_data[indexes]
    lists_label = lists_label[indexes]
    return lists_data, lists_label


def random_covidx(data, label):
    pneumonia_number = np.sum(label == 0)
    normal_number = np.sum(label == 1)
    covid_number = np.sum(label == 2)
    index_p = list(range(pneumonia_number))
    index_n = list(range(pneumonia_number, pneumonia_number + normal_number))
    index_c = list(range(pneumonia_number + normal_number, label.size))
    np.random.shuffle(index_p)
    np.random.shuffle(index_n)
    np.random.shuffle(index_c)
    indexes = index_p + index_n + index_c
    data_shuffle = data[indexes]
    label_shuffle = label[indexes]
    print('Data and label shuffling.')
    return data_shuffle, label_shuffle


def get_im_cv2(img_names, path, batch_size):
    '''
    read all images into a 4 dimensionality np array
    :param img_names: a list
    :param path: image directory
    :param batch_size:
    :return: 4 dimensionality np array
    '''
    imgs = np.zeros((batch_size, 480, 480, 3))
    for index, item in enumerate(img_names):
        img_path = path + '/' + item
        img = cv2.imread(img_path)
        img = img / 255
        imgs[index] = img
    return imgs


def load_covidx(h5_path, sample_number=0, isRGB=False):
    '''
    the combination of get_class_info() and gey_image_path()
    :param txt_train_file: .txt directory
    :param save_path_train: image directory
    :param add_aug: the path of augmentation images path
    :return: two lists
    '''
    data, label = load_h5_as_np(h5_path, isRGB)
    pneumonia_number = np.sum(label == 0)
    normal_number = np.sum(label == 1)
    covid_number = np.sum(label == 2)
    print("The number of pneumonia is %d" % pneumonia_number)
    print("The number of normal is %d" % normal_number)
    print("The number of COVID-19 is %d" % covid_number)
    data, label = random_covidx(data, label)
    if sample_number:
        pneumonia_data = data[0:sample_number]
        normal_data = data[pneumonia_number:pneumonia_number + sample_number]
        covid_data = data[pneumonia_number + normal_number:pneumonia_number + normal_number + sample_number]
        data_samples = np.concatenate((pneumonia_data, normal_data, covid_data))
        pneumonia_label = label[0:sample_number]
        normal_label = label[pneumonia_number:pneumonia_number + sample_number]
        covid_laebl = label[pneumonia_number + normal_number:pneumonia_number + normal_number + sample_number]
        label_samples = np.concatenate((pneumonia_label, normal_label, covid_laebl))
        print('%d images are selected from each of class.' % sample_number)
        return data_samples, label_samples
    else:
        return data, label



def convert_to_one_hot_matrix(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def get_train_batch(data, label, batch_size):
    '''
    a data generator to avoid out of memory for Keras.fit_generator() method
    :param lists_data:
    :param lists_label:
    :param batch_size:
    :param path:
    :return: a np.array of images, and a np.array of one-hot class labels for training set
    '''
    data_number = label.size
    while True:
        for i in range(0, data_number, batch_size):
            datas = data[i: i + batch_size]
            datas = datas / 255
            labels = label[i: i + batch_size]
            labels = convert_to_one_hot_matrix(labels, 3).T
            yield (datas, labels)
            # yield ({'input': datas}, {'output': labels})


def get_valid_batch(data, label, batch_size):
    '''
    same as previous function, but for testing set
    :param lists_data:
    :param lists_label:
    :param batch_size:
    :param path:
    :return:
    '''
    data_number = label.size
    while True:
        for i in range(0, data_number, batch_size):
            datas = data[i: i + batch_size]
            datas = datas / 255
            labels = label[i: i + batch_size]
            labels = convert_to_one_hot_matrix(labels, 3).T
            yield (datas, labels)


def get_fold_valid(lists_data, batch_size):
    '''
    combination all the function mentioned above, generating two generator (for testing and validation),
    and step number for both sets
    :param lists_data: train data and label lists
    :param batch_size:
    :param path:
    :return: two generators and two int numbers
    '''
    lists_data_train, lists_label_train = random_image(lists_data)
    data_number = lists_label_train.size
    train_size = int(data_number * 0.9)
    valid_size = data_number - train_size
    print("Training size is " + str(train_size))
    print("Validation size is " + str(valid_size))
    train_remainder = train_size % batch_size
    valid_remainder = valid_size % batch_size
    train_generator = get_train_batch(lists_data_train[:(train_size - train_remainder)],
                                      lists_label_train[:(train_size - train_remainder)], batch_size)
    valid_generator = get_valid_batch(lists_data_train[train_size:(data_number - valid_remainder)],
                                      lists_label_train[train_size:(data_number - valid_remainder)], batch_size)
    step_per_epoch = train_size // batch_size
    validation_steps = valid_size // batch_size
    if validation_steps == 0:
        validation_steps = 1
    print('Actual training number used (divisible by batchsize): ' + str(train_size - train_remainder))
    print('Actual validation number used (divisible by batchsize): ' + str(valid_size - valid_remainder))
    print('Step per epoch is ' + str(step_per_epoch))
    print('Validation steps is ' + str(validation_steps))
    return train_generator, valid_generator, step_per_epoch, validation_steps


def get_test_data(lists_data, batch_size):
    '''
    same as previous function, but return only one testing set generator and one step number
    :param lists_data:
    :param batch_size:
    :param path:
    :return:
    '''
    lists_data_test, lists_label_test = lists_data
    test_size = lists_label_test.size
    print("Testing batch size is " + str(test_size))
    test_remainder = test_size % batch_size
    test_generator = get_train_batch(lists_data_test[:(test_size - test_remainder)],
                                      lists_label_test[:(test_size - test_remainder)], batch_size)
    steps = test_size // batch_size
    print('Actual testing number used (divisible by batchsize): ' + str(test_size - test_remainder))
    print('Number of step is ' + str(steps))
    return test_generator, steps


def plot(history):
    '''
    draw the loss and accuracy for training and validation sets
    :param history:
    :return: None
    '''
    epochs = len(history.history['loss'])

    plt.subplot(2, 1, 1)
    plt.scatter(range(epochs), history.history['loss'], label='loss')
    plt.scatter(range(epochs), history.history['val_loss'], label='val_loss')
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc='upper right', fancybox=True)

    plt.subplot(2, 1, 2)
    plt.scatter(range(epochs), history.history["accuracy"], label="accuracy")
    plt.scatter(range(epochs), history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right', fancybox=True)

    plt.savefig("plot.png", dpi=300)
    plt.show()


def save_as_h5(txt_file, image_path, save_path):
    tic = time.time()
    dicts_train = get_class_info(txt_file, image_path)
    lists_data, lists_label = get_image_path(dicts_train)
    sample_numbers = len(lists_data)
    np_images = np.zeros((sample_numbers, 480, 480))
    np_label = np.zeros((sample_numbers, 1))
    for index in range(sample_numbers):
        filename = lists_data[index]
        print(filename)
        read_image_path = image_path + '/' + filename
        image = cv2.imread(read_image_path, cv2.IMREAD_GRAYSCALE)
        np_images[index] = image
        np_label[index] = lists_label[index]
    if not os.path.exists(save_path):
        h5f = h5py.File(save_path, 'w')
        h5f.create_dataset('data', data=np_images, dtype='uint8')
        h5f.create_dataset('label', data=np_label, dtype='uint8')
        h5f.close()
    toc = time.time()
    print('.h5 file generating time is %.2f' % (toc - tic))


def load_h5_as_np(h5_path, isRGB=False):
    f = h5py.File(h5_path, 'r')
    data = np.array(f["data"][:])
    label = np.array(f["label"][:])
    if isRGB:
        data = gray2rgb(data)
        return data, label
    data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
    return data, label

def gray2rgb(data):
    number = data.shape[0]
    rgbs = np.zeros((number, data.shape[1], data.shape[2], 3), dtype='uint8')
    for index, img in enumerate(data):
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        rgbs[index] = rgb
        # if (index % 100 == 0):
        #     print('%d Gray to RGB done.' % (index // 100))
    print('Gray to RGB done.')
    return rgbs


