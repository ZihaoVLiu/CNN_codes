import tensorflow as tf
import h5py
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
from matplotlib import pyplot as plt


def draw_confusion_matrix(lists_test, y_pred, is_norm=False):
    # offset the labels
    labels = ('pneumonia', 'normal', 'COVID-19')
    tick_marks = np.array(range(len(labels))) + 0.5

    y_test = lists_test[1]
    y_test = y_test[:y_pred.shape[0]]
    con_mat = confusion_matrix(y_test, y_pred.argmax(axis=1))
    if is_norm:
        con_mat = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
        con_mat = np.around(con_mat, decimals=2)
    sns.heatmap(con_mat, annot=True, cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.grid(True, which='minor', linestyle='-')
    plt.show()

def get_performance(lists_test, y_pred, average='weighted'):
    y_test = lists_test[1]
    y_test = y_test[:y_pred.shape[0]]
    y_pred_one = y_pred.argmax(axis=1)
    accuracy = metrics.accuracy_score(y_test, y_pred_one)
    precision = metrics.precision_score(y_test, y_pred_one, average=average)
    recall = metrics.recall_score(y_test, y_pred_one, average=average)
    f1 = metrics.f1_score(y_test, y_pred_one, average=average)
    return accuracy, precision, recall, f1

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

def load_h5_as_np(h5_path, isRGB=False):
    f = h5py.File(h5_path, 'r')
    data = np.array(f["data"][:])
    label = np.array(f["label"][:])
    if isRGB:
        data = gray2rgb(data)
        return data, label
    data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
    return data, label

def random_covidx(data, label):
    number = label.size
    index = list(range(number))
    np.random.shuffle(index)
    data_shuffle = data[index]
    label_shuffle = label[index]
    return data_shuffle, label_shuffle


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
        data, label = random_covidx(data_samples, label_samples)
        return data, label
    else:
        data, label = random_covidx(data, label)
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
    # print("Testing batch size is " + str(test_size))
    test_remainder = test_size % batch_size
    test_generator = get_train_batch(lists_data_test[:(test_size - test_remainder)],
                                      lists_label_test[:(test_size - test_remainder)], batch_size)
    steps = test_size // batch_size
    print('Actual testing number used (divisible by batchsize): ' + str(test_size - test_remainder))
    print('Number of step is ' + str(steps))
    return test_generator, steps


def predict_image(image_path, model):
    '''
    predict the class of input data
    :param image_path: a path string or a list of path string
    :param model: the pretrained model
    :return: a np.array
    '''
    def predict(image_path):
        image = cv2.imread(image_path)
        image = image / 255
        return image
    if type(image_path) == str:
        image = predict(image_path)
        image = image.reshape(1, 480, 480, 3)
        result = model.predict(image, batch_size=1)
        predict_class = np.argmax(result)
        if predict_class == 0:
            predict_label = 'pneumonia'
        elif predict_class == 1:
            predict_label = 'normal'
        else:
            predict_label = 'COVID-19'
        print('The prediction result is ' + predict_label)
        return result
    if type(image_path) == list:
        image_number = len(image_path)
        batch_image = np.zeros((image_number, 480, 480, 3))
        for i, path in enumerate(image_path):
            batch_image[i] = predict(path)
        result = model.predict(batch_image, batch_size=image_number)
        print('Prediction results have been returned.')
        return result
    else:
        print('Input format error.')
        return


def score_evaluate(scores, threshold):
    '''
    evaluate scores which smaller than the threshold
    :param scores: np.array returned by predict_image() function
    :param threshold: a threshold to filter the score
    :return: indexes which score greater than the threshold
    '''
    max_score = np.max(scores, axis=1)
    count_no_good = np.sum(max_score < threshold)
    index = np.where(max_score < threshold)
    print('The number of prediction score less than %.2f is %d' % (threshold, int(count_no_good)))
    return index





if __name__ == '__main__':
    # change the base path to your directory.
    base_path = '/Users/zihaoliu/Desktop/Graduate_UW/SYDE660/CNN/data'
    model_path = '{}/Xception.h5'.format(base_path)
    test_set_path = '{}/test_resize_gray_h5.h5'.format(base_path)
    # load the Xception model.
    model = tf.keras.models.load_model(model_path)

    # if you want to load all testing images file into the memory, uncomment the following 4 lines.
    # data, label = load_h5_as_np(test_set_path, isRGB=True)
    # data, label = random_covidx(data, label)
    # label = convert_to_one_hot_matrix(label, 3).T
    # # under Mac OS system (GPU not allowed), the CNN prediction processing is very slow
    # # therefore, only first three images are loaded to predict
    # prediction = model.predict(data[0:3]/255, batch_size=3)

    # or, you can load some images you interested
    image_path = 'The image directory you want to predict, or a bunch of images (image directories in a list)'
    scores = predict_image(image_path, model)
    index = score_evaluate(scores, 0.8)


    '''
    if you want to verify the accuracy of test set on google colab, 
    you must copy all the code into a python notebook, and execute the following codes
    '''
    print('Testing stage starts.')
    # you can load all images from the .h5 file if you set sample_number = 0
    # or, you can choose the sample number of each class, such as 100 (totally 300),
    # In order to save space, the images in the h5 file are grayscale images, therefore,
    # it is necessary to use the grayscale to rgb function in cv2 library (set isRGB to True).
    dataset = load_covidx(test_set_path, sample_number=100, isRGB=True)
    print("Number of test examples = " + str(len(dataset[0])))
    # In order to avoid GPU memory overflow, you need to use the generator to feed the images
    # to the model in batches (you can change batch_size to an appropriate number, such as 32, 64)
    generator_test, steps = get_test_data(dataset, batch_size=100)

    # Make predictions and draw a confusion matrix
    preds = model.predict_generator(generator=generator_test, steps=steps, verbose=1)
    # Never run draw_confusion_matrix() function independently,
    # be sure to run all functions in order before running this function
    draw_confusion_matrix(dataset, preds, is_norm=True)
    performance = get_performance(dataset, preds)
    print('Accuracy: %.6f \nPrecision: %.6f \nRecall: %.6f \nF1-score: %.6f \n' % performance)
    # evaluate = model.evaluate_generator(generator=generator_test, steps=steps, verbose=1)

    # then you can uncomment and execute the following code to see which samples' score are closer
    # index = score_evaluate(preds, 0.7)
    # print(preds[index])
    # # actual class
    # print(dataset[1][index])
