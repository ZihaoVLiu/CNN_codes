from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

'''
1. Conduct data_augmentation() function
2. Run the write_txt() function to add augmented images information on .txt files.\
3. Done. You can run the Liu-ShiNet.py file
Note: Maybe you can use data augmentation on testing set, but I did not.
'''

def data_augmentation(number, dict_covid, image_path, save_path):
    '''
    conduct data augmentation
    :param number: the augmented number of each images
    :param dict_covid: the dictionary returned by get_class_info(txt_file, image_path) function
    :param image_path: where to get the images
    :param save_path: where to save
    :return: None
    '''
    data_gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    print('Data augmentation starts.')
    for name in dict_covid.values():
        img = load_img(image_path + '/' + name)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in data_gen.flow(x, batch_size=1, save_to_dir=save_path, save_prefix='aug_'+name[:-4], save_format='png'):
            i += 1
            if i == number:
                break
        print(name, 'augmentation done.')
    print('Total', len(dict_covid), 'image augmentation done.')


def get_augmentation_info(save_path):
    '''
    get a list of augmented image names
    :param save_path: where did the augmented images save
    :return: two lists
    '''
    aug_name_list = []
    name_list = os.listdir(save_path)
    for name in name_list:
        if name[:4] == 'aug_':
            aug_name_list.append(name)
    aug_label_list = [2] * len(aug_name_list)
    return aug_name_list, aug_label_list


def write_txt(txt_path, save_path):
    '''
    write all image names and class labels into the given .txt file
    :param txt_path:
    :param save_path:
    :return: None
    '''
    aug_name_list, aug_label_list = get_augmentation_info(save_path)
    total_num = len(aug_name_list)
    with open(txt_path, mode='a+') as f:
        for index in range(total_num):
            content = 'aug' + str(index + 1) + ' ' + aug_name_list[index] + ' ' + 'COVID-19' + ' ' + 'aug'
            f.writelines(content)
            f.writelines('\n')
            if index % 100 == 0:
                print(index, 'writing done.')
    print('Totally writing done.')


