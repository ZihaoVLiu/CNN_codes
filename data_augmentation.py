from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os


def data_augmentation(number, dict_covid, image_path, save_path):
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
    aug_name_list = []
    name_list = os.listdir(save_path)
    for name in name_list:
        if name[:4] == 'aug_':
            aug_name_list.append(name)
    aug_label_list = [2] * len(aug_name_list)
    return aug_name_list, aug_label_list
