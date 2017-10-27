import numpy as np
import pandas as pd
import os
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Convolution2D, MaxPooling2D, Cropping2D, Activation, Dropout

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

import timeit

DATA_FOLDER = './data/'
IMG_FOLDER = os.path.join(DATA_FOLDER, 'IMG')
DATA_FILE = os.path.join(DATA_FOLDER, 'driving_log.csv')


class Generator:

    def __init__(self, file_paths, measurements, batch_size = 32):
        """

        :param filepaths: list of image files
        :param measurements: Corresponding steering angle
        """
        self.file_paths = file_paths
        self.measurements = measurements
        self.num_samples_before_flipping = len(file_paths)
        self.num_samples_after_flipping = 2*self.num_samples_before_flipping
        self.num_samples = self.num_samples_after_flipping
        self.batch_size = batch_size

    def next(self):
        while 1:  # Loop forever so the generator never terminates
            # shuffle data
            self.measurements, self.file_paths = shuffle(self.measurements, self.file_paths)
            for offset in range(0, self.num_samples_before_flipping, self.batch_size):
                end=min(offset + self.batch_size, self.num_samples_before_flipping)
                batch_measurements = self.measurements[offset:end]
                batch_file_paths = self.file_paths[offset:end]
                batch_images = convert_images_path(batch_file_paths)
                batch_images, batch_measurements = flipped_images(batch_images, batch_measurements)
                X = np.array(batch_images)
                y = np.array(batch_measurements)
                yield X, y


def save_flipped_image():
    fname = os.path.join(IMG_FOLDER, 'center_2017_10_26_11_03_01_702.jpg')
    img = cv2.imread(fname)

    plt.imshow(img[:, :, ::-1])
    plt.xticks([]), plt.yticks([])
    plt.savefig('./figures/original.png')
    flipped = np.fliplr(img)
    plt.imshow(flipped[:, :, ::-1])
    plt.xticks([]), plt.yticks([])
    plt.savefig('./figures/flipped.png')
    return


def convert_images_path(filepath_list):
    res = []
    for filepath in filepath_list:
        name = filepath.split('/')[-1]
        newname = os.path.join(IMG_FOLDER, name)
        img = cv2.imread(newname)
        res.append(img)

    return np.array(res)


def flipped_images(imgs, meas):
    shape = imgs.shape
    n = shape[0]  # initial number of images
    new_shape = (2 * n, shape[1], shape[2], shape[3])
    new_imgs = np.ndarray(shape=new_shape)
    # copy images on the first half
    new_imgs[:n, :, :, :] = imgs[:, :, :, :]
    for i in range(n):
        new_imgs[i + n, :, :, :] = np.fliplr(imgs[i, :, :, :])

    new_meas = np.concatenate((meas, -meas))

    return new_imgs, new_meas


# Read csv file
def read_raw_data(correction=0.2):
    datafile = DATA_FOLDER + 'driving_log.csv'
    # read csv as a pandas dataframe
    df = pd.read_csv(datafile, header=None,
                     names=['file_center', 'file_left', 'file_right', 'steering', 'throttle', 'brake', 'speed'])
    # Correct steering measurements for left and right cameras
    df['steering_left'] = df['steering'] + correction
    df['steering_right'] = df['steering'] - correction

    # Extract steering angle as a numpy array
    measurements = np.concatenate((df['steering'].as_matrix(), df['steering_left'].as_matrix(),
                                   df['steering_right'].as_matrix()))

    # Extract images file list (note addition operation on list effectively concatenate)
    filepath_list = df['file_center'].tolist() + df['file_left'].tolist() + df['file_right'].tolist()

    return filepath_list, measurements


def make_generators(filepaths, measurements):
    # split in training and validation
    filepaths_train, filepaths_valid, measurements_train, measurements_valid = \
        train_test_split(filepaths, measurements, test_size=0.2)
    # create generators
    train_generator = Generator(filepaths_train, measurements_train)
    valid_generator = Generator(filepaths_valid, measurements_valid)

    return train_generator, valid_generator


def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.) - 0.5, input_shape=(160, 320, 3)))
    # cropping argument is defined as follows : ((top_crop, bottom_crop), (left_crop, right_crop))
    model.add(Cropping2D(cropping=((55, 25), (0, 0))))
    model.add(Convolution2D(16, 5, 5))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D((5, 5)))
    model.add(Convolution2D(32, 3, 3))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D((3, 3)))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    return model


def train_model(model, train_generator, valid_generator, plot=False):

    history_object = model.fit_generator(train_generator.next(), samples_per_epoch=train_generator.num_samples,
                                         validation_data = valid_generator.next(),
                                         nb_val_samples = valid_generator.num_samples, nb_epoch = 4)


    model.save('model.h5')

    if plot:
        ### plot the training and validation loss for each epoch
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.savefig('training_error.png')

    return


# Build data set
if __name__ == '__main__':
    filepath_list, measurements = read_raw_data()
    train_generator, valid_generator = make_generators(filepath_list, measurements)
    model = build_model()
    train_model(model, train_generator, valid_generator, plot=True)


