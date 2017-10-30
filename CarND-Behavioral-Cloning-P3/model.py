import numpy as np
import pandas as pd
import os
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Convolution2D, MaxPooling2D, Cropping2D, Activation, Dropout
from keras import optimizers

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Next two lines are necessary to save figures on EC2 instance
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_FOLDER = './data/'
IMG_FOLDER = os.path.join(DATA_FOLDER, 'IMG')
DATA_FILE = os.path.join(DATA_FOLDER, 'driving_log.csv')


class Generator:
    """
    Class to make the generators that are passed to Keras.
    """
    def __init__(self, file_paths, measurements, batch_size=32):
        """
        Class creator. In addition to the list of files and the array of measurements, it stores the number of samples
        so that ti can be passed to keras.
        :param filepaths: list of image files
        :param measurements: Corresponding steering angle
        """
        self.file_paths = file_paths
        self.measurements = measurements
        self.num_samples_before_flipping = len(file_paths)
        # Note the factor2, this is due to fact that the image flipping is done by the generator and therefore will
        # double the size of the batch.
        self.num_samples_after_flipping = 2 * self.num_samples_before_flipping
        self.num_samples = self.num_samples_after_flipping
        self.batch_size = batch_size

    def next(self):
        """
        Function that actually implements the generator. Additionnaly to feeding the right amount of data every time it
        is called, it converts the paths that are contained in the file path list ot a local path (training data have
        been stored in three different locations initially), read the corresponding images in RGB format, and
        furthermore, it adds a flipped version of each image to the batch.
        """
        while 1:  # Loop forever so the generator never terminates
            # shuffle data
            self.measurements, self.file_paths = shuffle(self.measurements, self.file_paths)
            for offset in range(0, self.num_samples_before_flipping, self.batch_size):
                end = min(offset + self.batch_size, self.num_samples_before_flipping)
                batch_measurements = self.measurements[offset:end]
                batch_file_paths = self.file_paths[offset:end]
                batch_images = read_images(batch_file_paths)
                batch_images, batch_measurements = flipped_images(batch_images, batch_measurements)
                X = np.array(batch_images)
                y = np.array(batch_measurements)
                yield X, y


def save_flipped_image():
    """
    His function is only used to make visualization of the image flipping operation. One image carefully is specified
    here.
    """
    fname = os.path.join(IMG_FOLDER, 'center_2017_10_26_11_03_01_702.jpg')
    img = cv2.imread(fname)
    # BGR to RGB conversion
    plt.imshow(img[:, :, ::-1])
    plt.xticks([]), plt.yticks([])
    plt.savefig('./figures/original.png')
    flipped = np.fliplr(img)
    plt.imshow(flipped[:, :, ::-1])
    plt.xticks([]), plt.yticks([])
    plt.savefig('./figures/flipped.png')
    return


def read_images(filepath_list):
    """
    function that takes a list of image paths, convert it to a local path and read the corresponding images as anumpy
    array.
    :param filepath_list: list of paths
    :return: numpy array containing all images
    """
    res = []
    for filepath in filepath_list:
        name = filepath.split('/')[-1]
        newname = os.path.join(IMG_FOLDER, name)
        img = cv2.imread(newname)
        # Note : Open cv reads the images in BGR format. The conversion from BGR to RGB is done below thanks to the ::-1
        # index operation.
        res.append(img[:, :, ::-1])

    return np.array(res)


def flipped_images(imgs, meas):
    """
    Function that take a batch of images and the corresponding steering value, compute a flipped version of the image
    (symmetry operation w.r.t. a vertical axis) and of the corresponding steering angle an return a augmented version
    of the arrays.
    :param imgs: numpy.ndarray. Array of images
    :param meas: numpy,array. Array of measures
    :return: new_imgs, new_meas which are the initial arrays augmented with flipped data.
    """
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
def read_raw_data(correction=0.15):
    """
    Read data from the driving_log.csv file and produce the numpy array of the steering angle measurements and a list
    of the corresponding images. The images from the three cameras are used. For the left and right camera, a correction
    angle is added (or substracted) to the center steering angle.
    :param correction: orrection applied positively for the images from the left and negatvely from the right.
    :return: a list of file paths and the corresponding steering angle measurements.
    """
    datafile = DATA_FOLDER + 'driving_log.csv'
    # read csv as a pandas dataframe
    df = pd.read_csv(datafile, header=None,
                     names=['file_center', 'file_left', 'file_right', 'steering', 'throttle', 'brake', 'speed'])
    # shuffle data
    df = df.sample(frac=1.)
    # Correct steering measurements for left and right cameras. Note that the angles are negative when turning left,
    # hence the correction has to be added (i.e. reduced steering) for the left image, while it has to be substracted
    # (i.e. increased steering) for the right image.
    df['steering_left'] = df['steering'] + correction
    df['steering_right'] = df['steering'] - correction

    # Extract steering angle as a numpy array
    measurements = np.concatenate((df['steering'].as_matrix(), df['steering_left'].as_matrix(),
                                   df['steering_right'].as_matrix()))

    # Extract images file list (note that the addition operation on list effectively concatenate)
    filepath_list = df['file_center'].tolist() + df['file_left'].tolist() + df['file_right'].tolist()

    return filepath_list, measurements


def make_generators(filepaths, measurements):
    """
    Function that creates two objects of the class Generator : one for the validation and one for the training.
    :param filepaths: path to files
    :param measurements: array of measures
    :return: two Generator object
    """
    # split in training and validation
    filepaths_train, filepaths_valid, measurements_train, measurements_valid = \
        train_test_split(filepaths, measurements, test_size=0.2)
    # create generators
    train_generator = Generator(filepaths_train, measurements_train)
    valid_generator = Generator(filepaths_valid, measurements_valid)

    return train_generator, valid_generator


def build_model():
    """
    Definition of the Deep Learning model. The following architecture is implemented :
    - Normalizaiton layer
    - Image cropping layer
    - Convolution layer with 16 filters of size (5,5)
    - Max Pooling with a filter of size (5,5)
    - Dropout with a dropout rate of 10%
    - ReLu activation
    - Convolution layer with 32 filters of size (5,5)
    - Max Pooling with a filter of size (5,5)
    - Dropout with a dropout rate of 10%
    - ReLu activation
    - Dense layer with 128 hidden units
    - Dropout with a dropout rate of 20%
    - ReLu activation
    - Linear layer to compute the output.
    The loss function used is Mean Square Error. The optimizer is adam with default parameters.
    :return: the keras model
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.) - 0.5, input_shape=(160, 320, 3)))
    # cropping argument is defined as follows : ((top_crop, bottom_crop), (left_crop, right_crop))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Convolution2D(16, 5, 5))
    model.add(MaxPooling2D((5, 5)))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 5, 5))
    model.add(MaxPooling2D((5, 5)))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    return model


def train_model(model, train_generator, valid_generator, plot=False, **kwargs):
    """
    Perform the training of the Keras model.
    :param model: Kera model
    :param train_generator: Generator Object providing batches of training data
    :param valid_generator: Generator Object providing batches of validations data
    :param plot: Boolean param indicating if the plt of training/val errors with # of epochs need to be saved.
    :param kwargs: Additional parameter to pass to the fit function
    :return:
    """
    history_object = model.fit_generator(train_generator.next(), samples_per_epoch=train_generator.num_samples,
                                         validation_data=valid_generator.next(),
                                         nb_val_samples=valid_generator.num_samples, **kwargs)
    # save model after training.
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

# Main program.
if __name__ == '__main__':
    # Read raw data
    filepath_list, measurements = read_raw_data()
    # Create data generators
    train_generator, valid_generator = make_generators(filepath_list, measurements)
    # Build Keras model
    model = build_model()
    # train the mdoel
    train_model(model, train_generator, valid_generator, plot=True, nb_epoch=5)
