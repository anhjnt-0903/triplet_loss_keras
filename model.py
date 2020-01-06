from keras.datasets import mnist
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, concatenate,  Dropout, \
                        Conv2D, Activation, BatchNormalization, Dense, MaxPooling2D, \
                        Reshape
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
import matplotlib.pyplot as plt, numpy as np


from triplet_loss import batch_all_triplet_loss


def create_base_network(image_input_shape, embedding_size):
    """
    Base network to be shared (eq. to feature extraction).
    """
    input_image = Input(shape=image_input_shape)

    x = Flatten()(input_image)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(embedding_size)(x)

    base_network = Model(inputs=input_image, outputs=x)
    plot_model(base_network, to_file='base_network.png', show_shapes=True, show_layer_names=True)
    return base_network


if __name__ == "__main__":

    batch_size = 32
    epochs = 25
    train_flag = True  # either     True or False

    embedding_size = 64

    no_of_components = 2  # for visualization -> PCA.fit_transform()

    step = 10

    # The data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.
    input_image_shape = (28, 28, 1)
    x_val = x_test[:2000, :, :]
    y_val = y_test[:2000]

    base_network = create_base_network(input_image_shape, embedding_size)
    input_images = Input(shape=input_image_shape, name='input_image')
    embeddings = base_network(input_images)
    model = Model(inputs=input_images,
                        outputs=embeddings)

    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    opt = Adam(lr=0.0001)  # choose optimiser. RMS is good too!

    model.compile(loss=batch_all_triplet_loss, optimizer=opt)

    filepath = '/content/gdrive/My Drive/Colab Notebooks/Triplet_loss/weights' + "/triplet_loss_" + '.{epoch:02d}-{loss:.2f}.hdf5'

    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,
                                save_weights_only=True, save_best_only=True, mode='auto', period=1)

    tensor_board = TensorBoard(log_dir='/content/gdrive/My Drive/Colab Notebooks/Triplet_loss/logs')

    x_train = x_train.reshape((60000, 28, 28, 1))
    x_val = x_val.reshape((2000, 28, 28, 1))

    with tf.device('/device:GPU:0'):
        H = model.fit(
                    x=x_train,
                    y=y_train,
                    batch_size=5,
                    epochs=100,
                    validation_data=(x_val, y_val),
                    callbacks=[checkpoint, tensor_board])
