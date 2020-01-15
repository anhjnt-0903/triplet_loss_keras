import keras
from keras.models import Model
from keras.layers import Layer
from keras.layers import Conv2D, RepeatVector, Lambda, Add, \
                            Flatten, UpSampling2D, MaxPooling2D, \
                            Dropout, Cropping2D, Input, concatenate, \
                            BatchNormalization, Activation, \
                            AveragePooling2D, GlobalAveragePooling2D, \
                            GlobalAveragePooling1D, ZeroPadding2D, Dense, Reshape
from keras.utils import plot_model

from model import CustomModel


if __name__ == "__main__":
    weight_path = "/content/gdrive/My Drive/Colab Notebooks/Metric Learning/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    model = CustomModel(weight_path)
    
    plot_model(model, to_file='/content/gdrive/My Drive/Colab Notebooks/Metric Learning/resnet50/model.png')

    model.summary()

    