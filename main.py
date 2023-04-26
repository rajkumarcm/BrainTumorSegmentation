#%%
import numpy as np
from matplotlib import pyplot as plt
import nibabel
from skimage.transform import resize
import os
import time
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from os.path import abspath, join
from os import listdir
import tensorflow as tf
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3,3,1), init=None, padding='same'):
        super(ConvLayer, self).__init__()
        if init is None:
            init = tf.keras.initializers.GlorotNormal()
        # self.conv = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation='selu',
        #                                    kernel_initializer=init, bias_initializer=init)
        self.conv = tf.keras.layers.Conv3D(filters, kernel_size, padding=padding, activation='selu',
                                           kernel_initializer=init, bias_initializer=init)
        self.bn = tf.keras.layers.BatchNormalization()
        self.mp = tf.keras.layers.MaxPool3D((2,2,1))

    def call(self, inputs):
        x = self.bn(inputs)
        x = self.conv(x)
        x = self.mp(x)
        return x

class Downsampling_block(tf.keras.layers.Layer):
    def __init__(self):
        super(Downsampling_block, self).__init__()
        # 120x120x1024 -> 60x60x64
        self.conv1 = ConvLayer(64)
        # 60x60x64 -> 30x30x128
        self.conv2 = ConvLayer(128)
        # 30x30x128 -> 15x15x256
        self.conv3 = ConvLayer(256)
        # 15x15x256 -> 7x7x512
        self.conv4 = ConvLayer(512)
        # 7x7x512 -> 1x1x1024
        # self.conv5 = tf.keras.layers.Conv3D(1024, (7,7,1), padding='valid', activation='selu')
        # self.mp5 = tf.keras.layers.MaxPool3D((2,2,1))
        self.conv5 = ConvLayer(1024)
        self.conv6 = ConvLayer(1024)

    def __call__(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


class TConvLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(5,5,1), init=None):
        super(TConvLayer, self).__init__()
        if init is None:
            init = tf.keras.initializers.GlorotNormal()

        self.tconv = tf.keras.layers.Conv3DTranspose(filters, kernel_size, padding='valid',
                                                      activation='selu', kernel_initializer=init,
                                                      bias_initializer=init)
        self.us = UpSampling3D(size=(2,2,1), interpolation='bilinear')

    def __call__(self, inputs):
        x = self.us(self.tconv(inputs))
        return x


class Upsampling_block(tf.keras.layers.Layer):
    def __init__(self):
        super(Upsampling_block, self).__init__()

        # 1x1x1024 -> 7x7x1024
        self.tconv1 = TConvLayer(1024)

        # 7x7x1024 -> 15x15x512
        self.tconv2 = TConvLayer(512)

        # 15x15x512 -> 30x30x256
        self.tconv3 = TConvLayer(256)

        # 30x30x128 -> 60x60x64
        self.tconv4 = TConvLayer(1)

        # 60x60x64 -> 120x120x1024
        # self.tconv5 = TConvLayer(115,)

        # 60x60x64 -> 120x120x1024
        # self.tconv6 = TConvLayer(115)


    def __call__(self, inputs):
        x = self.tconv1(inputs)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        # x = self.tconv5(x)
        return x


class ImageSegmentation:
    def __init__(self):
        self.DATA_DIR = "/home/raj/from_host/MICCAI_BraTS2020_TrainingData/"
        self.DATA_DIR = r"C:\Users\Rajkumar\tf_container\MICCAI_BraTS2020_TrainingData"
        self.IMG_SIZE = 120

        self.BATCH_SIZE = 5
        self.PREFETCH = 2
        self.EPOCHS = 100
        self.DATA_PCALLS = 5

        image_fnames = os.listdir(self.DATA_DIR)

        tr_fnames, ts_fnames = train_test_split(image_fnames, test_size=0.3)
        tr_fnames, vl_fnames = train_test_split(tr_fnames, test_size=0.3)

        self.tr_steps = len(tr_fnames) // self.BATCH_SIZE
        self.vl_steps = len(vl_fnames) // self.BATCH_SIZE

        tr_dset = tf.data.Dataset.from_generator(self.datagen, output_types=((tf.float32, tf.int64)),
                                                 output_shapes=((self.IMG_SIZE, self.IMG_SIZE, 115),
                                                                (self.IMG_SIZE, self.IMG_SIZE, 115)),
                                                 args=(tr_fnames,))
        self.tr_dset = tr_dset.batch(self.BATCH_SIZE, num_parallel_calls=self.DATA_PCALLS)\
                         .prefetch(self.PREFETCH)

        vl_dset = tf.data.Dataset.from_generator(self.datagen, output_types=((tf.float32, tf.int64)),
                                                 output_shapes=((self.IMG_SIZE, self.IMG_SIZE, 115),
                                                                (self.IMG_SIZE, self.IMG_SIZE, 115)),
                                                 args=(vl_fnames,))
        self.vl_dset = vl_dset.batch(self.BATCH_SIZE, num_parallel_calls=self.DATA_PCALLS)\
                         .prefetch(self.PREFETCH)

        ts_dset = tf.data.Dataset.from_generator(self.datagen, output_types=((tf.float32, tf.int64)),
                                                 output_shapes=((self.IMG_SIZE, self.IMG_SIZE, 115),
                                                                (self.IMG_SIZE, self.IMG_SIZE, 115)),
                                                 args=(ts_fnames,))
        self.ts_dset = ts_dset.batch(self.BATCH_SIZE, num_parallel_calls=self.DATA_PCALLS)\
                         .prefetch(self.PREFETCH)


    def datagen(self, fnames):
        for i in range(len(fnames)):
            img_fname = fnames[i].decode() + os.path.sep + fnames[i].decode() + "_t1.nii"
            seg_fname = fnames[i].decode() + os.path.sep + fnames[i].decode() + "_seg.nii"
            img = nibabel.load(join(self.DATA_DIR, img_fname))
            img = img.get_fdata().astype(np.float32)
            img = resize(img, [self.IMG_SIZE, self.IMG_SIZE, 115])

            seg = nibabel.load(join(self.DATA_DIR, seg_fname))
            seg = seg.get_fdata().astype(np.int64)
            seg = resize(seg, [self.IMG_SIZE, self.IMG_SIZE, 115])

            yield img, seg


    def build_model(self):

        model = Sequential([InputLayer(input_shape=[self.IMG_SIZE, self.IMG_SIZE, 115, 1],
                                       batch_size=None),
                            Downsampling_block(),
                            Upsampling_block(),
                            Conv3D(1, (7,7,1), padding='valid'),
                            Conv3D(1, (7,7,1), padding='valid'),
                            Conv3D(1, (5,5,1), padding='valid'),
                            Reshape([self.IMG_SIZE, self.IMG_SIZE, 115])
                          ])

        model.summary()
        model.compile(optimizer="sgd", loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])
        return model

    def train(self, model):
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                                            patience=3, verbose=1,
                                                            mode='min', min_delta=1e-6, min_lr=0)

        t_nan = tf.keras.callbacks.TerminateOnNaN()

        es_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3,
                                                 patience=9, verbose=1, mode='auto',
                                                 baseline=None, restore_best_weights=False,
                                                 start_from_epoch=0)

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                                    'checkpoints', monitor='val_loss', verbose=1,
                                    save_best_only=True, save_weights_only=False,
                                    mode='auto', save_freq='epoch')

        model.fit(self.tr_dset, validation_data=self.vl_dset, epochs=self.EPOCHS, verbose=1,
                  steps_per_epoch=self.tr_steps,
                  validation_steps=self.vl_steps,
                  workers=10, use_multiprocessing=True,
                  callbacks=[lr_scheduler, t_nan, es_cb, checkpoint_cb])

    def test_dataset(self):
        for X, Y in self.tr_dset:
            start_time = time.time()
            print(X.shape)
            print(Y.shape)
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"Time elapsed: {elapsed}")

if __name__ == "__main__":
    img_seg = ImageSegmentation()
    model = img_seg.build_model()
    # img_seg.train(model)
    img_seg.test_dataset()





















