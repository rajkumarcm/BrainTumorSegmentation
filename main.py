#%%
import numpy as np
from matplotlib import pyplot as plt
import nibabel
from skimage.transform import resize
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from os.path import abspath, join
from os import listdir
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split


#%%
DATA_DIR = "/home/raj/from_host/MICCAI_BraTS2020_TrainingData/"
DATA_DIR = r"C:\Users\Rajkumar\tf_container\MICCAI_BraTS2020_TrainingData"
image_fnames = os.listdir(DATA_DIR)
IMG_SIZE = 120

#%%
tmp_fname = image_fnames[0]
img_fname = tmp_fname + os.path.sep + tmp_fname + "_t1.nii"
file1 = nibabel.load(join(DATA_DIR, img_fname))
tmp_img = file1.get_fdata()

#%%
slice = tmp_img[..., 60]  # 10 -> random channel
fig = plt.figure(figsize=(4, 4))
plt.imshow(slice, cmap="bone")
plt.axis("off")
fig.tight_layout()
plt.show()

#%%
annt_fname = tmp_fname + os.path.sep + tmp_fname + "_seg.nii"
seg1 = nibabel.load(join(DATA_DIR, annt_fname))
tmp_seg = seg1.get_fdata()

#%%
fig = plt.figure()
plt.imshow(tmp_seg[..., 60])
plt.axis("off")
fig.tight_layout()
plt.show()

#%%
seg_indices = np.where(tmp_seg > 0)
min_x, max_x = np.min(seg_indices[0]), np.max(seg_indices[0])
min_y, max_y = np.min(seg_indices[1]), np.max(seg_indices[1])
min_z, max_z = np.min(seg_indices[2]), np.max(seg_indices[2])

#%%
recons_img = np.zeros_like(tmp_img)
recons_img[tmp_seg > 0] = tmp_img[tmp_seg > 0]

#%%
fig, axes = plt.subplots(1, 2)
axes[0].imshow(tmp_img[..., 60])
axes[0].set_title("Original Image")
axes[1].imshow(recons_img[..., 60])
axes[1].set_title("Segmented Image")
axes[0].axis("off")
axes[1].axis("off")
fig.tight_layout()
plt.show()

#%%
mask = np.ma.masked_where(tmp_seg > 0, tmp_seg)

#%%
fig = plt.figure()
plt.imshow(tmp_img[..., 60], interpolation='none')
plt.imshow(mask[..., 60], cmap="gray", alpha=0.7, interpolation='none')
plt.axis("off")
fig.tight_layout()
plt.show()

#%%
# from nilearn import plotting
#
# display = plotting.plot_anat(join(DATA_DIR, img_fname))
# display.add_overlay(join(DATA_DIR, annt_fname), cmap="hsv", colorbar=True)
# plt.show()

#%%
BATCH_SIZE = 5
PREFETCH = 2
EPOCHS = 100
DATA_PCALLS = 5

#%%
image_fnames = os.listdir(DATA_DIR)
tr_fnames, ts_fnames = train_test_split(image_fnames, test_size=0.3)
tr_fnames, vl_fnames = train_test_split(tr_fnames, test_size=0.3)

def datagen(fnames):
    for i in range(len(fnames)):
        img_fname = fnames[i].decode() + os.path.sep + fnames[i].decode() + "_t1.nii"
        seg_fname = fnames[i].decode() + os.path.sep + fnames[i].decode() + "_seg.nii"
        img = nibabel.load(join(DATA_DIR, img_fname))
        img = img.get_fdata().astype(np.float32)
        img = resize(img, [IMG_SIZE, IMG_SIZE, 115])

        seg = nibabel.load(join(DATA_DIR, seg_fname))
        seg = seg.get_fdata().astype(np.int64)
        seg = resize(seg, [IMG_SIZE, IMG_SIZE, 115])

        yield img, seg

# for tmp_x, tmp_y in datagen(tr_fnames):
#     print(tmp_x.shape)
#     print(tmp_y.shape)

tr_dset = tf.data.Dataset.from_generator(datagen, output_types=((tf.float32, tf.int64)),
                                         output_shapes=((IMG_SIZE, IMG_SIZE, 115),
                                                        (IMG_SIZE, IMG_SIZE, 115)),
                                         args=(tr_fnames,))
tr_dset = tr_dset.batch(BATCH_SIZE, num_parallel_calls=DATA_PCALLS).prefetch(PREFETCH)

vl_dset = tf.data.Dataset.from_generator(datagen, output_types=((tf.float32, tf.int64)),
                                         output_shapes=((IMG_SIZE, IMG_SIZE, 115),
                                                        (IMG_SIZE, IMG_SIZE, 115)),
                                         args=(vl_fnames,))
vl_dset = vl_dset.batch(BATCH_SIZE, num_parallel_calls=DATA_PCALLS).prefetch(PREFETCH)

ts_dset = tf.data.Dataset.from_generator(datagen, output_types=((tf.float32, tf.int64)),
                                         output_shapes=((IMG_SIZE, IMG_SIZE, 115),
                                                        (IMG_SIZE, IMG_SIZE, 115)),
                                         args=(ts_fnames,))
ts_dset = ts_dset.batch(BATCH_SIZE, num_parallel_calls=DATA_PCALLS).prefetch(PREFETCH)

#%%
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
        self.us = UpSampling3D(size=(2,2,1))

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

model = Sequential([InputLayer(input_shape=[IMG_SIZE, IMG_SIZE, 115, 1],
                               batch_size=None),
                    Downsampling_block(),
                    Upsampling_block(),
                    Conv3D(1, (7,7,1), padding='valid'),
                    Conv3D(1, (7,7,1), padding='valid'),
                    Conv3D(1, (5,5,1), padding='valid'),
                    Reshape([IMG_SIZE, IMG_SIZE, 115])
                    ])

model.summary()


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

model.compile(optimizer="sgd", loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(tr_dset, validation_data=vl_dset, epochs=EPOCHS, verbose=1,
          steps_per_epoch=len(tr_fnames)//BATCH_SIZE,
          validation_steps=len(vl_fnames)//BATCH_SIZE,
          workers=10, use_multiprocessing=True,
          callbacks=[lr_scheduler, t_nan, es_cb, checkpoint_cb])
























