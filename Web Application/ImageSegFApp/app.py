from distutils.log import debug
from fileinput import filename
from flask import request
from flask import *
import torch
import numpy as np
import os
from os.path import join
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf
import nibabel

def extract_patches_for_test(img):
    
    """
    Main
    """
    patch_i1 = img[60:180, 60:180, :64]
    patch_i2 = img[60:180, 60:180, 43:107]
    patch_i3 = img[60:180, 60:180, 91:155]
    
    patches = []
    
    patches.append(patch_i1[None, ...])
    patches.append(patch_i2[None, ...])
    patches.append(patch_i3[None, ...])
    patches = np.concatenate(patches, axis=0)
    return patches

def reconstruct_mask(preds):
    # preds.shape = [N, 4, 120, 120, 64]
    preds = np.argmax(preds, axis=1)
    mask = np.zeros([240, 240, 155], dtype=np.uint8)
    
    mask[60:180, 60:180, :64] = preds[0]
    mask[60:180, 60:180, 91:155] = preds[2]
    mask[60:180, 60:180, 43:107] = preds[1]
    
    return mask
   
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3),
                 act='selu', init=None, padding='same'):
        super(ConvLayer, self).__init__()

        self.bn = torch.nn.BatchNorm3d(in_channels)
        self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding,
                                    groups=in_channels)
        self.act = torch.nn.ReLU()
        self.mp = torch.nn.MaxPool3d((2,2,2))

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.conv(x)
        x = self.mp(x)
        return x

class TConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, size=None, kernel_size=(3,3,3), interpolation='nearest', groups=2):
        super(TConvLayer, self).__init__()

        if size is None:
            self.us = torch.nn.Upsample(scale_factor=(2,2,2), mode=interpolation)
        else:
            self.us = torch.nn.Upsample(size=size, mode=interpolation)
        self.bn = torch.nn.BatchNorm3d(in_channels)  # in_channels*2 because BatchNorm follows Upsample
        self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding='same',
                                    groups=groups)
        self.act = torch.nn.SELU()

    def forward(self, inputs):
        x = self.us(inputs)
        x = self.bn(x)
        x = self.act(self.conv(x))
        return x

class UNet(torch.nn.Module):
    def __init__(self, N_CHANNELS, N_CLASSES):
        super(UNet, self).__init__()
        self.act = torch.nn.SELU()

        self.clayer1 = ConvLayer(N_CHANNELS, 64, kernel_size=(5, 5, 5))
        self.bn1 = torch.nn.BatchNorm3d(64)
        self.conv1 = torch.nn.Conv3d(64, 64, kernel_size=(5, 5, 5), padding='same', groups=2)

        self.clayer2 = ConvLayer(64, 128, kernel_size=(5, 5, 5))
        self.bn2 = torch.nn.BatchNorm3d(128)
        self.conv2 = torch.nn.Conv3d(128, 128, kernel_size=(5, 5, 5), padding='same', groups=2)

        self.clayer3 = ConvLayer(128, 256)
        self.bn3 = torch.nn.BatchNorm3d(256)
        self.conv3 = torch.nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding='same', groups=2)

        self.clayer4 = ConvLayer(256, 256)
        self.bn4 = torch.nn.BatchNorm3d(256)
        self.conv4 = torch.nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding='same', groups=2)

        #--- Upsampling block

        self.tclayer1 = TConvLayer(256, 256, size=[15, 15, 8])
        self.tbn1 = torch.nn.BatchNorm3d(256)
        self.tconv1 = torch.nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding='same', groups=2)
        self.tln1 = torch.nn.InstanceNorm3d(256)

        self.tclayer2 = TConvLayer(256, 128)
        self.tbn2 = torch.nn.BatchNorm3d(128)
        self.tconv2 = torch.nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding='same', groups=2)
        self.tln2 = torch.nn.InstanceNorm3d(128)

        self.tclayer3 = TConvLayer(128, 64)
        self.tbn3 = torch.nn.BatchNorm3d(64)
        self.tconv3 = torch.nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding='same', groups=2)
        self.tln3 = torch.nn.InstanceNorm3d(64)

        self.tclayer4 = TConvLayer(64, N_CLASSES, kernel_size=(5, 5, 5), groups=1)
        self.tbn4 = torch.nn.BatchNorm3d(N_CLASSES)
        self.tconv4 = torch.nn.Conv3d(N_CLASSES, N_CLASSES,
                            kernel_size=(5, 5, 5), padding='same')


    def forward(self, X):
        act = torch.nn.SELU()

        """---------------------------------------------------------------------------------------
        Downsampling block
        ---------------------------------------------------------------------------------------"""
        Y1 = self.clayer1(X)  # torch.Size([1, 1, 120, 120, 64])
        Y1 = self.bn1(Y1)
        Y1 = self.conv1(Y1)  # torch.Size([1, 64, 60, 60, 32])

        Y2 = self.clayer2(self.act(Y1))  # torch.Size([1, 128, 30, 30, 16])
        Y2 = self.bn2(Y2)
        Y2 = self.conv2(Y2)

        Y3 = self.clayer3(self.act(Y2))  #  # torch.Size([1, 256, 15, 15, 8])
        Y3 = self.bn3(Y3)
        Y3 = self.conv3(Y3)

        Y4 = self.clayer4(self.act(Y3))
        Y4 = self.bn4(Y4)
        Y4 = act(self.conv4(Y4))  # torch.Size([1, 512, 7, 7, 4])

        """---------------------------------------------------------------------------------------
        Upsampling block
        ---------------------------------------------------------------------------------------"""
        Y3_R = self.tclayer1(Y4)  # 15x15x256
        Y3_R = self.tbn1(Y3_R)
        Y3_R = self.tconv1(Y3_R)
        Y3_R = self.act(Y3_R + Y3)
        Y3_R = self.tln1(Y3_R)

        Y2_R = self.tclayer2(Y3_R)  # 30x30x128
        Y2_R = self.tbn2(Y2_R)
        Y2_R = self.tconv2(Y2_R)
        Y2_R = self.act(Y2_R + Y2)
        Y2_R = self.tln2(Y2_R)

        Y1_R = self.tclayer3(Y2_R)  # 60x60x64
        Y1_R = self.tbn3(Y1_R)
        Y1_R = self.tconv3(Y1_R)
        Y1_R = self.act(Y1_R + Y1)
        Y1_R = self.tln3(Y1_R)

        Y = self.tclayer4(Y1_R)
        Y = self.tbn4(Y)
        Y = self.tconv4(Y)
        # Y = torch.nn.Softmax(dim=-1)(Y)

        return Y
    
model = UNet(N_CHANNELS=1, N_CLASSES=4)
model.load_state_dict(torch.load('final_model.pt', map_location='cpu'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload():
    if request.method == "POST":
        f = request.files['img_file']
        img_fname = f.filename
        img_bytes = f.stream.read()
        with open(join("imgs", f.filename), "wb") as f:
            f.write(img_bytes)

        nib_img = nibabel.load(join("imgs", img_fname))
        img = nib_img.get_fdata().astype('uint8')
        img = tf.image.convert_image_dtype(img, dtype=tf.float32, saturate=True).numpy()
        patches = extract_patches_for_test(img)
        patches = patches[..., None]
        preds = []
        model.eval()
        with torch.no_grad():
            for i in range(patches.shape[0]):
                patch = patches[i:i+1].copy()
                patch = np.transpose(patch, [0, -1, 1, 2, 3])
                pred = model(torch.Tensor(patch).to('cpu')).detach().cpu().numpy()
                preds.append(pred)
        pred = np.concatenate(preds, axis=0)
        pred = reconstruct_mask(pred)
        pred_fname = img_fname.split("_t1")[0] + "_seg.nii"
        pred = nibabel.Nifti1Image(pred, nib_img.affine)
        # nibabel.save(pred, os.path.join("preds", pred_fname))
        nibabel.save(pred, os.path.join(os.path.join("static", "preds"), "tmp_pred.nii"))
        return render_template("acknowledge.html", name=pred_fname)
        # return render_template("acknowledge.html", url=url_for("preds", filename=pred_fname))




