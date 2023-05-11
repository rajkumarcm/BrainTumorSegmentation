#%%
import numpy as np
from matplotlib import pyplot as plt
import nibabel
from nibabel.processing import conform
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_FROCE_GPU_ALLOW_GROWTH']='true'
import tensorflow as tf
import time
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from os.path import abspath, join
from os import listdir
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import jaccard_score, f1_score
np.random.seed(123)

class BraTDataset(Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''
    def __init__(self, df, list_IDs, data_dir, img_size, depth_size=60, n_classes=4):
        self.list_IDs = list_IDs
        self.df = df.copy()
        self.DATA_DIR = data_dir
        self.IMG_SIZE = img_size
        self.DEPTH_SIZE = depth_size
        self.N_CLASSES = n_classes

    def __len__(self):
        #Denotes the total number of samples'
        return len(self.list_IDs)

    def onehot_image(self, seg):
        seg_manual = np.zeros([seg.shape[0], seg.shape[1], seg.shape[2], self.N_CLASSES],
                              dtype=np.int64)
        seg_manual[seg == 0, 0] = 1
        seg_manual[seg == 1, 1] = 1
        seg_manual[seg == 2, 2] = 1
        seg_manual[seg == 4, 3] = 1
        return seg_manual

    def load_data(self, img_fname, seg_fname):
        img = nibabel.load(join(self.DATA_DIR, img_fname))
        # img = conform(img, [self.IMG_SIZE, self.IMG_SIZE, self.DEPTH_SIZE])
        img = img.get_fdata().astype('uint8')
        img = tf.image.convert_image_dtype(img, dtype=tf.float32, saturate=True).numpy()
        img = img[..., None]

        seg = nibabel.load(join(self.DATA_DIR, seg_fname))
        # seg = conform(seg, [self.IMG_SIZE, self.IMG_SIZE, self.DEPTH_SIZE])
        seg = seg.get_fdata().astype(np.int64)
        seg = self.onehot_image(seg)  # IMG_SIZE x IMG_SIZE x DEPTH x N_CLASSES
        # seg = np.transpose(seg, [2, 0, 1, 3])
        return img, seg.astype(np.float32)

    def extract_patch(self, img, seg):
        indices = np.where(seg > 0)
        med_x = int(np.median(indices[0]))
        med_y = int(np.median(indices[1]))
        med_z = int(np.median(indices[2]))

        if med_x + 60 >= img.shape[0]:
            s_x = img.shape[0] - 120
        else:
            s_x = med_x - 60
        e_x = s_x + 120

        if med_y + 60 >= img.shape[1]:
            s_y = img.shape[1] - 120
        else:
            s_y = med_y - 60
        e_y = s_y + 120

        if med_z + 30 >= img.shape[2]:
            s_z = img.shape[2] - 64
        else:
            s_z = med_z - 32
        e_z = s_z + 64

        img_patch = img[s_x:e_x, s_y:e_y, s_z:e_z]
        seg_patch = seg[s_x:e_x, s_y:e_y, s_z:e_z]

        return img_patch, seg_patch

    def __getitem__(self, index):
        #Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        img_fpath = self.df.img_path.get(ID)
        seg_fpath = self.df.seg_path.get(ID)

        # Augmentation only for train
        # X = self.transforms(img)
        img, seg = self.load_data(img_fpath, seg_fpath)
        img, seg = self.extract_patch(img, seg)
        img = np.transpose(img, [3, 0, 1, 2])
        seg = np.transpose(seg, [3, 0, 1, 2])
        return img, seg

def populate_files_df(DATA_DIR):
    fnames = listdir(DATA_DIR)
    img_paths = []
    seg_paths = []
    for fname in fnames:
        file_prefix = fname + os.path.sep + fname
        img_paths.append(file_prefix + "_t1.nii")
        seg_paths.append(file_prefix + "_seg.nii")
    df = pd.DataFrame({"img_path":img_paths, "seg_path":seg_paths})
    return df

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

class ImageSegmentation:
    def __init__(self):
        self.DATA_DIR = r"C:\Users\Rajkumar\tf_container\MICAAI_BraTS_Collated"
        self.IMG_SIZE = 120
        self.DEPTH_SIZE = 64

        self.BATCH_SIZE = 10
        self.DATA_PCALLS = self.N_WORKERS = 20
        self.PREFETCH = 40
        self.EPOCHS = 20
        self.N_CLASSES = 4
        self.RANDOM_STATE = 123
        self.N_CHANNELS = N_MODES = 1
        self.LR = 1e-3
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        df = populate_files_df(self.DATA_DIR)
        indices = list(range(df.shape[0]))
        np.random.shuffle(indices)
        df = df.iloc[indices]

        """
        Train, Test split
        """
        indices = set(indices)
        tr_indices = set(np.random.choice(list(indices), size=int(0.7*len(indices)),
                                          replace=False))
        ts_indices = indices.difference(tr_indices)

        tr_df = df.iloc[list(tr_indices)]
        ts_df = df.iloc[list(ts_indices)]

        tr_dset = BraTDataset(tr_df, tr_df.index, data_dir=self.DATA_DIR,
                              img_size=self.IMG_SIZE, depth_size=self.DEPTH_SIZE, n_classes=self.N_CLASSES)

        self.tr_dloader = DataLoader(tr_dset, batch_size=self.BATCH_SIZE,
                                     shuffle=True, num_workers=self.N_WORKERS,
                                     prefetch_factor=self.PREFETCH)
        self.tr_size = tr_df.shape[0]
        ts_dset = BraTDataset(ts_df, ts_df.index, data_dir=self.DATA_DIR,
                              img_size=self.IMG_SIZE, depth_size=self.DEPTH_SIZE, n_classes=self.N_CLASSES)

        self.ts_dloader = DataLoader(ts_dset, batch_size=self.BATCH_SIZE,
                                     shuffle=False, num_workers=self.N_WORKERS,
                                     prefetch_factor=self.PREFETCH)
        self.ts_size = ts_df.shape[0]


    def onehot_image(self, seg):
        seg_manual = np.zeros([self.IMG_SIZE, self.IMG_SIZE, self.DEPTH_SIZE, self.N_CLASSES],
                              dtype=np.uint8)
        seg_manual[seg == 0, 0] = 1
        seg_manual[seg == 1, 1] = 1
        seg_manual[seg == 2, 2] = 1
        seg_manual[seg == 4, 3] = 1
        return seg_manual


    def build_model(self):
        model = UNet(N_CHANNELS=self.N_CHANNELS, N_CLASSES=self.N_CLASSES)
        model.to(self.device)
        return model

    def train(self, model):
        model.train()
        model.state_dict()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True,
                                         threshold=1e-3, min_lr=1e-7)
        train_loss_item = list([])
        # test_loss_item = list([])

        for epoch in range(self.EPOCHS):
            train_loss, steps_train = 0, 0
            jac_score = best_jac = 0
            model.train()
            with tqdm(total=self.tr_size, desc="Epoch {}".format(epoch)) as pbar:
                for img, seg in self.tr_dloader:

                    img, seg = img.to(self.device), seg.to(self.device)
                    optimizer.zero_grad()
                    output = model(img)
                    loss = criterion(output, seg)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    steps_train += 1

                    tmp_seg = torch.argmax(seg, dim=1).detach().cpu().numpy().ravel()
                    tmp_output = torch.argmax(torch.nn.Softmax(dim=1)(output), dim=1).detach().cpu().numpy().ravel()
                    jscore = jaccard_score(y_pred=tmp_output, y_true=tmp_seg, average='macro')
                    jac_score += jscore
                    train_loss_item.append([epoch, loss.item()])
                    pbar.update(self.BATCH_SIZE)
                    pbar.set_postfix_str(f"Train Loss: {np.round(train_loss/steps_train, 5)}, "
                                         f"Jaccard score: {np.round(jac_score/steps_train, 5)}")

            lr_scheduler.step((jac_score / steps_train))
            if jac_score/steps_train > best_jac:
                best_jac = jac_score/steps_train
                print(f'Jaccard score was better than previous: saving the model {epoch}')
                torch.save(model.state_dict(), "final_model.pt")


    def test(self, model):
        preds = []
        segs = []
        imgs = []
        model.zero_grad()
        model.eval()
        with torch.no_grad():
            for i, (img, seg) in zip(range(50), self.ts_dloader):
                imgs.append(img.detach().cpu().numpy())
                segs.append(seg.detach().cpu().numpy())
                pred = model(img.to(self.device))
                preds.append(pred.detach().cpu().numpy())
        imgs = np.concatenate(imgs, axis=0)
        preds = np.concatenate(preds, axis=0)
        segs = np.concatenate(segs, axis=0)
        np.save("imgs_arr.npy", imgs)
        np.save("preds.npy", preds)
        np.save("segs_arr.npy", segs)
        print("files saved ...")

if __name__ == "__main__":
    training = True
    img_seg = ImageSegmentation()
    model = img_seg.build_model()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if "final_model.pt" in os.listdir(abspath("")):
        model.load_state_dict(torch.load('final_model.pt', map_location=device))
    if training:
        img_seg.train(model)
    else:
        img_seg.test(model)
    # model = tf.keras.models.load_model("checkpoints")

    # img_seg.test_dataset()






















