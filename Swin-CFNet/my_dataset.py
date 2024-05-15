import os
import torch
import torch.utils.data as data
from PIL import Image
from osgeo import gdal
import numpy as np
import cv2
from torchvision import transforms as T
import random
import re




#  随机数据增强
#  image 图像
#  label 标签
def truncated_linear_stretch(image, truncated_value, max_out=255, min_out=0):
    def gray_process(gray):
        truncated_down = np.percentile(gray, truncated_value)
        truncated_up = np.percentile(gray, 100 - truncated_value)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out
        gray = np.clip(gray, min_out, max_out)
        gray = np.uint8(gray)
        return gray

    image_stretch = []
    for i in range(image.shape[2]):
        # 只拉伸RGB
        if (i < 3):
            gray = gray_process(image[:, :, i])
        else:
            gray = image[:, :, i]
        image_stretch.append(gray)
    image_stretch = np.array(image_stretch)
    image_stretch = image_stretch.swapaxes(1, 0).swapaxes(1, 2)
    return image_stretch


#  随机数据增强
#  image 图像
#  label 标签
def DataAugmentation(image, label, mode):
    if (mode == "train"):
        hor = random.choice([True, False])
        if (hor):
            #  图像水平翻转
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=1)
        ver = random.choice([True, False])
        if (ver):
            #  图像垂直翻转
            image = np.flip(image, axis=0)
            label = np.flip(label, axis=0)
        stretch = random.choice([True, False])
        if (stretch):
            image = truncated_linear_stretch(image, 0.5)
    if (mode == "val"):
        stretch = random.choice([0.8, 1, 2])
        # if(stretch == 'yes'):
        # 0.5%线性拉伸
        image = truncated_linear_stretch(image, stretch)
    return image, label
def imgread(fileName):
    # print(fileName)
    dataset = gdal.Open(fileName)
    # print(fileName)
    width = dataset.RasterXSize
    # print(fileName)
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, width, height)
    # 如果是image的话,因为label是单通道
        # (C,H,W)->(H,W,C)
    if(len(data.shape)==3):
        # data[data==-3.4028e+38]=0
        data = data.swapaxes(1, 0).swapaxes(1, 2)
    return data
class My_dataset(data.Dataset):
    def __init__(self, voc_root, set="GID5",  mode: str = "train"):
        super(My_dataset, self).__init__()
        assert set in ["GID5", "GID15","Potsdam","Potsdam_without_background"], "year must be in ['2007', '2012']"
        root = os.path.join(voc_root, set)
        print(root)
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        root=os.path.join(root,mode)
        image_dir = os.path.join(root, 'images')
        mask_dir = os.path.join(root, 'labels')
        file_names = os.listdir(image_dir)
        self.mode=mode
        # print(file_names)

        self.images = [os.path.join(image_dir, x ) for x in file_names]
        self.masks = [os.path.join(mask_dir, x ) for x in file_names]
        assert (len(self.images) == len(self.masks))



        self.as_tensor = T.Compose([
            # 将numpy的ndarray转换成形状为(C,H,W)的Tensor格式,且/255归一化到[0,1.0]之间
            T.ToTensor(),
        ])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = imgread(self.images[index])
        img = img / 1.0
        if(self.mode=="train"):
            target =imgread(self.masks[index])
            img,target=DataAugmentation(img,target,self.mode)
            img = np.ascontiguousarray(img)
            target=target.astype(np.int64)
            target=torch.tensor(target)
            img=self.as_tensor(img)
            return img.float(),target
        elif self.mode=="val":
            target=imgread(self.masks[index])
            img,target=DataAugmentation(img,target,self.mode)
            img = np.ascontiguousarray(img)
            target = target.astype(np.int64)
            target = torch.tensor(target)
            img = self.as_tensor(img)
            return img.float(), target
        elif self.mode=="test":
            target=imgread(self.masks[index])
            # img,target=DataAugmentation(img,target,self.mode)
            # img = np.ascontiguousarray(img)
            target = target.astype(np.int64)
            target = torch.tensor(target)
            img = self.as_tensor(img)
            return img.float(), target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    # print(images[0].dtype)
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


# dataset = VOCSegmentation(voc_root="/data/", transforms=get_transform(train=True))
# d1 = dataset[0]
# print(d1)
