import os
import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from PIL import Image


bbox_prop = ['height', 'left', 'top', 'width', 'label']


def get_img_boxes(f, idx=0):
    """
    get the 'height', 'left', 'top', 'width', 'label'
    of bounding boxes of an image
    :param f: h5py.File
    :param idx: index of the image
    :return: dictionary
    """
    meta = {key: [] for key in bbox_prop}

    box = f[bboxs[idx][0]]
    for key in box.keys():
        if box[key].shape[0] == 1:
            meta[key].append(int(box[key][0][0]))
        else:
            for i in range(box[key].shape[0]):
                meta[key].append(int(f[box[key][i][0]][()].item()))
    return meta


def get_img_name(f, idx=0):
    img_name = ''.join(map(chr, f[names[idx][0]][()].flatten()))
    return(img_name)


class MyDataset(Dataset):
    def __init__(self, img_dir, img_list, bbox_list, transform):
        super(MyDataset, self).__init__()
        self.img_dir = img_dir
        self.img_list = img_list
        self.bbox_list = bbox_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_dir + self.img_list[idx])
        num_bboxs = len(self.bbox_list[idx]['label'])
        boxes = []
        labels = []
        for i in range(num_bboxs):
            height = self.bbox_list[idx]['height'][i]
            left = self.bbox_list[idx]['left'][i]
            top = self.bbox_list[idx]['top'][i]
            width = self.bbox_list[idx]['width'][i]
            label = self.bbox_list[idx]['label'][i]
            boxes.append([left, top, left+width, top+height])
            labels.append(label)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Let label in range of 0-9(label should start at 0)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # target is a dict
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        return transform(img, target)


def collate_fn(batch):
    return tuple(zip(*batch))


def resize(boxes, dims=(128, 128)):
    new_dims = torch.FloatTensor([dims[1],
                                  dims[0],
                                  dims[1],
                                  dims[0]]).unsqueeze(0)
    return boxes * new_dims


def transform(img, target):
    h = img.height
    w = img.width
    old_dims = torch.FloatTensor([w, h, w, h]).unsqueeze(0)
    # percent coordinates * new dims
    target['boxes'] = resize(target['boxes'] / old_dims)
    return transform_train(img), target

transform_train = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomOrder([
       transforms.RandomHorizontalFlip(),
       transforms.RandomRotation(15,)
    ]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


file_path = 'train/train/'
f = h5py.File('train/train/digitStruct.mat', 'r')
names = f['digitStruct/name']
bboxs = f['digitStruct/bbox']
train_imgs_list = []
train_bboxs_list = []
size = names.size
size_max = 0
for i in range(size):
    print("Index:", i)
    im = Image.open(file_path + get_img_name(f, i))
    w, h = im.size
    size_max = max(w, h, size_max)
    train_imgs_list.append(get_img_name(f, i))
    train_bboxs_list.append(get_img_boxes(f, i))


if __name__ == '__main__':
    train_loader = DataLoader(MyDataset(file_path,
                                        train_imgs_list,
                                        train_bboxs_list,
                                        transform_train),
                              batch_size=2,
                              collate_fn=collate_fn,
                              shuffle=True,
                              num_workers=0)

    # Pretrained model: Faster R-CNN
    net_ft = models.detection.faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = net_ft.roi_heads.box_predictor.cls_score.in_features
    # class: 0-10, 0 is background
    num_classes = 11
    # Replace the predictor head with our defined num_classes
    net_ft.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(net_ft.parameters(), lr=0.005, momentum=0.9)
    cos_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft,
                                                            T_max=10)
    net_ft = torch.nn.DataParallel(net_ft,
                                   device_ids=range(torch.cuda.device_count()))
    torch.backends.cudnn.benchmark = True
    net_ft = net_ft.cuda()

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, targets = data
            images = list(image.cuda() for image in images)
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            # zero the parameter gradients
            optimizer_ft.zero_grad()

            # forward + backward + optimize
            loss_dict = net_ft(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer_ft.step()

            # print statistics
            running_loss += losses.item()
            print('[%d, %5d] loss: %.3f'
                  % (epoch + 1, i + 1, running_loss / (i + 1)))
        cos_lr_scheduler.step()

    print('Finished Training')

    # Save trained neural network
    PATH = './HW2_net_cos.pth'
    torch.save(net_ft, PATH)

    
