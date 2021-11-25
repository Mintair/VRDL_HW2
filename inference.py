import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader, Dataset

import json
from PIL import Image


PATH = './HW2_net_cos.pth'
file_path = 'test/test/'

# Transform

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset


class MyDataset(Dataset):
    def __init__(self, img_dir, img_list, transform):
        super(MyDataset, self).__init__()
        self.img_dir = img_dir
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_dir + self.img_list[idx])
        return self.transform(img)




if __name__ == '__main__':
    submission = []
    net = torch.load(PATH)
    submission = []
    test_img_list = []
    output_list = {}
    test_img_list = os.listdir("test/test")
    
    test_loader = DataLoader(MyDataset("test/test/",
                                       test_img_list,
                                       transform_test),
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=0)
    
    with torch.no_grad():
        net.eval()
        for i, img in enumerate(test_loader, 0):
            pred = net(img)
            output_list[test_img_list[i]] = pred[0]
        
    for img_name in test_img_list:
        print(img_name)
        # the image_name is as same as the image_id
        image_id = int(img_name[:-4])
        # add each detection box infomation into list
        box_num = int(output_list[img_name]['boxes'].size()[0])
        for box in range(box_num):
            det_box_info = {}

            # An integer to identify the image
            det_box_info["image_id"] = image_id
        
            # A list ( [left_x, top_y, width, height] )
            left = output_list[img_name]["boxes"][box][0].item()
            top = output_list[img_name]["boxes"][box][1].item()
            width = output_list[img_name]["boxes"][box][2].item() - left
            height = output_list[img_name]["boxes"][box][3].item() - top
            det_box_info["bbox"] = [left, top, width, height]
            # A float number between 0 ~ 1 which means the confidence of the bbox
            det_box_info["score"] = output_list[img_name]["scores"][box].item()

            # An integer which means the label class
            det_box_info["category_id"] = output_list[img_name]["labels"][box].item() + 1
            print(det_box_info)
            submission.append(det_box_info)

    # Write the list to answer.json 
    json_object = json.dumps(submission, indent=4)

    with open("answer.json", "w") as outfile:
        outfile.write(json_object)





