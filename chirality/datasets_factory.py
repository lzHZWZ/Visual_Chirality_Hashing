
import torch
from torchvision import datasets, transforms
from PIL import Image
import os

data_label = {"original":1, "flip":0}

def get_transform(need_size):
    transforms_list = []
    transforms_list += [
        transforms.Resize((need_size,need_size)),
    ]

    transforms_list += [transforms.ToTensor()]

    data_transforms = transforms.Compose(transforms_list)
    return data_transforms

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self,data_dir,transform):
        self.data_info = get_img_info(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_info)

def get_img_info(data_dir):
    subroot = []
    data_info = list()
    for root, dirs, _ in os.walk(data_dir):
        for sub_dir in dirs:
            img_names = os.listdir(os.path.join(root, sub_dir))
            for i in range(len(img_names)):
                img_name = img_names[i]
                absolute_image_info = os.path.join(root, sub_dir, img_name)
                label = data_label[sub_dir]
                data_info.append((absolute_image_info, int(label)))
    return data_info