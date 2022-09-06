import os
import random
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
from PIL import Image
from skimage.transform import resize
from collections import OrderedDict

import albumentations as albu


def is_img_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])

class Augment:
    def __init__(self):
        pass

    def transform0(self, torch_tensor):
        return torch_tensor   
    def transform1(self, torch_tensor):
        #torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        #torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        #torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        #torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        #torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        #torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor


def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def load_img(filepath, transform=None):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    if (transform):
        img = transform(image=img)['image']
    # img = img.astype(np.float32)
    # img = img/255.
    return img
    
def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module.module.' in k:
                name = k[14:]
            elif 'module.' in k:
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def TensorResize(tensor, X):
    np_arr = tensor.cpu().detach().numpy()
    np_arr = np_arr.transpose((1, 2, 0))

    im_np_resize = resize(np_arr, (X, X))
    im_np_resize = im_np_resize.transpose((2, 0, 1))
    return torch.from_numpy(im_np_resize)
 
class train_set(Dataset):
    def __init__(self, dataset_dir, list_dir, label_aug=None, transform=None, crop_mode=1, img_size=256, use_rotLoss=False, label_smooth=0.0):
        super(train_set, self).__init__()

        self.dataset_dir = dataset_dir
        
        self.images = []
        f = open(list_dir, 'r')
        contents = f.readlines()
        for line in contents:
            path, label = line.strip('\n').split(' ')
            if (use_rotLoss and int(label)!=0):
                continue
            self.images.append({'path':path, 'label':int(label)})

        self.label_aug = label_aug
        self.transform = transform
        self.crop_mode = crop_mode
        self.img_size = img_size
        self.use_rotLoss = use_rotLoss
        self.smooth = label_smooth
        self.size = len(self.images)

        print("------------------------------------------------------------------")
        print('train set:', dataset_dir, '(list:', list_dir, ')')
        print('label_aug:', label_aug)
        print('transform:', transform)
        print('dataset size:', self.size)
        print("------------------------------------------------------------------")
        # self.aug = Augment()
        # self.transforms = [method for method in dir(self.aug) if callable(getattr(self.aug, method)) if not method.startswith('_')]    


    def __len__(self):
        return self.size


    def __getitem__(self, index):
        image_anno = self.images[index % self.size]
        image_path = os.path.join(self.dataset_dir, image_anno['path'])
        image = Image.open(image_path).convert('RGB')
        #print('#################', Image.fromarray(image).shape)
        label = image_anno['label']

        
        
        crop_mode = self.crop_mode
        if (crop_mode == -1):
            crop_mode = random.randint(1, 3)

        h, w, c = np.array(image).shape
        if (h < self.img_size or w < self.img_size):
            crop_mode = 1
        if (crop_mode == 1):  
            long_edge = max(h, w)
            Padding = transforms.Pad(((long_edge-w)//2, (long_edge-h)//2, (long_edge-w+1)//2, (long_edge-h+1)//2), padding_mode='constant', fill=128)
            Resize = transforms.Resize((self.img_size, self.img_size))
            image = Resize(Padding(image))
        if (crop_mode == 2):
            Resize = transforms.Resize(self.img_size)
            image = Resize(image)
        

        if (self.use_rotLoss):      
            # assert label==0
            rot_times = random.randint(0, 3)
            label = (label + rot_times) % 4
            for rot_time in range(rot_times):
                image = image.transpose(Image.ROTATE_270)

            
        image = self.transform(image)

        #image = image.permute(2,0,1)
        
        one_hot = torch.full((4,), self.smooth/4.)
        one_hot[label] += (1.-self.smooth)
        # one_hot[(label+2)%4] = self.smooth/2.
        # one_hot[label] = (1.-self.smooth)

        return image, one_hot

