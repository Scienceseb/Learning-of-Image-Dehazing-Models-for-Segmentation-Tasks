from os import listdir
from os.path import join
import torch.utils.data as data
import torchvision.transforms as transforms
from util import is_image_file, load_img
import numpy as np
from PIL import Image
import cv2



class DatasetFromFolder_3(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder_3, self).__init__()
        self.fog_path = join(image_dir, "a")
        self.photo_path = join(image_dir, "b")
        self.seg_path=join(image_dir,'c')
        self.image_filenames = [x for x in listdir(self.fog_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]


        self.transform = transforms.Compose(transform_list)



        

    def __getitem__(self, index):
        # Load Image

        input = load_img(join(self.fog_path, self.image_filenames[index]))
        input = input.convert("RGB")
        input = self.transform(input)




        target = load_img(join(self.photo_path, self.image_filenames[index]))
        target = target.convert("RGB")
        target = self.transform(target)



        seg = load_img(join(self.seg_path, self.image_filenames[index]))
        seg=seg.convert("RGB")

        seg = self.transform(seg)
        
        nom= self.image_filenames[index]



        return input, target, seg,nom

    def __len__(self):
        return len(self.image_filenames)
