from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import os
import csv
import torch


class Loader(Dataset):
    def __init__(self, root, transform=None, target_transform=None):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []
        self.bounding_boxes = None
        self.labels = None

        imgs_list = os.listdir(self.root)
        root_csv = os.path.join(self.root, 'ground_truth.csv')

        for img_file in imgs_list:
            if img_file.endswith('.jpg'):
                self.samples.append(img_file)

        bounds_pot = []
        with open(root_csv) as csv_file:
            csv_r = csv.reader(csv_file, delimiter=',')
            count = 0
            for row in csv_r:
                if count !=0: #ignore first row
                    xmin, xmax, ymin, ymax = int(row[1]), int(row[3]), int(row[2]), int(row[4])
                    bounds_pot.append([xmin, xmax, ymin, ymax])
                count+=1

        self.bounding_boxes = torch.as_tensor(bounds_pot, dtype=torch.float32)
        self.labels = torch.ones((len(imgs_list),), dtype=torch.int64)

    def __getitem__(self, index):

        img = self.samples[index]
        img_path = os.path.join(self.root, img)
        im = Image.open(img_path)
        pix = np.array(im)
        pix = pix[:, :, 0:3]
        im = Image.fromarray(np.uint8(pix))

        bounding_box = self.bounding_boxes[index]
        label = self.labels[index]

        target = {"boxes": bounding_box, "labels": label}

        if self.transform is not None:
            im = self.transform(im)

        return im, target

    def __len__(self):
        return len(self.samples)
