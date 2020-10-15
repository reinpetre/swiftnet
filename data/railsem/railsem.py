from torch.utils.data import Dataset
from pathlib import Path

class_info = ["background", "rail"]
color_info = [[0, 0, 0], [255, 255, 255]]

id_to_map = {1: 255}

class Railsem(Dataset):
    num_classes = 2
    
    class_info = class_info
    color_info = color_info

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, root: Path, transforms: lambda x: x, subset='train', open_depth=False, labels_dir='raw_masks', epoch=None):

        self.root = root
        self.images_dir = self.root / 'raw_images' / subset
        self.labels_dir = self.root / 'raw_masks' / subset
        self.depth_dir = self.root / 'depth' / subset
        print("Image dir", self.images_dir)
        print("Labels dir", self.labels_dir)
        self.subset = subset
        self.has_labels = subset != 'test'
        self.open_depth = open_depth
        self.images = list(sorted(self.images_dir.glob('*.jpg')))
        if self.has_labels:
            self.labels = list(sorted(self.labels_dir.glob('*.png')))
        self.transforms = transforms
        self.epoch = epoch

        print(f'Num images: {len(self.images)}')
        if self.has_labels:
            print(f'Num labels: {len(self.labels)}')


    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        ret_dict = {
            'image': self.images[item],
            'name': self.images[item].stem,
            'subset': self.subset,
        }
        if self.has_labels:
            ret_dict['labels'] = self.labels[item]
        if self.epoch is not None:
            ret_dict['epoch'] = int(self.epoch.value)
        return self.transforms(ret_dict)
