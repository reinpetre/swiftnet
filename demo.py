import argparse
from pathlib import Path
import importlib.util
from evaluation import evaluate_semseg
import torch
import numpy as np
import cv2
from evaluation import StorePreds

class ToTensor(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''
    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        im = im.transpose(2, 0, 1).astype(np.float32)
        im = torch.from_numpy(im).div_(255)
        dtype, device = im.dtype, im.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        im = im.sub_(mean).div_(std).clone()
        if not lb is None:
            lb = torch.from_numpy(lb.astype(np.int64).copy()).clone()
        return dict(im=im, lb=lb)


def import_module(path):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parser = argparse.ArgumentParser(description='Detector train')
parser.add_argument('--config', type=str, help='Path to configuration .py file', default="./configs/rn18_single_scale.py")
parser.add_argument('--profile', dest='profile', action='store_true', help='Profile one forward pass')
parser.add_argument('--img-path', dest='img_path', type=str, default='./rs07650.png',)

if __name__ == '__main__':
    args = parser.parse_args()
    conf_path = Path(args.config)
    conf = import_module(args.config)

    model = conf.model.cuda()
    
    print(type(model))
    
    # model.eval()
    # model.cuda()

    class_info = conf.dataset_val.class_info

    palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

    # prepare data
    to_tensor = ToTensor(
        mean=(0.3257, 0.3690, 0.3223),  # city, rgb
        std=(0.2112, 0.2148, 0.2115),
    )
    image = cv2.imread(args.img_path)[:, :, ::-1]
    image = to_tensor(dict(im=image, lb=None))['im'].unsqueeze(0).cuda()
    

    outputs = model(image, target_size =(1024, 1920), image_size=(1024,1920))[0].argmax(dim=1).squeeze().detach().cpu().numpy()

    # inference
    #pred = torch.argmax(outputs[0]).squeeze().detach().cpu().numpy()
    print(np.unique(outputs))
    outputs[np.where(outputs == 1)] = 255
    print(np.unique(outputs))
    print(outputs.shape)
   # print(pred.shape)
    cv2.imwrite('./res.jpg', outputs)
