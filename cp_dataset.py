#coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os.path as osp
import numpy as np
import json

class CPDataset(data.Dataset):
    """Dataset for CP-VTON.
    """
    def __init__(self, opt):
        super(CPDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode # train or test or self-defined
        self.stage = opt.stage # GMM or TOM
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Resize((256,192)),   \
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        # load data list
        im_names = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_names.append(line.strip())


        self.im_names = im_names

    def name(self):
        return "CPDataset"

    def __getitem__(self, index):
        im_name = self.im_names[index]

        # cloth image & cloth mask
        if self.stage == 'GMM':
            c = []
            cm = []
            if_c = []
            clothes = ["0.png", "1.png", "2.png", "4.png"]
            masks = ["0_mask.png", "1_mask.png", "2_mask.png", "4_mask.png"]
            for f_name, fm_name in zip(clothes, masks):
                c_path = osp.join(self.data_path, im_name, f_name)
                cm_path = osp.join(self.data_path, im_name, fm_name)
                if os.path.isfile(c_path) and os.path.isfile(cm_path):
                    c.append(Image.open(c_path))
                    cm.append(Image.open(cm_path))
                    if_c.append(True)
                else:
                    c.append(Image.new('RGB',(102,147)))
                    c.append(Image.new('L',(102,147)))
                    if_c.append(False)

        else:
            c = Image.open(osp.join(self.data_path, 'warp-cloth', c_name))
            cm = Image.open(osp.join(self.data_path, 'warp-mask', c_name))
        


        for i in range(len(c)):
            c[i] = self.transform(c[i])  # [-1,1]
        c = torch.cat(c,dim=0)

        for i in range(len(cm)):
            cm[i] = np.array(cm[i])
            cm[i] = (cm[i] >= 128).astype(np.float32)
            cm[i]= torch.from_numpy(cm[i]) # [0,1]
            cm[i].unsqueeze_(0)

        cm = torch.cat(cm,dim=0)

        # person image 
        im = Image.open(osp.join(self.data_path, im_name, "99.png"))
        im = self.transform(im) # [-1,1]

        # load parsing image
        im_parse = Image.open(osp.join(self.data_path, im_name, "12.png"))
        parse_array = np.array(im_parse)
        parse_shape = (parse_array > 0).astype(np.float32)
        parse_head = (parse_array == 1).astype(np.float32)

        parse_cloth = []

        for n,i in enumerate(if_c):
            if i == False:
                parse_cloth.append((parse_array > 5).astype(np.float32))
            else:
                parse_cloth.append((parse_array == n+2).astype(np.float32))

        # parse_inner = (parse_array == 2).astype(np.float32)
        # parse_outer = (parse_array == 3).astype(np.float32)
        # parse_bottom = (parse_array == 4).astype(np.float32)
        # parse_shoe = (parse_array == 5).astype(np.float32)
        

        # shape downsample
        parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.fine_width//16, self.fine_height//16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        shape = self.transform(parse_shape) # [-1,1]
        phead = torch.from_numpy(parse_head) # [0,1]

        pcm_cloth = []
        im_cloth = []
        for i in parse_cloth:
            pcm_cloth.append(torch.from_numpy(i)) # [0,1]

            # inner cloth
            im_cloth.append((im * i + (1 - i))) # [-1,1], fill 1 for other parts

        pcm_cloth = torch.cat(pcm_cloth,dim=0)
        im_cloth = torch.cat(im_cloth,dim=0)


        # pcm_i = torch.from_numpy(parse_inner) # [0,1]
        # pcm_o = torch.from_numpy(parse_outer) # [0,1]
        # pcm_b = torch.from_numpy(parse_bottom) # [0,1]
        # pcm_s = torch.from_numpy(parse_shoe) # [0,1]
        # im_i = im * pcm_i + (1 - pcm_i) # [-1,1], fill 1 for other parts
        # im_o = im * pcm_o + (1 - pcm_o) # [-1,1], fill 1 for other parts
        # im_b = im * pcm_b + (1 - pcm_b) # [-1,1], fill 1 for other parts
        # im_s = im * pcm_s + (1 - pcm_s) # [-1,1], fill 1 for other parts

        im_h = im * phead - (1 - phead) # [-1,1], fill 0 for other parts

        # load pose points

        
        pose_name = osp.join(self.data_path, im_name, "pose.txt")
        with open(pose_name, 'r') as f:
            pose_label = f.readline().split()
            pose_data = np.array(pose_label,dtype=int)
            pose_data = pose_data.reshape((-1,2))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i,0]
            pointy = pose_data[i,1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = self.transform(one_map)
            pose_map[i] = one_map[0]

        # just for visualization
        im_pose = self.transform(im_pose)
        
        # cloth-agnostic representation
        agnostic = torch.cat([shape, im_h, pose_map], 0) 

        if self.stage == 'GMM':
            im_g = Image.open('grid.png')
            im_g = self.transform(im_g)
        else:
            im_g = ''

        result = {
            'c_name':   c_name,     # for visualization
            'im_name':  im_name,    # for visualization or ground truth
            'cloth':    c,          # for input
            'cloth_mask':     cm,   # for input
            'image':    im,         # for visualization
            'agnostic': agnostic,   # for input
            'parse_cloth': im_cloth,    # for ground truth
            # 'parse_cloth': im_c,    # for ground truth
            'shape': shape,         # for visualization
            'head': im_h,           # for visualization
            'pose_image': im_pose,  # for visualization
            'grid_image': im_g,     # for visualization
            }

        return result

    def __len__(self):
        return len(self.im_names)

class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()

        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default = "./data/zalando")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_id.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=1)
    
    opt = parser.parse_args()
    dataset = CPDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    print('Size of the dataset: %05d, dataloader: %04d' \
            % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

    from IPython import embed; embed()

