#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, load_checkpoint

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images, save_images
from torchvision.utils import save_image


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    
    parser.add_argument("--dataroot", default = "data/zalando")
    parser.add_argument("--tom_dataroot", default = "data/zalando")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_id.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for test')
    parser.add_argument("--display_count", type=int, default = 1)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

def test_gmm(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    base_name = os.path.basename(opt.name)
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)

    # save_dir = os.path.join(opt.result_dir, base_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    # if not os.path.exists(warp_cloth_dir):
    #     os.makedirs(warp_cloth_dir)
    # warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    # if not os.path.exists(warp_mask_dir):
    #     os.makedirs(warp_mask_dir)

    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        
        c_names = inputs['c_name']
        im_name = inputs['im_name'][0]
        warp_cloth_dir = os.path.join(save_dir, im_name)
        if not os.path.exists(warp_cloth_dir):
            os.makedirs(warp_cloth_dir)



        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        pcm = inputs['parse_cloth_mask'].cuda()
        im_c =  inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
        # head_mask = inputs['head_mask'].cuda()
            
        # grid, theta = model(agnostic, c)
        # warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        # warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        # warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

        warped_cloth = []
        warped_mask = []
        warped_grid = []
        visuals = []

        for i in range(c.shape[1]):
            input_agnostic = torch.cat([agnostic,pcm[:,i]],dim=1)
            grid, theta = model(input_agnostic, c[:,i])
            warped_cloth.append(F.grid_sample(c[:,i], grid, padding_mode='border'))
            warped_mask.append(F.grid_sample(cm[:,i], grid, padding_mode='zeros'))
            warped_grid.append(F.grid_sample(im_g, grid, padding_mode='zeros'))


            visuals.append([ [shape, im_h, im_pose], 
                       [c[:,i], warped_cloth[i], im_c[:,i]], 
                       [warped_grid[i], (warped_cloth[i]+im)*0.5, im]])
            # print(c_names[i][0][:-4] +'_wc.png',c_names[i][0][:-4] +'_wcm.png')
            # print((warped_cloth[i]).shape,warped_mask[i].shape)
            cname1 = c_names[i][0][:-4] +'_wc.png'
            cname2 = c_names[i][0][:-4] +'_wcm.png'
            cname3 = c_names[i][0][:-4] +'_orgwc.png'
            # cname4 = c_names[i][0][:-4] +'_123.png'
            # cname5 = c_names[i][0][:-4] +'_1234.png'
            # print("a:",im_c[:,i].max(),"b:",im_c[:,i].min(),"c:",warped_cloth[i].max(),"d:",warped_cloth[i].min(),
            #     "e:",warped_mask[i].max(),"f:",warped_mask[i].min())
            save_image((warped_cloth[i]+1)*0.5, os.path.join(warp_cloth_dir, cname1)) 
            save_image((im_c[:,i]+1)*0.5, os.path.join(warp_cloth_dir, cname3)) 
            save_image(warped_mask[i]*2-1, os.path.join(warp_cloth_dir, cname2)) 
            # save_image(im, os.path.join(warp_cloth_dir, cname4)) 
            # save_image((im+1)*0.5, os.path.join(warp_cloth_dir, cname5)) 

        for i in range(c.shape[1]):

            input_agnostic = torch.cat([agnostic,pcm[:,i]],dim=1)
            grid, theta = model(input_agnostic, c[:,i])
            warped_cloth.append(F.grid_sample(c[:,i], grid, padding_mode='border'))
            warped_mask.append(F.grid_sample(cm[:,i], grid, padding_mode='zeros'))
            warped_grid.append(F.grid_sample(im_g, grid, padding_mode='zeros'))
            visuals.append([ [shape, im_h, im_pose], 
                       [c[:,i], warped_cloth[i], im_c[:,i]],
                       [cm[:,i]*2-1, warped_mask[i]*2-1, pcm[:,i]*2-1],
                       [warped_grid[i], (warped_cloth[i]+im)*0.5, im]])

            if i < len(c.shape[1])-1:
                cname1 = c_names[i][0][:-4] +'_wc.png'
                cname2 = c_names[i][0][:-4] +'_wcm.png'
                cname3 = c_names[i][0][:-4] +'_orgwc.png'
            else:
                cname1 = '5_wc.png'
                cname2 = '5_wcm.png'
                cname3 = '5_orgwc.png'
            # cname4 = c_names[i][0][:-4] +'_123.png'
            # cname5 = c_names[i][0][:-4] +'_1234.png'
            # print("a:",im_c[:,i].max(),"b:",im_c[:,i].min(),"c:",warped_cloth[i].max(),"d:",warped_cloth[i].min(),
            #     "e:",warped_mask[i].max(),"f:",warped_mask[i].min())
            save_image((warped_cloth[i]+1)*0.5, os.path.join(warp_cloth_dir, cname1)) 
            save_image((im_c[:,i]+1)*0.5, os.path.join(warp_cloth_dir, cname3)) 
            save_image(warped_mask[i]*2-1, os.path.join(warp_cloth_dir, cname2)) 
            # save_image(im, os.path.join(warp_cloth_dir, cname4)) 
            # save_image((im+1)*0.5, os.path.join(warp_cloth_dir, cname5)) 

        # save_images(warped_cloth, c_names, warp_cloth_dir) 
        # save_images(warped_mask*2-1, c_names, warp_mask_dir) 
            
        if (step+1) % opt.display_count == 0:
            for j, k in zip(range(5),['combine_inner', 'combine_outer', 'combine_bottom',
                                         'combine_shoe_left', 'combine_shoe_right']):
                board_add_images(board, k, visuals[j], step+1)


        # if (step+1) % opt.display_count == 0:
        #     board_add_images(board, 'combine_inner', visuals[0], step+1)
        #     board_add_images(board, 'combine_outer', visuals[1], step+1)
        #     board_add_images(board, 'combine_bottom', visuals[2], step+1)
        #     board_add_images(board, 'combine_shoe', visuals[3], step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)


def test_tom(opt, test_loader, model, board):
    model.cuda()
    model.eval()
    
    base_name = os.path.basename(opt.checkpoint)
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try_on_dir = os.path.join(save_dir, 'try-on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)
    print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        
        im_names = inputs['im_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        
        outputs = model(torch.cat([agnostic, c],1))
        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        visuals = [ [im_h, shape, im_pose], 
                   [c, 2*cm-1, m_composite], 
                   [p_rendered, p_tryon, im]]
            
        save_images(p_tryon, im_names, try_on_dir) 
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)


def main():
    opt = get_opt()
    print(opt)
    print("Start to test stage: %s, named: %s!" % (opt.stage, opt.name))
   
    # create dataset 
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))
   
    # create model & train
    if opt.stage == 'GMM':
        model = GMM(opt)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_gmm(opt, train_loader, model, board)
    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_tom(opt, train_loader, model, board)
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
  
    print('Finished test %s, named: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
