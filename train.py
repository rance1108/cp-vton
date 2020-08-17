#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    
    parser.add_argument("--dataroot", default = "data/zalando")
    parser.add_argument('-td', "--tom_dataroot", default = "data/zalando")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_id.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 20000)
    parser.add_argument("--keep_step", type=int, default = 100000)
    parser.add_argument("--decay_step", type=int, default = 100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

def train_gmm(opt, train_loader, model, board):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
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
        head_mask = inputs['head_mask'].cuda()
        # if_c = inputs['if_c'].cuda()
        # import pdb 
        # pdb.set_trace()
        # grid, theta = model(agnostic, c)
        # warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        # warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        # warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

        warped_cloth = []
        warped_mask = []
        warped_grid = []
        visuals = []
        loss = 0
        
        for i in range(c.shape[1]):
            # if if_c[:,i].all() == True:
            input_agnostic = torch.cat([agnostic,pcm[:,i]],dim=1)
            grid, theta = model(input_agnostic, c[:,i])
            warped_cloth.append(F.grid_sample(c[:,i], grid, padding_mode='border'))
            warped_mask.append(F.grid_sample(cm[:,i], grid, padding_mode='zeros'))
            warped_grid.append(F.grid_sample(im_g, grid, padding_mode='zeros'))
            visuals.append([ [shape, im_h, im_pose], 
                       [c[:,i], warped_cloth[i], im_c[:,i]], 
                       [warped_grid[i], (warped_cloth[i]+im)*0.5, im]])

            loss += criterionL1(warped_cloth[i], im_c[:,i])    



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (step+1) % opt.display_count == 0:
            for j, k in zip(range(4),['combine_inner', 'combine_outer', 'combine_bottom', 'combine_shoe']):

                board_add_images(board, k, visuals[j], step+1)

            # board_add_images(board, 'combine_inner', visuals[0], step+1)
            # board_add_images(board, 'combine_outer', visuals[1], step+1)
            # board_add_images(board, 'combine_bottom', visuals[2], step+1)
            # board_add_images(board, 'combine_shoe', visuals[3], step+1)
            board.add_scalar('metric', loss.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))


def train_tom(opt, train_loader, model, board):
    model.cuda()
    model.train()
    
    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
        im = inputs['image'].cuda()
        im_nobg = inputs['im_nobg'].cuda()
        im_pose = inputs['pose_image'].cuda()
        pose_map = inputs['pose_map'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()

        # agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()

        bg = inputs['bg'].cuda()
        # padding = torch.zeros((im.shape[0],2,im.shape[2],im.shape[3])).cuda()


        # for i in range(c.shape[1]):

        #     if i < 1 :
        #         agnostic = torch.cat([shape, im_h, pose_map, padding], 1)
        #     else:
        #         agnostic = torch.cat([shape, p_tryon, pose_map, padding], 1)

        #     input_agnostic = torch.cat([agnostic,c[:,i]],dim=1)
        #     outputs = model(input_agnostic)

        #     p_rendered, m_composite = torch.split(outputs, 3,1)
        #     p_rendered = F.tanh(p_rendered)
        #     m_composite = F.sigmoid(m_composite)
        #     p_tryon = c[:,i] * m_composite+ p_rendered * (1 - m_composite)

        #     visuals.append([ [im_h, shape, im_pose], 
        #            [c[:,i], cm[:,i]*2-1, m_composite*2-1], 
        #            [p_rendered, p_tryon, im_nobg]])
        #     loss_mask += criterionMask(m_composite, cm[:,i])

        # loss_l1 = criterionL1(p_tryon, im_nobg)
        # loss_vgg = criterionVGG(p_tryon, im_nobg)

        # loss = loss_l1 + loss_vgg + loss_mask
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()



        agnostic = inputs['agnostic'].cuda()

        visuals = []

        loss_l1 = 0
        loss_vgg = 0
        loss_mask = 0
        loss = 0


        input_agnostic = torch.cat([agnostic,c.view(c.shape[0],c.shape[1]*c.shape[2],c.shape[3],c.shape[4])],dim=1)
        print(input_agnostic.shape)
        outputs = model(input_agnostic)

        p_rendered, m_composite = torch.split(outputs, [3,4],1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c[:,0] * m_composite[:,0]+ \
                    c[:,1] * m_composite[:,1]+ \
                    c[:,2] * m_composite[:,2]+ \
                    c[:,3] * m_composite[:,3]+ \
        p_rendered * (1 - torch.mean(m_composite,1))

        visuals.append([ [im_h, shape, im_pose], 
               [c[:,0], cm[:,0]*2-1, m_composite[:,0]*2-1],
               [c[:,1], cm[:,1]*2-1, m_composite[:,1]*2-1],
               [c[:,2], cm[:,2]*2-1, m_composite[:,2]*2-1],
               [c[:,3], cm[:,3]*2-1, m_composite[:,3]*2-1], 
               [p_rendered, p_tryon, im_nobg]])
        print(m_composite.shape,cm.shape,cm[:,0].shape,m_composite[:,0:0].shape)
        for i in range(4):
            loss_mask += criterionMask(m_composite[:,i], cm[:,i])

        loss_l1 = criterionL1(p_tryon, im_nobg)
        loss_vgg = criterionVGG(p_tryon, im_nobg)

        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        
        # outputs = model(torch.cat([agnostic, c],1))
        # p_rendered, m_composite = torch.split(outputs, 3,1)
        # p_rendered = F.tanh(p_rendered)
        # m_composite = F.sigmoid(m_composite)
        # p_tryon = c * m_composite+ p_rendered * (1 - m_composite)

        # visuals = [ [im_h, shape, im_pose], 
        #            [c, cm*2-1, m_composite*2-1], 
        #            [p_rendered, p_tryon, im]]
            
        # loss_l1 = criterionL1(p_tryon, im)
        # loss_vgg = criterionVGG(p_tryon, im)
        # loss_mask = criterionMask(m_composite, cm)
        # loss = loss_l1 + loss_vgg + loss_mask
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine_inner', visuals[0], step+1)
            board_add_images(board, 'combine_outer', visuals[1], step+1)
            board_add_images(board, 'combine_bottom', visuals[2], step+1)
            board_add_images(board, 'combine_shoe', visuals[3], step+1)
            board.add_scalar('metric', loss.item(), step+1)
            board.add_scalar('L1', loss_l1.item(), step+1)
            board.add_scalar('VGG', loss_vgg.item(), step+1)
            board.add_scalar('MaskL1', loss_mask.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f' 
                    % (step+1, t, loss.item(), loss_l1.item(), 
                    loss_vgg.item(), loss_mask.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))



def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))
   
    # create dataset 
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))
   
    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth'))
    elif opt.stage == 'TOM':
        model = UnetGenerator(25+7, 4+3, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_tom(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'tom_final.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
        
  
    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
