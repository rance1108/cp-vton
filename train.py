#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint, NLayerDiscriminator, GANLoss, CNNAE2ResNet

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images


import itertools

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



def train_gmm(opt, train_loader, G_A, G_B, D_A, D_B, board):
    G_A.cuda()
    G_A.train()
    
    # G_B.cuda()
    # G_B.train()

    # D_A.cuda()
    # D_A.train()

    # D_B.cuda()
    # D_B.train()


    # criterion
    criterionL1 = nn.L1Loss()
    # criterionGAN = GANLoss('lsgan').cuda()
    # criterionCycle = torch.nn.L1Loss()
    # criterionIdt = torch.nn.L1Loss()
    """Calculate the loss for generators G_A and G_B"""
    # lambda_idt = 0.5    
    # lambda_A = 10.0
    # lambda_B = 10.0
    # lambda_L1 = 10.0

    # optimizer
    optimizerG = torch.optim.Adam(itertools.chain(G_A.parameters()), lr=opt.lr, betas=(0.5, 0.999))
    # optimizerD = torch.optim.Adam(itertools.chain(D_A.parameters()), lr=opt.lr, betas=(0.5, 0.999))

    # optimizerG = torch.optim.Adam(itertools.chain(G_A.parameters(), G_B.parameters()), lr=opt.lr, betas=(0.5, 0.999))
    # optimizerD = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()), lr=opt.lr, betas=(0.5, 0.999))

    # optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))


    schedulerG = torch.optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))

    # schedulerD = torch.optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda = lambda step: 1.0 -
    #         max(0, step - opt.keep_step) / float(opt.decay_step + 1))



    from networks import CNNAE2ResNet
    model_path = "/media/rance/hdd/relighting_humans_pytorch/models/shared_model_059.pth"
    
    m_shared = CNNAE2ResNet()
    m_shared.load_state_dict(torch.load(model_path))

    m_shared.train_dropout = False  
    m_shared.cuda()
    
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
        im_inout = inputs['im_inout'].cuda()
        parse_inout = inputs['parse_inout'].cuda()
        bg = inputs['bg'].cuda()

        transport = inputs['transport'].cuda()
        mask_1024 = inputs['mask_1024'].cuda()
        albedo = inputs['albedo'].cuda()
        shading = inputs['shading'].cuda()
        light = inputs['light'].cuda()

        # print("TRANSPORT22222", transport.shape,transport.max(),transport.min())
        # print("light222222222", light.shape,light.max(),light.min())
        # print("mask_1024222222222", mask_1024.shape,mask_1024.max(),mask_1024.min())
        # print("albedo22222", albedo.shape,albedo.max(),albedo.min())
        # print("shading2222222", shading.shape,shading.max(),shading.min())

        # aa, vv, cc = m_shared(mask_1024)
        # print("aa",aa.shape,aa.max(),aa.min())
        # print("vv",vv.shape,vv.max(),vv.min())
        # print("cc",cc.shape,cc.max(),cc.min())

        # head_mask = inputs['head_mask'].cuda()
        # print(torch.min(torch.sum([cm[:,0],cm[:,1]],dim=1)),torch.min(torch.sum([cm[:,0],cm[:,1]],dim=1)).shape)
        # cm[:,0] = torch.sum([])

        C_unwarp_warp = []
        M_unwarp_warp = []
        G_unwarp_warp = []

        # C_unwarp_warp_unwarp = []
        # M_unwarp_warp_unwarp = []
        # G_unwarp_warp_unwarp = []

        # C_warpGT_unwarp = []
        # M_warpGT_unwarp = []
        # G_warpGT_unwarp = []



        # C_warpGT_unwarp_warp = []
        # M_warpGT_unwarp_warp = []
        # G_warpGT_unwarp_warp = []


        # C_itself_A = []
        # M_itself_A = []
        # G_itself_A = []


        # C_itself_B = []
        # M_itself_B = []
        # G_itself_B = []

        visuals = []
        loss = 0
        
        input_agnostic = torch.cat([agnostic,parse_inout],dim=1)
        grid1, theta1 = G_A(input_agnostic, torch.cat([c[:,0],pcm[:,0]],dim=1), None) 
        c1 = F.grid_sample(c[:,0], grid1, padding_mode='border')
        m1 = F.grid_sample(cm[:,0], grid1, padding_mode='zeros')
        g1 = F.grid_sample(im_g, grid1, padding_mode='zeros')

        grid2, theta2 = G_A(input_agnostic, torch.cat([c[:,1],pcm[:,1]],dim=1), None) 

        c2 = F.grid_sample(c[:,1], grid2, padding_mode='border')
        m2 = F.grid_sample(cm[:,1], grid2, padding_mode='zeros')
        g2 = F.grid_sample(im_g, grid2, padding_mode='zeros')

        grid3, theta3 = G_A(input_agnostic, torch.cat([c[:,2],pcm[:,2]],dim=1), None) 

        c3 = F.grid_sample(c[:,2], grid3, padding_mode='border')
        m3 = F.grid_sample(cm[:,2], grid3, padding_mode='zeros')
        g3 = F.grid_sample(im_g, grid3, padding_mode='zeros')

        grid4, theta4 = G_A(input_agnostic, torch.cat([c[:,3],pcm[:,3]],dim=1), None) 

        c4 = F.grid_sample(c[:,3], grid4, padding_mode='border')
        m4 = F.grid_sample(cm[:,3], grid4, padding_mode='zeros')
        g4 = F.grid_sample(im_g, grid4, padding_mode='zeros')


        grid5, theta5 = G_A(input_agnostic, torch.cat([c[:,4],pcm[:,4]],dim=1), None) 

        c5 = F.grid_sample(c[:,4], grid5, padding_mode='border')
        m5 = F.grid_sample(cm[:,4], grid5, padding_mode='zeros')
        g5 = F.grid_sample(im_g, grid5, padding_mode='zeros')


        # c_com = c1 * pcm[:,0]  + c2 * pcm[:,1]
        # c_com = c1 * pcm[:,0] + c2 * pcm[:,1]

        c_com = c1 * pcm[:,0]  + c2 * pcm[:,1] + c3 * pcm[:,2] + c4 * pcm[:,3] + c5 * pcm[:,4]

        # for param in D_A.parameters():
        #     param.requires_grad = False


        loss_L1_all = 0.1* criterionL1(c_com, im_inout)
        # loss_L1_1 = criterionL1(c1 * pcm[:,0], im_c[:,0] * pcm[:,0])  
        loss_L1_1 = criterionL1(c1 , im_c[:,0] )
        loss_L1_2 = criterionL1(c2 , im_c[:,1] )
        loss_L1_3 = criterionL1(c3 , im_c[:,2] )
        loss_L1_4 = criterionL1(c4 , im_c[:,3] )
        loss_L1_5 = criterionL1(c5 , im_c[:,4] )
        # loss_L1_2 = criterionL1(c2 * pcm[:,1], im_c[:,1] * pcm[:,1])

        # loss_G_A = criterionGAN(D_A(c_com), True)

        # loss_G = loss_G_A + loss_L1
        # loss_L1 = loss_L1_1 
        loss_L1 = loss_L1_1 + loss_L1_2 + loss_L1_3 + loss_L1_4 + loss_L1_5 + loss_L1_all
        optimizerG.zero_grad()
        loss_L1.backward()
        # loss_G.backward()
        optimizerG.step() 

        # for param in D_A.parameters():
        #     param.requires_grad = True
            
        # optimizerD.zero_grad() 

        # loss_DA = backward_D_basic(D_A, im_inout, c_com)

        # optimizerD.step()

        visuals.append([ [shape, im_pose], 
                   [c[:,0], im_c[:,0]],
                   [c[:,1], im_c[:,1]],
                   [c[:,2], im_c[:,2]],
                   [c[:,3], im_c[:,3]],
                   [c[:,4], im_c[:,4]],
                   [im_c[:,0], pcm[:,0]*2-1],
                   [im_c[:,1], pcm[:,1]*2-1],
                   [im_c[:,2], pcm[:,2]*2-1],
                   [im_c[:,3], pcm[:,3]*2-1],
                   [im_c[:,4], pcm[:,4]*2-1],
                   [c1, c2],
                   [c3, c4],
                   [c5, c5],
                   [m1*2-1, m2*2-1],
                   [m3*2-1, m4*2-1],
                   [m5*2-1, m5*2-1],
                   [g1,g2],
                   [g3,g4],
                   [g5,g5],
                   [c_com,c_com+bg],
                   [im_inout,parse_inout*2-1]
                   ])


        if (step+1) % opt.display_count == 0:
            for j, k in zip(range(1),['combined']):
                board_add_images(board, k, visuals[j], step+1)
            board.add_scalar('loss_L1', loss_L1.item(), step+1)
            board.add_scalar('loss_L1_1', loss_L1_1.item(), step+1)
            board.add_scalar('loss_L1_2', loss_L1_2.item(), step+1)
            board.add_scalar('loss_L1_3', loss_L1_3.item(), step+1)
            board.add_scalar('loss_L1_4', loss_L1_4.item(), step+1)
            board.add_scalar('loss_L1_5', loss_L1_5.item(), step+1)
            board.add_scalar('loss_L1_all', loss_L1_all.item(), step+1)

            # t = time.time() - iter_start_time
            # print('step: %8d, time: %.3f, loss: %4f ' \
            #     % (step+1, t, loss_L1.item() ), flush=True)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f , loss_l1: %4f , loss_l2: %4f , loss_l3: %4f , loss_l4: %4f , loss_l5: %4f , loss_lall: %4f ' \
                % (step+1, t, loss_L1.item(), loss_L1_1.item(), loss_L1_2.item(), loss_L1_3.item(), loss_L1_4.item(), loss_L1_5.item(), loss_L1_all.item()), flush=True)

            # t = time.time() - iter_start_time
            # print('step: %8d, time: %.3f, loss: %4f , loss_l1: %4f loss_G: %4f loss_D: %4f' \
            #     % (step+1, t, loss_G, loss_L1.item(), loss_G_A.item(), loss_DA.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(G_A, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d_G_A.pth' % (step+1)))
            # save_checkpoint(D_A, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d_D_A.pth' % (step+1)))



        # # for i in range(c.shape[1]):
        # for i in range(2):

        #     input_agnostic = torch.cat([agnostic,pcm[:,i]],dim=1)
        #     # print(type(agnostic),type(pcm[:,i]),type(input_agnostic))
        #     grid, theta = G_A(input_agnostic, c[:,i], None)                                                  #G_A(A)
        #     C_unwarp_warp.append(F.grid_sample(c[:,i], grid, padding_mode='border'))
        #     M_unwarp_warp.append(F.grid_sample(cm[:,i], grid, padding_mode='zeros'))
        #     G_unwarp_warp.append(F.grid_sample(im_g, grid, padding_mode='zeros'))
        #     # input_agnostic = torch.cat([agnostic,M_unwarp_warp[i]],dim=1)
        #     # grid2, theta = G_B(input_agnostic, C_unwarp_warp[i])                                # G_B(G_A(A))
        #     # C_unwarp_warp_unwarp.append(F.grid_sample(C_unwarp_warp[i], grid2, padding_mode='border'))
        #     # M_unwarp_warp_unwarp.append(F.grid_sample(M_unwarp_warp[i], grid2, padding_mode='zeros'))
        #     # G_unwarp_warp_unwarp.append(F.grid_sample(im_g, grid2, padding_mode='zeros'))



        #     # input_agnostic = torch.cat([agnostic,cm[:,i]],dim=1)
        #     # grid3, theta = G_B(input_agnostic, im_c[:,i])                                # G_B(B) # G_A(G_B(B))
        #     # C_warpGT_unwarp.append(F.grid_sample(im_c[:,i], grid3, padding_mode='border'))
        #     # M_warpGT_unwarp.append(F.grid_sample(pcm[:,i], grid3, padding_mode='zeros'))
        #     # G_warpGT_unwarp.append(F.grid_sample(im_g, grid3, padding_mode='zeros'))
        #     # input_agnostic = torch.cat([agnostic,M_warpGT_unwarp[i]],dim=1)
        #     # grid4, theta = G_A(input_agnostic, C_warpGT_unwarp[i])                                 # G_A(G_B(B))
        #     # C_warpGT_unwarp_warp.append(F.grid_sample(C_warpGT_unwarp[i], grid4, padding_mode='border'))
        #     # M_warpGT_unwarp_warp.append(F.grid_sample(M_warpGT_unwarp[i], grid4, padding_mode='zeros'))
        #     # G_warpGT_unwarp_warp.append(F.grid_sample(im_g, grid4, padding_mode='zeros'))

        #     ##################BACKPROP#########################################################
        #                 # Identity loss
        #     # if lambda_idt > 0:
        #     #     # G_A should be identity if real_B is fed: ||G_A(B) - B||


        #     #     input_agnostic = torch.cat([agnostic,pcm[:,i]],dim=1)
        #     #     grid5, theta = G_A(input_agnostic,im_c[:,i], None)                                                  #G_A(A)
        #     #     C_itself_A.append(F.grid_sample(im_c[:,i], grid5, padding_mode='border'))
        #     #     M_itself_A.append(F.grid_sample(pcm[:,i], grid5, padding_mode='zeros'))
        #     #     G_itself_A.append(F.grid_sample(im_g, grid5, padding_mode='zeros'))

        #     #     loss_idt_A = criterionIdt(C_itself_A[i], im_c[:,i]) * lambda_B * lambda_idt


        #     #     # # G_B should be identity if real_A is fed: ||G_B(A) - A||

        #     #     # input_agnostic = torch.cat([agnostic,cm[:,i]],dim=1)
        #     #     # grid6, theta = G_B(input_agnostic,c[:,i])                                                  
        #     #     # C_itself_B.append(F.grid_sample(c[:,i], grid6, padding_mode='border'))
        #     #     # M_itself_B.append(F.grid_sample(cm[:,i], grid6, padding_mode='zeros'))
        #     #     # G_itself_B.append(F.grid_sample(im_g, grid6, padding_mode='zeros'))


        #     #     # loss_idt_B = criterionIdt(C_itself_B[i], c[:,i]) * lambda_A * lambda_idt
        #     # else:
        #     #     loss_idt_A = 0
        #     #     # loss_idt_B = 0


        # C_unwarp_warp_after = torch.stack(C_unwarp_warp, dim=1)
        # M_unwarp_warp_after = torch.stack(M_unwarp_warp, dim=1)
        # G_unwarp_warp_after = torch.stack(G_unwarp_warp, dim=1)

        # # C_unwarp_warp_after, M_unwarp_warp_after, G_unwarp_warp_after = G_A(C_unwarp_warp, M_unwarp_warp,
        # #  G_unwarp_warp, translator=True)


        # for param in D_A.parameters():
        #     param.requires_grad = False

        # # for param in D_B.parameters():
        # #     param.requires_grad = False


        # optimizerG.zero_grad()


        # # loss_mask = torch.max(1-torch.sum(M_unwarp_warp,dim=1),torch.tensor([0.]))

        # # loss_mask = torch.sum(F.relu(1-torch.sum(M_unwarp_warp,dim=1)))

        # # # GAN loss D_A(G_A(A))
        # # loss_G_A = criterionGAN(D_A(C_unwarp_warp_after.contiguous().view(C_unwarp_warp_after.size(0),
        # #  C_unwarp_warp_after.size(1)*C_unwarp_warp_after.size(2), C_unwarp_warp_after.size(3), C_unwarp_warp_after.size(4))), True)
        # # GAN loss D_A(G_A(A))
        # loss_G_A = criterionGAN(D_A(C_unwarp_warp_after[:,0]), True)

        # # # GAN loss D_B(G_B(B))
        # # loss_G_B = criterionGAN(D_B(C_warpGT_unwarp[i]), True)

        # # # Forward cycle loss || G_B(G_A(A)) - A||
        # # loss_cycle_A = criterionCycle(C_unwarp_warp_unwarp[i], c[:,i]) * lambda_A

        # # # Backward cycle loss || G_A(G_B(B)) - B||
        # # loss_cycle_B = criterionCycle(C_warpGT_unwarp_warp[i], im_c[:,i]) * lambda_B

        # # loss_L1 =  criterionL1(C_unwarp_warp_after.contiguous().view(C_unwarp_warp_after.size(0),
        # #  C_unwarp_warp_after.size(1)*C_unwarp_warp_after.size(2), C_unwarp_warp_after.size(3), C_unwarp_warp_after.size(4)), 
        # # im_c.contiguous().view(im_c.size(0),
        # #  im_c.size(1)*im_c.size(2), im_c.size(3), im_c.size(4))) * lambda_L1

        # loss_L1 =  criterionL1(C_unwarp_warp_after[:,0], im_c[:,0]) * lambda_L1
        # # combined loss and calculate gradients
        # loss_G = loss_G_A + loss_L1
        # # loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B + loss_L1
        # loss_G.backward()
        # optimizerG.step()  


        # for param in D_A.parameters():
        #     param.requires_grad = True
            
        # # for param in D_B.parameters():
        # #     param.requires_grad = True

        # optimizerD.zero_grad() 




        # loss_DA = backward_D_basic(D_A, im_c[:,0], 
        # C_unwarp_warp_after[:,0])

        # # loss_DA = backward_D_basic(D_A, im_c.contiguous().view(im_c.size(0),
        # #  im_c.size(1)*im_c.size(2), im_c.size(3), im_c.size(4)), 
        # # C_unwarp_warp_after.contiguous().view(C_unwarp_warp_after.size(0),
        # #  C_unwarp_warp_after.size(1)*C_unwarp_warp_after.size(2), C_unwarp_warp_after.size(3), C_unwarp_warp_after.size(4)))
        #     # loss_DB = backward_D_basic(D_B, c[:,i], C_warpGT_unwarp_warp[i])
        # optimizerD.step()


        #     # visuals.append([ [shape, im_h, im_pose], 
        #     #            [c[:,i], C_unwarp_warp[i], im_c[:,i]],
        #     #            [cm[:,i]*2-1, M_unwarp_warp[i]*2-1, pcm[:,i]*2-1],

        #     #            [G_unwarp_warp[i], (C_unwarp_warp[i]+im)*0.5, im],

        #     #            [G_unwarp_warp_unwarp[i], (C_unwarp_warp_unwarp[i]+c[:,i])*0.5, M_unwarp_warp_unwarp[i]*2-1],

        #     #            [G_warpGT_unwarp[i], (C_warpGT_unwarp[i]+c[:,i])*0.5, M_warpGT_unwarp[i]*2-1],

        #     #            [G_warpGT_unwarp_warp[i], (C_warpGT_unwarp_warp[i]+im)*0.5, M_warpGT_unwarp_warp[i]*2-1],

        #     #            [G_itself_A[i], (C_itself_A[i]), M_itself_A[i]*2-1],

        #     #            [G_itself_B[i], (C_itself_B[i]), M_itself_B[i]*2-1]])


        # for i in range(1):

        #     visuals.append([ [shape, im_h, im_pose], 

        #                # [c[:,i], C_unwarp_warp[:,i], im_c[:,i]],

        #                # [cm[:,i]*2-1, M_unwarp_warp[:,i]*2-1, pcm[:,i]*2-1],

        #                [c[:,i], C_unwarp_warp_after[:,i], im_c[:,i]],

        #                [cm[:,i]*2-1, M_unwarp_warp_after[:,i]*2-1, pcm[:,i]*2-1],

        #                # [G_unwarp_warp[:,i], (C_unwarp_warp[:,i]), im],

        #                [G_unwarp_warp_after[:,i], C_unwarp_warp_after[:,i], M_unwarp_warp_after[:,i]],

        #                # [G_itself_A[i], (C_itself_A[i]), M_itself_A[i]*2-1]
        #                ])





        #     # loss += criterionL1(C_unwarp_warp[i], im_c[:,i])    



        # # optimizerG.zero_grad()
        # # loss.backward()
        # # optimizerG.step()
            
        # if (step+1) % opt.display_count == 0:
        #     # for j, k in zip(range(3),['combine_inner', 'combine_outer', 'combine_bottom']):
        #     # for j, k in zip(range(5),['combine_inner', 'combine_outer', 'combine_bottom',
        #     #                              'combine_shoe_left', 'combine_shoe_right']):
        #     for j, k in zip(range(1),['combine_inner']):

        #         board_add_images(board, k, visuals[j], step+1)

        #     # board_add_images(board, 'combine_inner', visuals[0], step+1)
        #     # board_add_images(board, 'combine_outer', visuals[1], step+1)
        #     # board_add_images(board, 'combine_bottom', visuals[2], step+1)
        #     # board_add_images(board, 'combine_shoe', visuals[3], step+1)
        #     board.add_scalar('TOTAL loss', loss_G, step+1)
        #     board.add_scalar('loss_G_A', loss_G_A.item(), step+1)
        #     # board.add_scalar('loss_G_B', loss_G_B.item(), step+1)
        #     board.add_scalar('loss_L1', loss_L1.item(), step+1)
        #     # board.add_scalar('loss_cycle_A', loss_cycle_A.item(), step+1)
        #     # board.add_scalar('loss_cycle_B', loss_cycle_B.item(), step+1)
        #     # board.add_scalar('loss_idt_A', loss_idt_A.item(), step+1)
        #     # board.add_scalar('loss_idt_B', loss_idt_B.item(), step+1)
        #     board.add_scalar('loss_DA', loss_DA.item(), step+1)
        #     # board.add_scalar('loss_DB', loss_DB.item(), step+1)

        #     # board.add_scalar('loss_mask', loss_mask.item(), step+1)

        #     t = time.time() - iter_start_time
        #     # print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)
        #     print('step: %8d, time: %.3f, loss: %4f , loss_l1: %4f loss_G: %4f loss_D: %4f' \
        #         % (step+1, t, loss_G, loss_L1.item(), loss_G_A.item(), loss_DA.item()), flush=True)
        #     # print('step: %8d, time: %.3f, loss: %4f , loss_l1: %4f loss_G: %4f loss_D: %4f loss_idt: %4f loss_mask: %4f' \
        #     #     % (step+1, t, loss_G, loss_L1.item(), loss_G_A.item(), loss_DA.item(),
        #     #         loss_idt_A.item(),loss_mask), flush=True)
        #     # print('step: %8d, time: %.3f, loss: %4f , loss_l1: %4f loss_G: %4f loss_D: %4f loss_cyc: %4f loss_idt: %4f' \
        #     #     % (step+1, t, loss_G, loss_L1.item(), loss_G_A.item()+loss_G_B.item(), loss_DA.item()+loss_DB.item(), \
        #     #         loss_cycle_A.item()+loss_cycle_B.item(),loss_idt_A.item()+loss_idt_B.item() ), flush=True)

        # if (step+1) % opt.save_count == 0:
        #     save_checkpoint(G_A, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d_G_A.pth' % (step+1)))
        #     # save_checkpoint(G_B, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d_G_B.pth' % (step+1)))
        #     save_checkpoint(D_A, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d_D_A.pth' % (step+1)))
        #     # save_checkpoint(D_B, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d_D_B.pth' % (step+1)))


def train_tom(opt, train_loader, model, board):
    model.cuda()
    model.train()
    
    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()

    criterionRelight = CNNAE2ResNet().cuda()
    
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
        combined = inputs['combined'].cuda()
        # padding = torch.zeros((im.shape[0],2,im.shape[2],im.shape[3])).cuda()

        transport = inputs['transport'].cuda()
        mask_1024 = inputs['mask_1024'].cuda()
        albedo = inputs['albedo'].cuda()
        shading = inputs['shading'].cuda()
        light = inputs['light'].cuda()


        combined_mask = torch.clamp(torch.sum(cm[:,0]+cm[:,1]+cm[:,2]+cm[:,3]+cm[:,4],dim=1,keepdim=True),0,1)
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


        # c[:,0] = (c[:,0] * cm[:,0])
        # c[:,1] = (c[:,1] * cm[:,1])
        # c[:,2] = (c[:,2] * cm[:,2])
        # c[:,3] = (c[:,3] * cm[:,3])
        # c[:,4] = (c[:,4] * cm[:,4])

        # agnostic = torch.cat([shape, bg, pose_map], 1)
        input_agnostic = torch.cat([agnostic,c.view(c.shape[0],c.shape[1]*c.shape[2],c.shape[3],c.shape[4]),
            combined,combined_mask],dim=1)
        # input_agnostic = torch.cat([agnostic,c.view(c.shape[0],c.shape[1]*c.shape[2],c.shape[3],c.shape[4])],dim=1)
        outputs = model(input_agnostic)

        p_rendered, m_composite = torch.split(outputs, [3,1],1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)

        # p_tryon = 0.2*(c[:,0] * m_composite[:,0:1])+ \
        #             0.2*(c[:,1] * m_composite[:,1:2])+ \
        #             0.2*(c[:,2] * m_composite[:,2:3])+ \
        #             0.2*(c[:,3] * m_composite[:,3:4])+ \
        #             0.2*(c[:,4] * m_composite[:,4:5])+ \
        # p_rendered * (1 - torch.mean(m_composite,1,keepdim=True))

        # p_tryon = m_composite * (((c[:,0] )+ \
        #             (c[:,1] )+ \
        #             (c[:,2] )+ \
        #             (c[:,3] )+ \
        #             (c[:,4] ))+(1-torch.sum(cm,1)))+ \
        # p_rendered * (1 - m_composite)

        p_tryon = (m_composite * combined)+ \
        p_rendered * (1 - m_composite)



        visuals.append([ [shape, im_pose], 
               [c[:,0], cm[:,0]*2-1],
               [c[:,1], cm[:,1]*2-1],
               [c[:,2], cm[:,2]*2-1],
               [c[:,3], cm[:,3]*2-1],
               [c[:,4], cm[:,4]*2-1],
               [bg, bg],
               [combined,combined_mask*2-1],
               [p_rendered, m_composite*2-1], 
               [p_tryon, im]])


        # visuals.append([ [im_h, shape, im_pose], 
        #        [c[:,0], cm[:,0]*2-1, cm[:,0]*2-1],
        #        [c[:,1], cm[:,1]*2-1, cm[:,1]*2-1],
        #        [c[:,2], cm[:,2]*2-1, cm[:,2]*2-1],
        #        [c[:,3], cm[:,3]*2-1, cm[:,3]*2-1], 
        #        [c[:,4], cm[:,4]*2-1, cm[:,4]*2-1], 
        #        [(((c[:,0] )+ \
        #             (c[:,1] )+ \
        #             (c[:,2] )+ \
        #             (c[:,3] )+ \
        #             (c[:,4] ))+(1-torch.sum(cm,1))),
        #        bg, m_composite*2-1], 
        #        [p_rendered, p_tryon, im]])
        # for i in range(5):
        #     loss_mask += criterionMask(m_composite[:,i:i+1], cm[:,i])


        loss_mask = criterionMask(m_composite, combined_mask)

        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)


        p_tryon = p_tryon[:,[2,1,0]]

        p_tryon = torch.nn.functional.interpolate(p_tryon, size=(1024,1024), scale_factor=None, mode='nearest', align_corners=None)
        p_tryon = p_tryon * mask_1024
        res_transport, res_albedo, res_light = criterionRelight(p_tryon)
        print(res_transport.shape,res_albedo.shape,res_light.shape)
        shade = torch.matmul(res_albedo,res_light)
        loss_shading = criterionL1(shade, shading)

        loss = 0.1*loss_l1 + 0.5* loss_vgg + 0.1*loss_mask + loss_shading
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
            # board_add_images(board, 'combine_outer', visuals[1], step+1)
            # board_add_images(board, 'combine_bottom', visuals[2], step+1)
            # board_add_images(board, 'combine_shoe', visuals[3], step+1)
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


def backward_D_basic(netD, real, fake):
    """Calculate GAN loss for the discriminator
    Parameters:
        netD (network)      -- the discriminator D
        real (tensor array) -- real images
        fake (tensor array) -- images generated by a generator
    Return the discriminator loss.
    We also call loss_D.backward() to calculate the gradients.
    """

    criterionGAN = GANLoss('lsgan').cuda()
    # Real
    pred_real = netD(real)
    loss_D_real = criterionGAN(pred_real, True)
    # Fake
    pred_fake = netD(fake.detach())
    loss_D_fake = criterionGAN(pred_fake, False)
    # Combined loss and calculate gradients
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    loss_D.backward()
    return loss_D









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
        G_A = GMM(opt)
        # G_B = GMM(opt)
        # D_A = NLayerDiscriminator(3, 64, n_layers=3, norm_layer=nn.InstanceNorm2d)
        # D_B = NLayerDiscriminator(3, 64, n_layers=3, norm_layer=nn.InstanceNorm2d)  

        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)

        # train_gmm(opt, train_loader, G_A, G_B, D_A, D_B, board)

        train_gmm(opt, train_loader, G_A, None, None, None, board)

        # save_checkpoint(G_A, os.path.join(opt.checkpoint_dir, opt.name, 'gmmA2B_final.pth'))
        # save_checkpoint(G_B, os.path.join(opt.checkpoint_dir, opt.name, 'gmmB2A_final.pth'))

        save_checkpoint(G_A, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth'))


    elif opt.stage == 'TOM':
        model = UnetGenerator(20+15+3+1, 1+3, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_tom(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'tom_final.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
        
  
    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
