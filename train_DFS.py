from __future__ import print_function
import argparse
import os
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from networks import define_G, define_D, GANLoss, print_network
from data_3 import get_training_set, get_test_set, get_val_set
import torch.backends.cudnn as cudnn
from features import Vgg16, Resnet101
import random
import numpy
import torch.nn.functional as F
import pytorch_ssim
from visualizer import Visualizer
from collections import OrderedDict
import sys

visualizer = Visualizer()


def main(name_exp, segloss=False, cuda=True, finetune=False):
    # Training settings
    parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
    parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=8, help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='generator filt+ers in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
    opt = parser.parse_args()

    cudnn.benchmark = True



    def val():
        net_current = "path_exp/checkpoint/DFS/{}/netG_model_current.pth".format(name_exp)
        netVal = torch.load(net_current)
        netVal.eval()
        SEG_NET.eval()
        features.eval()
        with torch.no_grad():
            total_mse = 0
            total_mse2 = 0
            avg_psnr_depth = 0
            avg_psnr_dehaze = 0
            avg_ssim_depth = 0
            avg_ssim_dehaze = 0
            for batch in validation_data_loader:
                input, target, depth = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
                if cuda == True:
                    input = input.cuda()
                    target = target.cuda()
                    depth = depth.cuda()
                    
                

                dehaze = netVal(input)
                prediction = SEG_NET(dehaze)

                avg_ssim_dehaze += pytorch_ssim.ssim(dehaze, target).item()

                mse = criterionMSE(prediction, depth)
                total_mse += mse.item()
                avg_psnr_depth += 10 * log10(1 / mse.item())

                mse2 = criterionMSE(dehaze, target)
                total_mse2 += mse2.item()
                avg_psnr_dehaze += 10 * log10(1 / mse2.item())

                avg_ssim_depth += pytorch_ssim.ssim(prediction, depth).item()


                visual_ret_val = OrderedDict()

                visual_ret_val['Haze'] = input
                visual_ret_val['Seg estimate'] = prediction
                visual_ret_val['Dehaze '] = dehaze
                visual_ret_val['GT dehaze'] = target
                visual_ret_val['GT Seg '] = depth

                visualizer.display_current_results(visual_ret_val, epoch, True)


            print("===> Validation")
            #f.write("===> Testing: \r\n")

            print("===> PSNR seg: {:.4f} ".format(avg_psnr_depth / len(validation_data_loader)))
            #f.write("===> PSNR depth: {:.4f} \r\n".format(avg_psnr_depth / len(validation_data_loader)))

            print("===> Mse seg: {:.4f} ".format(total_mse / len(validation_data_loader)))
            #f.write("===> Mse depth: {:.4f} \r\n".format(total_mse / len(validation_data_loader)))

            print("===> SSIM seg: {:.4f} ".format(avg_ssim_depth / len(validation_data_loader)))
            #f.write("===> SSIM depth: {:.4f} \r\n".format(avg_ssim_depth / len(validation_data_loader)))

            return total_mse / len(validation_data_loader)






    def testing():
        path = "path_exp/checkpoint/DFS/{}/netG_model_best.pth".format(name_exp)
        net = torch.load(path)
        net.eval()
        SEG_NET.eval()
        features.eval()
        with torch.no_grad():
            total_mse = 0
            total_mse2 = 0
            avg_psnr_depth = 0
            avg_psnr_dehaze = 0
            avg_ssim_depth = 0
            avg_ssim_dehaze = 0
            for batch in testing_data_loader:
                input, target, depth = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
                if cuda == True:
                    input = input.cuda()
                    target = target.cuda()
                    depth = depth.cuda()

                dehaze = net(input)
                prediction = SEG_NET(dehaze)

                avg_ssim_dehaze += pytorch_ssim.ssim(dehaze, target).item()

                mse = criterionMSE(prediction, depth)
                total_mse += mse.item()
                avg_psnr_depth += 10 * log10(1 / mse.item())

                mse2 = criterionMSE(dehaze, target)
                total_mse2 += mse2.item()
                avg_psnr_dehaze += 10 * log10(1 / mse2.item())

                avg_ssim_depth += pytorch_ssim.ssim(prediction, depth).item()

            print("===> Testing")
            print("===> PSNR seg: {:.4f} ".format(avg_psnr_depth / len(testing_data_loader)))
            print("===> Mse seg: {:.4f} ".format(total_mse / len(testing_data_loader)))
            print("===> SSIM seg: {:.4f} ".format(avg_ssim_depth / len(testing_data_loader)))
            print("===> PSNR dehaze: {:.4f} ".format(avg_psnr_dehaze / len(testing_data_loader)))
            print("===> SSIM dehaze: {:.4f} ".format(avg_ssim_dehaze / len(testing_data_loader)))





    def checkpoint():
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(os.path.join("path_exp/checkpoint/DFS", name_exp)):
            os.mkdir(os.path.join("path_exp/checkpoint/DFS", name_exp))
        net_g_model_out_path = "path_exp/checkpoint/DFS/{}/netG_model_best.pth".format(name_exp)
        net_d_model_out_path = "path_exp/checkpoint/DFS/{}/netD_model_best.pth".format(name_exp)
        torch.save(netG, net_g_model_out_path)
        torch.save(netD, net_d_model_out_path)


    def checkpoint_current():
        if not os.path.exists(os.path.join("path_exp/checkpoint/DFS", name_exp)):
            os.mkdir(os.path.join("path_exp/checkpoint/DFS", name_exp))
        net_g_model_out_path = "path_exp/checkpoint/DFS/{}/netG_model_current.pth".format(name_exp)
        torch.save(netG, net_g_model_out_path)

    def checkpoint_seg():
        if not os.path.exists(os.path.join("path_exp/checkpoint/DFS", name_exp)):
            os.mkdir(os.path.join("path_exp/checkpoint/DFS", name_exp))
        net_g_model_out_path = "path_exp/checkpoint/DFS/{}/seg_net.pth".format(name_exp)
        torch.save(SEG_NET, net_g_model_out_path)



    torch.manual_seed(opt.seed)
    if cuda==True:
        torch.cuda.manual_seed(opt.seed)

    print(" ")
    print(name_exp)
    print(" ")

    print('===> Loading datasets')
    train_set = get_training_set('path_exp/cityscape/HAZE')
    val_set = get_val_set('path_exp/cityscape/HAZE')
    test_set = get_test_set('path_exp/cityscape/HAZE')


    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    validation_data_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    testing_data_loader= DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

    print('===> Building model')
    netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, [0])
    netD = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'batch', False, [0])

    criterionGAN = GANLoss()
    criterionL1 = nn.L1Loss()
    criterionMSE = nn.MSELoss()

    # setup optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))



    print('---------- Networks initialized -------------')
    print_network(netG)
    print_network(netD)
    print('-----------------------------------------------')


    real_a = torch.FloatTensor(opt.batchSize, opt.input_nc, 256, 256)
    real_b = torch.FloatTensor(opt.batchSize, opt.output_nc, 256, 256)
    real_c = torch.FloatTensor(opt.batchSize, opt.output_nc, 256, 256)

    if cuda==True:
        netD = netD.cuda()
        netG = netG.cuda()
        criterionGAN = criterionGAN.cuda()
        criterionL1 = criterionL1.cuda()
        criterionMSE = criterionMSE.cuda()
        real_a = real_a.cuda()
        real_b = real_b.cuda()
        real_c=real_c.cuda()

    real_a = Variable(real_a)
    real_b = Variable(real_b)
    real_c = Variable(real_c)



    SEG_NET = torch.load("path_exp/SEG_NET.pth")

    optimizerSeg = optim.Adam(SEG_NET.parameters(), lr=opt.lr/10, betas=(opt.beta1, 0.999))



    features = Vgg16()

    if cuda==True:
        SEG_NET.cuda()
        features.cuda()


    bon =100000000
    for epoch in range(opt.nEpochs):
        features.eval()

        if finetune== True and epoch>50:
            SEG_NET.train()
        else:
            SEG_NET.eval()

        loss_epoch_gen=0
        loss_epoch_dis=0
        total_segloss=0
        loss_seg=0
        i=0
        for iteration, batch in enumerate(training_data_loader, 1):

            netG.train()
            i=i+1

            # forward
            real_a_cpu, real_b_cpu, real_c_cpu = batch[0], batch[1], batch[2]

            with torch.no_grad():
                real_a = real_a.resize_(real_a_cpu.size()).copy_(real_a_cpu)

            with torch.no_grad():
                real_b = real_b.resize_(real_b_cpu.size()).copy_(real_b_cpu)

            with torch.no_grad():
                real_c = real_c.resize_(real_c_cpu.size()).copy_(real_c_cpu)


            fake_b = netG(real_a)

            ############################
            # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
            ###########################

            optimizerD.zero_grad()

            # train with fake
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = netD.forward(fake_ab.detach())
            loss_d_fake = criterionGAN(pred_fake, False)

            # train with real
            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = netD.forward(real_ab)
            loss_d_real = criterionGAN(pred_real, True)

            # Combined loss
            loss_d = (loss_d_fake + loss_d_real) * 0.5

            loss_d.backward()

            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
            ##########################
            optimizerG.zero_grad()
            # First, G(A) should fake the discriminator
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = netD.forward(fake_ab)
            loss_g_gan = criterionGAN(pred_fake, True)


            # Second, G(A) = B
            loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb

            features_y = features(fake_b)
            features_x = features(real_b)

            loss_content = criterionMSE(features_y[1], features_x[1])*10


            if segloss == True:
                fake_seg = SEG_NET(fake_b)
                loss_seg = criterionMSE(fake_seg, real_c) * 10

                total_segloss += loss_seg.item()

                features_y = features(fake_seg)
                features_x = features(real_c)

                ssim_seg = criterionMSE(features_y[1], features_x[1]) * 10

                loss_g = loss_g_gan + loss_g_l1 + loss_content + loss_seg


            else:
                loss_g = loss_g_gan + loss_g_l1+loss_content

            loss_epoch_gen+=loss_g.item()
            loss_epoch_dis+=loss_d.item()





            if finetune== True and epoch>50:
                loss_g.backward(retain_graph=True)

                optimizerG.step()

                loss_seg=loss_seg

                loss_seg.backward()

                optimizerSeg.zero_grad()

                optimizerSeg.step()

            else:
                loss_g.backward()
                optimizerG.step()



            errors_ret = OrderedDict()
            errors_ret['Total_G'] = float(loss_g)
            errors_ret['Content'] = float(loss_content)
            errors_ret['GAN'] = float(loss_g_gan)
            errors_ret['L1'] = float(loss_g_l1)
            errors_ret['D'] = float(loss_d)



            if i % 10 == 0:  # print training losses and save logging information to the disk
                if i > 0:
                    visualizer.plot_current_losses(epoch, i/(len(training_data_loader)*opt.batchSize), errors_ret)




        print("===> Epoch[{}]: Loss_D: {:.4f} Loss_G: {:.4f} Loss Seg: {:.4f} ".format(epoch, loss_epoch_dis,loss_epoch_gen, total_segloss))
        checkpoint_current()
        MSE=val()
        if MSE < bon:
            bon = MSE
            checkpoint()
            checkpoint_seg()
            print("BEST EPOCH SAVED")

    testing()




main("DFS", segloss=True, cuda=True, finetune=False)







