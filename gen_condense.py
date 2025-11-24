import os
import sys
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

import models.resnet as RN
import models.convnet as CN
import models.resnet_ap as RNAP
import models.densenet_cifar as DN
from gan_model import Generator, Discriminator
from utils import load_data, AverageMeter, accuracy, Normalize, Logger, rand_bbox, str2bool, calc_gradient_penalty, test
from augment import DiffAug, remove_aug, diffaug


def define_model(args, num_classes, e_model=None):
    '''Obtain model for training and validating
    '''
    if e_model:
        model = e_model
    else:
        model = args.match_model

    if args.data == 'mnist' or args.data == 'fmnist':
        nch = 1
    else:
        nch = 3

    if model == 'convnet':
        return CN.ConvNet(num_classes, channel=nch)
    elif model == 'resnet10':
        return RN.ResNet(args.data, 10, num_classes, nch=nch)
    elif model == 'resnet18':
        return RN.ResNet(args.data, 18, num_classes, nch=nch)
    elif model == 'resnet34':
        return RN.ResNet(args.data, 34, num_classes, nch=nch)
    elif model == 'resnet50':
        return RN.ResNet(args.data, 50, num_classes, nch=nch)
    elif model == 'resnet101':
        return RN.ResNet(args.data, 101, num_classes, nch=nch)
    elif model == 'resnet10_ap':
        return RNAP.ResNetAP(args.data, 10, num_classes, nch=nch)
    elif model == 'resnet18_ap':
        return RNAP.ResNetAP(args.data, 18, num_classes, nch=nch)
    elif model == 'resnet34_ap':
        return RNAP.ResNetAP(args.data, 34, num_classes, nch=nch)
    elif model == 'resnet50_ap':
        return RNAP.ResNetAP(args.data, 50, num_classes, nch=nch)
    elif model == 'resnet101_ap':
        return RNAP.ResNetAP(args.data, 101, num_classes, nch=nch)
    elif model == 'densenet':
        return DN.densenet_cifar(num_classes)


def train(args, epoch, generator, discriminator, optim_g, optim_d, trainloader, criterion, aug, aug_rand):
    '''The main training function for the generator
    '''
    generator.train()
    gen_losses = AverageMeter()
    disc_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (img_real, lab_real) in enumerate(trainloader):
        img_real = img_real.cuda()
        lab_real = lab_real.cuda()

        # train the generator
        discriminator.eval()
        optim_g.zero_grad()

        # obtain the noise with one-hot class labels
        noise = torch.normal(0, 1, (args.batch_size, args.dim_noise))
        lab_onehot = torch.zeros((args.batch_size, args.num_classes))
        lab_onehot[torch.arange(args.batch_size), lab_real] = 1
        noise[torch.arange(args.batch_size), :args.num_classes] = lab_onehot[torch.arange(args.batch_size)]
        noise = noise.cuda()

        img_syn = generator(noise)
        gen_source, gen_class = discriminator(img_syn)
        gen_source = gen_source.mean()
        gen_class = criterion(gen_class, lab_real)
        gen_loss = - gen_source + gen_class

        gen_loss.backward()
        optim_g.step()

        # train the discriminator
        discriminator.train()
        optim_d.zero_grad()
        lab_syn = torch.randint(args.num_classes, (args.batch_size,))
        noise = torch.normal(0, 1, (args.batch_size, args.dim_noise))
        lab_onehot = torch.zeros((args.batch_size, args.num_classes))
        lab_onehot[torch.arange(args.batch_size), lab_syn] = 1
        noise[torch.arange(args.batch_size), :args.num_classes] = lab_onehot[torch.arange(args.batch_size)]
        noise = noise.cuda()
        lab_syn = lab_syn.cuda()

        with torch.no_grad():
            img_syn = generator(noise)

        disc_fake_source, disc_fake_class = discriminator(img_syn)
        disc_fake_source = disc_fake_source.mean()
        disc_fake_class = criterion(disc_fake_class, lab_syn)

        disc_real_source, disc_real_class = discriminator(img_real)
        acc1, acc5 = accuracy(disc_real_class.data, lab_real, topk=(1, 5))
        disc_real_source = disc_real_source.mean()
        disc_real_class = criterion(disc_real_class, lab_real)

        gradient_penalty = calc_gradient_penalty(args, discriminator, img_real, img_syn)

        disc_loss = disc_fake_source - disc_real_source + disc_fake_class + disc_real_class + gradient_penalty
        disc_loss.backward()
        optim_d.step()

        gen_losses.update(gen_loss.item())
        disc_losses.update(disc_loss.item())
        top1.update(acc1.item())
        top5.update(acc5.item())

        if (batch_idx + 1) % args.print_freq == 0:
            print('[Train Epoch {} Iter {}] G Loss: {:.3f}({:.3f}) D Loss: {:.3f}({:.3f}) D Acc: {:.3f}({:.3f})'.format(
                epoch, batch_idx + 1, gen_losses.val, gen_losses.avg, disc_losses.val, disc_losses.avg, top1.val, top1.avg)
            )


def validate(args, generator, testloader, criterion, aug_rand):
    '''Validate the generator performance
    '''
    all_best_top1 = []
    all_best_top5 = []
    for e_model in args.eval_model:
        print('Evaluating {}'.format(e_model))
        model = define_model(args, args.num_classes, e_model).cuda()
        model.train()
        optim_model = torch.optim.SGD(model.parameters(), args.eval_lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay)

        generator.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        best_top1 = 0.0
        best_top5 = 0.0
        for epoch_idx in range(args.epochs_eval):
            for batch_idx in range(10 * args.ipc // args.batch_size + 1):
                # obtain pseudo samples with the generator
                lab_syn = torch.randint(args.num_classes, (args.batch_size,))
                noise = torch.normal(0, 1, (args.batch_size, args.dim_noise))
                lab_onehot = torch.zeros((args.batch_size, args.num_classes))
                lab_onehot[torch.arange(args.batch_size), lab_syn] = 1
                noise[torch.arange(args.batch_size), :args.num_classes] = lab_onehot[torch.arange(args.batch_size)]
                noise = noise.cuda()
                lab_syn = lab_syn.cuda()

                with torch.no_grad():
                    img_syn = generator(noise)
                    img_syn = aug_rand((img_syn + 1.0) / 2.0)

                if np.random.rand(1) < args.mix_p and args.mixup_net == 'cut':
                    lam = np.random.beta(args.beta, args.beta)
                    rand_index = torch.randperm(len(img_syn)).cuda()

                    lab_syn_b = lab_syn[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(img_syn.size(), lam)
                    img_syn[:, :, bbx1:bbx2, bby1:bby2] = img_syn[rand_index, :, bbx1:bbx2, bby1:bby2]
                    ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img_syn.size()[-1] * img_syn.size()[-2]))

                    output = model(img_syn)
                    loss = criterion(output, lab_syn) * ratio + criterion(output, lab_syn_b) * (1. - ratio)
                else:
                    output = model(img_syn)
                    loss = criterion(output, lab_syn)

                acc1, acc5 = accuracy(output.data, lab_syn, topk=(1, 5))

                losses.update(loss.item(), img_syn.shape[0])
                top1.update(acc1.item(), img_syn.shape[0])
                top5.update(acc5.item(), img_syn.shape[0])

                optim_model.zero_grad()
                loss.backward()
                optim_model.step()

            if (epoch_idx + 1) % args.test_interval == 0:
                test_top1, test_top5, test_loss = test(args, model, testloader, criterion)
                print('[Test Epoch {}] Top1: {:.3f} Top5: {:.3f}'.format(epoch_idx + 1, test_top1, test_top5))
                if test_top1 > best_top1:
                    best_top1 = test_top1
                    best_top5 = test_top5

        all_best_top1.append(best_top1)
        all_best_top5.append(best_top5)

    return all_best_top1, all_best_top5


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ipc', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--epochs-eval', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval-lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--eval-model', type=str, nargs='+', default=['convnet'])
    parser.add_argument('--dim-noise', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--test-interval', type=int, default=200)

    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./results/')
    parser.add_argument('--logs-dir', type=str, default='./logs/')
    parser.add_argument('--aug-type', type=str, default='color_crop_cutout')
    parser.add_argument('--mixup-net', type=str, default='cut')
    parser.add_argument('--bias', type=str2bool, default=False)
    parser.add_argument('--fc', type=str2bool, default=False)
    parser.add_argument('--mix-p', type=float, default=-1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--tag', type=str, default='test')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_dir = args.output_dir + args.tag
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.output_dir + '/outputs'):
        os.makedirs(args.output_dir + '/outputs')

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    args.logs_dir = args.logs_dir + args.tag
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    sys.stdout = Logger(os.path.join(args.logs_dir, 'logs.txt'))

    print(args)

    trainloader, testloader = load_data(args)

    generator = Generator(args).cuda()
    discriminator = Discriminator(args).cuda()

    optim_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.9))
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.9))
    criterion = nn.CrossEntropyLoss()

    aug, aug_rand = diffaug(args)

    best_top1s = np.zeros((len(args.eval_model),))
    best_top5s = np.zeros((len(args.eval_model),))
    best_epochs = np.zeros((len(args.eval_model),))
    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        train(args, epoch, generator, discriminator, optim_g, optim_d, trainloader, criterion, aug, aug_rand)

        # save image for visualization
        generator.eval()
        test_label = torch.tensor(list(range(10)) * 10)
        test_noise = torch.normal(0, 1, (100, 100))
        lab_onehot = torch.zeros((100, args.num_classes))
        lab_onehot[torch.arange(100), test_label] = 1
        test_noise[torch.arange(100), :args.num_classes] = lab_onehot[torch.arange(100)]
        test_noise = test_noise.cuda()
        test_img_syn = (generator(test_noise) + 1.0) / 2.0
        test_img_syn = make_grid(test_img_syn, nrow=10)
        save_image(test_img_syn, os.path.join(args.output_dir, 'outputs/img_{}.png'.format(epoch)))
        generator.train()

        if (epoch + 1) % args.eval_interval == 0:
            model_dict = {'generator': generator.state_dict(),
                          'discriminator': discriminator.state_dict(),
                          'optim_g': optim_g.state_dict(),
                          'optim_d': optim_d.state_dict()}
            torch.save(
                model_dict,
                os.path.join(args.output_dir, 'model_dict_{}.pth'.format(epoch)))
            print("img and data saved!")

            top1s, top5s = validate(args, generator, testloader, criterion, aug_rand)
            for e_idx, e_model in enumerate(args.eval_model):
                if top1s[e_idx] > best_top1s[e_idx]:
                    best_top1s[e_idx] = top1s[e_idx]
                    best_top5s[e_idx] = top5s[e_idx]
                    best_epochs[e_idx] = epoch
                print('Current Best Epoch for {}: {}, Top1: {:.3f}, Top5: {:.3f}'.format(e_model, best_epochs[e_idx], best_top1s[e_idx], best_top5s[e_idx]))

