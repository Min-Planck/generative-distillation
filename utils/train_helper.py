import torch

def calc_gradient_penalty(args, discriminator, img_real, img_syn):
    ''' Gradient penalty from Wasserstein GAN
    '''
    LAMBDA = 10
    n_size = img_real.shape[-1]
    batch_size = img_real.shape[0]
    n_channels = img_real.shape[1]

    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(img_real.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, n_channels, n_size, n_size)
    alpha = alpha.cuda()

    img_syn = img_syn.view(batch_size, n_channels, n_size, n_size)
    interpolates = alpha * img_real.detach() + ((1 - alpha) * img_syn.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = discriminator(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def test(args, model, testloader, criterion):
    '''Calculate accuracy
    '''
    from utils import AverageMeter, accuracy    
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (img, lab) in enumerate(testloader):
        img = img.cuda()
        lab = lab.cuda()

        with torch.no_grad():
            output = model(img)
        loss = criterion(output, lab)
        acc1, acc5 = accuracy(output.data, lab, topk=(1, 5))
        losses.update(loss.item(), output.shape[0])
        top1.update(acc1.item(), output.shape[0])
        top5.update(acc5.item(), output.shape[0])

    return top1.avg, top5.avg, losses.avg
