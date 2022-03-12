import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import AverageMeter, save_checkpoint, get_logger, parse_args, show_tensor_images
from hedge_transform import HedgeTransform
from VGGClass import VGGClass
# from unet import UNet
from model import UNet
from dataset import DynamicHedgedDataset, StaticHedgedDataset
from loss_functions import vgg_accuracy
from torch.utils.tensorboard import SummaryWriter
from config import device, print_freq
from optimizer_wrapper import OptimizerWrapper
import os


def train_net(args):
    checkpoint = args.unet_checkpoint
    best_loss = float('inf')
    writer = SummaryWriter()

    in_channel = 3
    out_channel = 3
    model = UNet(in_channel, out_channel).to(device)
    criterion = nn.MSELoss().to(device)

    # init / check point
    if checkpoint is None:
        start_epoch = 0
        if args.optimizer == 'adam':
            optimizer = OptimizerWrapper(torch.optim.Adam(model.parameters(), lr=args.lr))
        else:
            raise TypeError('optimizer {} is not supported.'.format(args.network))
        epochs_since_improvement = 0
    else:
        checkpoint = torch.load(checkpoint)
        model = checkpoint['u_net_model'].to(device)
        start_epoch = checkpoint['epoch'] + 1
        optimizer = checkpoint['optimizer']
        epochs_since_improvement = checkpoint['epoch_since_improvement']
        if args.change_lr:
            print("adjust lr to ", args.lr)
            optimizer.adjust_lr(args.lr)

    # density range
    density_range = np.arange(args.min_density, args.max_density, 0.1)

    # classification model
    vgg_model = VGGClass().to(device)
    vgg_model.eval()

    train_dataset = StaticHedgedDataset(os.path.join('..', 'data', 'synthetic_data', 'train'), density_range)
    val_dataset = StaticHedgedDataset(os.path.join('..', 'data', 'synthetic_data', 'val'), density_range)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    logger = get_logger()

    for ep in range(start_epoch, args.end_epoch):

        if ep == 399:
            print("Epoch {}: adjust lr to {}".format(ep+1, 3e-3))
            optimizer.adjust_lr(3e-3)

        # train
        train_loss = train(train_loader, model, criterion, optimizer, args.clip_val, ep, logger)

        # val
        val_loss, top_5_acc, top_1_acc, top_5_acc_gt, top_1_acc_gt = val(val_loader, model, vgg_model,
                                                                         criterion, ep, logger)

        logger.info('Epoch: [{0}]\t'
                    'Train Loss {train_loss:.5f}\t'
                    'Val Loss {val_loss:.5f}\t'
                    'top-1 accuracy {top1:.5f}\t'
                    'top-5 accuracy {top5:.5f}\t'
                    'top-1 accuracy (gt) {top1gt:.5f}\t'
                    'top-5 accuracy (gt) {top5gt:.5f}\t'.format(ep, train_loss=train_loss,
                                                                val_loss=val_loss,
                                                                top1=top_1_acc,
                                                                top5=top_5_acc,
                                                                top1gt=top_1_acc_gt,
                                                                top5gt=top_5_acc_gt))

        writer.add_scalar('model/train_loss', train_loss, ep)
        writer.add_scalar('model/val_loss', val_loss, ep)
        writer.add_scalar('top1/top_1_acc', top_1_acc, ep)
        writer.add_scalar('top1/top_1_acc_gt', top_1_acc_gt, ep)
        writer.add_scalar('top5/top_5_acc', top_5_acc, ep)
        writer.add_scalar('top5/top_5_acc_gt', top_5_acc_gt, ep)

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: {}".format(epochs_since_improvement))
        else:
            epochs_since_improvement = 0

        save_checkpoint(ep, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(train_loader, model, criterion, optimizer, clip_val, epoch, logger):
    # train mode
    model.train()

    # loss value init
    training_loss = AverageMeter()

    for i, (masked_images, ground_truths, _) in enumerate(train_loader):
        ground_truths = ground_truths.to(device)
        masked_images = masked_images.to(device)

        # L2 loss training
        dehedged_prediction = model(masked_images)
        loss = criterion(dehedged_prediction, ground_truths)
        # back prop
        optimizer.zero_grad()
        loss.backward()
        # update weights
        if clip_val:
            optimizer.clip_gradient(clip_val)
        optimizer.step()

        training_loss.update(loss.item())

        # log.txt info
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Train Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(epoch, i, len(train_loader),
                                                                              loss=training_loss))

    return training_loss.avg


def val(val_loader, u_net_model, vgg_model, criterion, epoch, logger):
    # eval mode
    u_net_model.eval()

    # loss values init
    validation_loss = AverageMeter()
    top_5_acc = AverageMeter()
    top_1_acc = AverageMeter()
    top_5_acc_gt = AverageMeter()
    top_1_acc_gt = AverageMeter()

    with torch.no_grad():
        for i, (masked_images, ground_truths, label) in enumerate(val_loader):
            ground_truths = ground_truths.to(device)
            masked_images = masked_images.to(device)
            label = label.to(device)
            # update L2 loss
            dehedged_predictions = u_net_model(masked_images)
            loss = criterion(dehedged_predictions, ground_truths)
            validation_loss.update(loss.item())

            # update vgg top1/5 accuracy loss
            top_1, top_5 = vgg_accuracy(vgg_model, dehedged_predictions, label)
            top_1_gt, top_5_gt = vgg_accuracy(vgg_model, ground_truths, label)

            top_5_acc.update(top_5)
            top_1_acc.update(top_1)

            top_5_acc_gt.update(top_5_gt)
            top_1_acc_gt.update(top_1_gt)

            # log.txt info
            if i % print_freq == 0:
                show_tensor_images([masked_images[0], dehedged_predictions[0], ground_truths[0]], ["masked image", "pred", "ground truth"])

                logger.info('Epoch: [{0}][{1}/{2}]\t'
                            'Val Loss {loss.val:.5f} ({loss.avg:.5f})\t'
                            'top-1 accuracy {top1.val:.5f} ({top1.avg:.5f})\t'
                            'top-5 accuracy {top5.val:.5f} ({top5.avg:.5f})\t'
                            'top-1 accuracy (gt) {top1gt.val:.5f} ({top1gt.avg:.5f})\t'
                            'top-5 accuracy (gt) {top5gt.val:.5f} ({top5gt.avg:.5f})\t'.format(epoch, i,
                                                                                               len(val_loader),
                                                                                               loss=validation_loss,
                                                                                               top1=top_1_acc,
                                                                                               top5=top_5_acc,
                                                                                               top1gt=top_1_acc_gt,
                                                                                               top5gt=top_5_acc_gt))

        return validation_loss.avg, top_5_acc.avg, top_1_acc.avg, top_5_acc_gt.avg, top_1_acc_gt.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
