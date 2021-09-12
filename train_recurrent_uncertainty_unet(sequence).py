import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import AverageMeter, save_checkpoint, load_checkpoint_model, get_logger, parse_args, show_uncertainty_result, \
    choose_best_checkpoint
from hedge_transform import HedgeTransform
from VGGClass import VGGClass
from model import UncertaintyUNet
from dataset import DynamicHedgedDataset, RecurrentStaticHedgedDataset
from loss_functions import vgg_accuracy, uncertainty_MSE_loss
from torch.utils.tensorboard import SummaryWriter
from config import device, print_freq
from optimizer_wrapper import OptimizerWrapper
import os


def train_net(args):
    checkpoint = args.recurrent_uncertainty_checkpoint
    best_loss = float('inf')
    writer = SummaryWriter()

    in_channel = 3 * 3  # new input + previous (pred + uncertainty)
    out_channel = 3
    out_uncertainty_channel = 3
    criterion = uncertainty_MSE_loss

    # init / check point
    if checkpoint is None:
        model = UncertaintyUNet(in_channel, out_channel, out_uncertainty_channel).to(device)
        start_epoch = 0
        epochs_since_improvement = 0
        if args.optimizer == 'adam':
            optimizer = OptimizerWrapper(torch.optim.Adam(model.parameters(), lr=args.lr))
        else:
            raise TypeError('optimizer {} is not supported.'.format(args.network))

    else:
        checkpoint = torch.load(checkpoint)
        model = checkpoint['u_net_model']
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epoch_since_improvement']
        optimizer = checkpoint['optimizer']
        # change lr if needed
        if args.change_lr:
            print("adjust lr to ", args.lr)
            optimizer.adjust_lr(args.lr)

    # density range
    density_range = np.arange(args.min_density, args.max_density, 0.1)

    # # classification model
    vgg_model = VGGClass().to(device)
    vgg_model.eval()

    # single_uncertainty model for first frame
    best_model = choose_best_checkpoint(args.pretrained_uncertainty_u_net)
    single_unet_model = load_checkpoint_model(best_model).to(device)
    single_unet_model.eval()

    # dataset/dataloader
    if args.dataset == 'static dataset':
        # we use batch size as sample per image, so each batch would all show same background with different hedge masks
        train_dataset = RecurrentStaticHedgedDataset(os.path.join('..', 'data', 'synthetic_data', 'train'),
                                                     density_range, args.sample_per_image, args.batch_size)
        val_dataset = RecurrentStaticHedgedDataset(os.path.join('..', 'data', 'synthetic_data', 'val'), density_range,
                                                   args.sample_per_image, args.batch_size)
    elif args.dataset == 'dynamic dataset':
        # TODO depreciated
        transform = HedgeTransform(density_range, args.batch_size, args.image_size,
                                   hedge_path=os.path.join('..', 'data', 'Hedge_masks'))
        train_dataset = DynamicHedgedDataset(os.path.join('..', 'data', 'imagenet_data', 'train'), transform)
        val_dataset = DynamicHedgedDataset(os.path.join('..', 'data', 'imagenet_data', 'val'), transform)
    else:
        raise TypeError("Dataset {} is not supported".format(args.dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    logger = get_logger()

    for ep in range(start_epoch, args.end_epoch):
        # train
        train_loss, train_mse, train_uncertainty = train(train_loader, model, single_unet_model, criterion, optimizer,
                                                         args.clip_val, args.sample_per_image, ep,
                                                         logger)

        # val
        val_loss, val_mse, val_uncertainty, top_5_acc, top_1_acc, top_5_acc_gt, top_1_acc_gt = val(val_loader, model,
                                                                                                   single_unet_model,
                                                                                                   vgg_model,
                                                                                                   criterion, args.sample_per_image,
                                                                                                   ep, logger)

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

        writer.add_scalar('train/train_loss', train_loss, ep)
        writer.add_scalar('train/uncertainty', train_uncertainty, ep)
        writer.add_scalar('train/mse', train_mse, ep)
        writer.add_scalar('val/val_loss', val_loss, ep)
        writer.add_scalar('val/uncertainty', val_uncertainty, ep)
        writer.add_scalar('val/mse', val_mse, ep)
        writer.add_scalar('top1/top_1_acc', top_1_acc, ep)
        writer.add_scalar('top1/top_1_acc_gt', top_1_acc_gt, ep)
        writer.add_scalar('top5/top_5_acc', top_5_acc, ep)
        writer.add_scalar('top5/top_5_acc_gt', top_5_acc_gt, ep)

        # due to the uncertainty property there are chance that the loss go up to inf or nan
        # so we have to treat them differently
        if math.isnan(val_loss) or math.isinf(val_loss):
            is_best = False
        else:
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: {}".format(epochs_since_improvement))
        else:
            epochs_since_improvement = 0

        save_checkpoint(ep, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(train_loader, model, single_unet_model, criterion, optimizer, clip_val, sample_per_image, epoch, logger):
    # train mode
    model.train()

    # loss value init
    training_loss = AverageMeter()
    mse_meter = AverageMeter()
    uncertainty_meter = AverageMeter()

    # to suppress IDE warning
    pred = None
    uncertainty = None

    for i, (masked_images, ground_truths, _) in enumerate(train_loader):
        ground_truths = ground_truths.to(device)
        masked_images = masked_images.to(device)
        # for every sample_per_image batch, the input would have a new background
        # we then have to process it with single u net model to get the initial pred + uncertainty
        # otherwise the pred and uncertainty from the previous batch would be used
        if i % sample_per_image == 0:
            pred, uncertainty = single_unet_model(masked_images)

        # construct input (new input + previous pred + previous uncertainty)
        # we detach it as pred and uncertainty should be removed from graph
        model_input = torch.cat([masked_images, pred, uncertainty], dim=1).detach()
        pred, uncertainty = model(model_input)
        loss, mse = criterion(pred, uncertainty, ground_truths)
        # back prop
        optimizer.zero_grad()
        loss.backward()
        # update weights
        if clip_val:
            optimizer.clip_gradient(clip_val)
        optimizer.step()
        training_loss.update(loss.item())
        mse_meter.update(mse.mean().item())
        uncertainty_meter.update(torch.mean(uncertainty))

        # log.txt info
        if i % print_freq == 0:
            show_uncertainty_result(masked_images[0], pred[0], ground_truths[0], uncertainty[0])
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Train Loss {loss.val:.5f} ({loss.avg:.5f})\t'
                        'Uncertainty = {uncertainty.avg:.5f}\t'
                        'MSE = {mse.avg:.5f}\t'.format(epoch, i, len(train_loader),
                                                       loss=training_loss,
                                                       uncertainty=uncertainty_meter,
                                                       mse=mse_meter))

    return training_loss.avg, mse_meter.avg, uncertainty_meter.avg


def val(val_loader, model, single_unet_model, vgg_model, criterion, sample_per_image, epoch, logger):
    # eval mode
    model.eval()

    # loss values init
    validation_loss = AverageMeter()
    top_1_acc = AverageMeter()
    top_5_acc = AverageMeter()
    top_1_acc_gt = AverageMeter()
    top_5_acc_gt = AverageMeter()
    mse_meter = AverageMeter()
    uncertainty_meter = AverageMeter()

    with torch.no_grad():
        for i, (masked_images, ground_truths, label) in enumerate(val_loader):
            ground_truths = ground_truths.to(device)
            masked_images = masked_images.to(device)
            label = label.to(device)
            # for every sample_per_image batch, the input would have a new background
            # we then have to process it with single u net model to get the initial pred + uncertainty
            # otherwise the pred and uncertainty from the previous batch would be used
            if i % sample_per_image == 0:
                pred, uncertainty = single_unet_model(masked_images)

            # construct input (new input + previous pred + previous uncertainty)
            # we detach it as pred and uncertainty should be removed from graph
            model_input = torch.cat([masked_images, pred, uncertainty], dim=1).detach()
            pred, uncertainty = model(model_input)
            loss, mse = criterion(pred, uncertainty, ground_truths)

            validation_loss.update(loss.item())
            mse_meter.update(mse.mean().item())
            uncertainty_meter.update(torch.mean(uncertainty))

            # update vgg top1/5 accuracy loss
            top_1, top_5 = vgg_accuracy(vgg_model, pred, label)
            top_1_gt, top_5_gt = vgg_accuracy(vgg_model, ground_truths, label)

            top_5_acc.update(top_5)
            top_1_acc.update(top_1)

            top_5_acc_gt.update(top_5_gt)
            top_1_acc_gt.update(top_1_gt)

            # log.txt info
            if i % print_freq == 0:
                show_uncertainty_result(masked_images[0], pred[0], ground_truths[0], uncertainty[0])

                logger.info('Epoch: [{0}][{1}/{2}]\t'
                            'Val Loss {loss.val:.5f} ({loss.avg:.5f})\t'
                            'Uncertainty {uncertainty.avg:.5f}\t'
                            'MSE {mse.avg:.5f}\t'
                            'top1 {top_1.val:.5f} ({top_1.avg:.5f})\t'
                            'top5 {top_5.val:.5f} ({top_5.avg:.5f})\t'
                            'top1 (gt) {top_1_gt.val:.5f} ({top_1_gt.avg:.5f})\t'
                            'top5 (gt) {top_5_gt.val:.5f} ({top_5_gt.avg:.5f})\t'.format(epoch, i,
                                                                                         len(val_loader),
                                                                                         loss=validation_loss,
                                                                                         uncertainty=uncertainty_meter,
                                                                                         mse=mse_meter,
                                                                                         top_1=top_1_acc,
                                                                                         top_5=top_5_acc,
                                                                                         top_1_gt=top_1_acc_gt,
                                                                                         top_5_gt=top_5_acc_gt))

        return validation_loss.avg, mse_meter.avg, uncertainty_meter.avg, top_5_acc.avg, top_1_acc.avg, top_5_acc_gt.avg, top_1_acc_gt.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
