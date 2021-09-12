from utils import AverageMeter, show_tensor_images, plot_graph
from config import device, print_freq
from loss_functions import vgg_accuracy

import os
import torch
import numpy as np

def test_MSE(test_loader, u_net_model, vgg_model, criterion, density, logger, root):
    # eval mode
    u_net_model.eval()

    # loss values init
    test_loss = AverageMeter()
    top_1_acc = AverageMeter()
    top_5_acc = AverageMeter()
    top_1_acc_gt = AverageMeter()
    top_5_acc_gt = AverageMeter()
    top_1_acc_msk = AverageMeter()
    top_5_acc_msk = AverageMeter()

    with torch.no_grad():
        for i, (masked_images, ground_truths, label) in enumerate(test_loader):
            ground_truths = ground_truths.to(device)
            masked_images = masked_images.to(device)
            label = label.to(device)
            # update L2 loss
            pred = u_net_model(masked_images)
            loss = criterion(pred, ground_truths)

            test_loss.update(loss.item())


            # update vgg top1/5 accuracy loss
            top_1, top_5 = vgg_accuracy(vgg_model, pred, label)
            top_1_gt, top_5_gt = vgg_accuracy(vgg_model, ground_truths, label)
            top_1_msk, top_5_msk = vgg_accuracy(vgg_model, masked_images, label)

            top_5_acc.update(top_5)
            top_1_acc.update(top_1)

            top_5_acc_gt.update(top_5_gt)
            top_1_acc_gt.update(top_1_gt)

            top_1_acc_msk.update(top_1_msk)
            top_5_acc_msk.update(top_5_msk)

            # log.txt info
            if i % print_freq == 0:
                path = os.path.join(root, str(round(density, 1)))
                os.makedirs(path, exist_ok=True)
                path = os.path.join(path, "fig"+str(i)+".png")

                show_tensor_images([masked_images[0], pred[0], ground_truths[0]], ["masked image", "prediction", "ground truth"], path)

                logger.info('Density: [{0:.1f}][{1}/{2}]\t'
                            'MSE Loss {loss.val:.5f} ({loss.avg:.5f})\t'
                            'top1 {top_1.val:.5f} ({top_1.avg:.5f})\t'
                            'top5 {top_5.val:.5f} ({top_5.avg:.5f})\t'
                            'top1 (gt) {top_1_gt.val:.5f} ({top_1_gt.avg:.5f})\t'
                            'top5 (gt) {top_5_gt.val:.5f} ({top_5_gt.avg:.5f})\t'.format(density, i,
                                                                                         len(test_loader),
                                                                                         loss=test_loss,
                                                                                         top_1=top_1_acc,
                                                                                         top_5=top_5_acc,
                                                                                         top_1_gt=top_1_acc_gt,
                                                                                         top_5_gt=top_5_acc_gt))

        return test_loss.avg, top_5_acc.avg, top_1_acc.avg, top_5_acc_gt.avg, top_1_acc_gt.avg, top_5_acc_msk.avg, top_1_acc_msk.avg


def generate_MSE_evaluation_report(result_by_density, path):
    results = [[] for _ in range(len(next(iter(result_by_density.items()))[1]))]
    dict_items = list(result_by_density.items())
    densities = result_by_density.keys()

    for i in range(len(result_by_density)):
        density, result = dict_items[i]
        for j in range(len(results)):
            results[j].append(result[j])

    # plot loss graph [0]
    plot_graph("MSE loss", "density", "loss", [densities], [results[0]], [""], os.path.join(path, "mse_loss.png"))
    np.save(os.path.join(path, "mse_loss.npy"), np.array(results[0]))

    # plot acc vs gt vs mask top 5 graph [1] + [3] + [5]
    plot_graph("top-5 vgg accuracy", "density", "accuracy", [densities]*3, [results[1], results[3], results[5]], ["model acc", "ground truth acc", "masked acc"], os.path.join(path, "top_5_vgg_acc.png"))
    np.save(os.path.join(path, "top_5_model_acc.npy"), np.array(results[1]))
    np.save(os.path.join(path, "top_5_ground_truth_acc.npy"), np.array(results[3]))
    np.save(os.path.join(path, "top_5_masked_acc.npy"), np.array(results[5]))

    # plot acc vs gt vs mask top 1 graph [2] + [4] + [6]
    plot_graph("top-1 vgg accuracy", "density", "accuracy", [densities]*3, [results[2], results[4], results[6]], ["model acc", "ground truth acc", "masked acc"], os.path.join(path, "top_1_vgg_acc.png"))
    np.save(os.path.join(path, "top_1_model_acc.npy"), np.array(results[2]))
    np.save(os.path.join(path, "top_1_ground_truth_acc.npy"), np.array(results[4]))
    np.save(os.path.join(path, "top_1_masked_acc.npy"), np.array(results[6]))
    return results