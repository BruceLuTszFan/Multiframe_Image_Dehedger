import numpy as np

from utils import AverageMeter, show_uncertainty_result, plot_graph, show_tensor_images
from config import device, print_freq
from loss_functions import vgg_accuracy

import os
import torch


def generate_recurrent_results(test_loader, u_net_model, preprocess_model, density, root):
    # eval mode
    u_net_model.eval()
    preprocess_model.eval()

    with torch.no_grad():
        for i, (masked_images, masked_images2, masked_images3, masked_images4, masked_images5, ground_truths,
                label) in enumerate(test_loader):
            ground_truths = ground_truths.to(device)
            masked_images = masked_images.to(device)
            masked_images2 = masked_images2.to(device)
            masked_images3 = masked_images3.to(device)
            masked_images4 = masked_images4.to(device)
            masked_images5 = masked_images5.to(device)

            # we produce prediction + uncertainty for first frame
            prev_pred, prev_uncertainty = preprocess_model(masked_images)

            # then construct input (new input + previous pred + previous uncertainty)
            # we detach it as pred and uncertainty should be removed from graph
            model_input = torch.cat([masked_images2, prev_pred, prev_uncertainty], dim=1).detach()
            prev_pred2, prev_uncertainty2 = u_net_model(model_input)

            model_input = torch.cat([masked_images3, prev_pred2, prev_uncertainty2], dim=1).detach()
            prev_pred3, prev_uncertainty3 = u_net_model(model_input)

            model_input = torch.cat([masked_images4, prev_pred3, prev_uncertainty3], dim=1).detach()
            prev_pred4, prev_uncertainty4 = u_net_model(model_input)

            model_input = torch.cat([masked_images5, prev_pred4, prev_uncertainty4], dim=1).detach()
            pred, uncertainty = u_net_model(model_input)

            # log.txt info
            if i % print_freq == 0:
                path = os.path.join(root, str(round(density, 1)))
                os.makedirs(path, exist_ok=True)
                path1 = os.path.join(path, "fig" + str(i) + "prev1.png")
                path2 = os.path.join(path, "fig" + str(i) + "prev2.png")
                path3 = os.path.join(path, "fig" + str(i) + "prev3.png")
                path4 = os.path.join(path, "fig" + str(i) + "prev4.png")
                path5 = os.path.join(path, "fig" + str(i) + ".png")

                show_tensor_images([masked_images[0], prev_pred[0], ground_truths[0], prev_uncertainty[0]],
                                   ["masked image", "prediction", "ground truth", "uncertainty map"], path1)
                show_tensor_images([masked_images2[0], prev_pred2[0], ground_truths[0], prev_uncertainty2[0]],
                                   ["masked image", "prediction", "ground truth", "uncertainty map"], path2)
                show_tensor_images([masked_images3[0], prev_pred3[0], ground_truths[0], prev_uncertainty3[0]],
                                   ["masked image", "prediction", "ground truth", "uncertainty map"], path3)
                show_tensor_images([masked_images4[0], prev_pred4[0], ground_truths[0], prev_uncertainty4[0]],
                                   ["masked image", "prediction", "ground truth", "uncertainty map"], path4)
                show_tensor_images([masked_images5[0], pred[0], ground_truths[0], uncertainty[0]],
                                   ["masked image", "prediction", "ground truth", "uncertainty map"], path5)


def test_recurrent_uncertainty(test_loader, u_net_model, preprocess_model, vgg_model, criterion, density,
                               logger, root):
    # eval mode
    u_net_model.eval()
    preprocess_model.eval()

    # loss values init
    test_loss_1_frame = AverageMeter()
    test_loss_2_frames = AverageMeter()
    test_loss_3_frames = AverageMeter()
    test_loss_5_frames = AverageMeter()

    top_1_acc_1_frame = AverageMeter()
    top_1_acc_2_frames = AverageMeter()
    top_1_acc_3_frames = AverageMeter()
    top_1_acc_5_frames = AverageMeter()

    top_5_acc_1_frame = AverageMeter()
    top_5_acc_2_frames = AverageMeter()
    top_5_acc_3_frames = AverageMeter()
    top_5_acc_5_frames = AverageMeter()

    top_1_acc_gt = AverageMeter()
    top_5_acc_gt = AverageMeter()

    mse_meter_1_frame = AverageMeter()
    mse_meter_2_frames = AverageMeter()
    mse_meter_3_frames = AverageMeter()
    mse_meter_5_frames = AverageMeter()

    uncertainty_meter_1_frame = AverageMeter()
    uncertainty_meter_2_frames = AverageMeter()
    uncertainty_meter_3_frames = AverageMeter()
    uncertainty_meter_5_frames = AverageMeter()

    top_1_acc_msk = AverageMeter()
    top_5_acc_msk = AverageMeter()

    with torch.no_grad():
        for i, (masked_images, masked_images2, masked_images3, masked_images4, masked_images5, ground_truths,
                label) in enumerate(test_loader):
            ground_truths = ground_truths.to(device)
            masked_images = masked_images.to(device)
            masked_images2 = masked_images2.to(device)
            masked_images3 = masked_images3.to(device)
            masked_images4 = masked_images4.to(device)
            masked_images5 = masked_images5.to(device)
            label = label.to(device)

            # we produce prediction + uncertainty for first frame
            prev_pred, prev_uncertainty = preprocess_model(masked_images)
            loss_1_frame, mse_1_frame = criterion(prev_pred, prev_uncertainty, ground_truths)

            # then construct input (new input + previous pred + previous uncertainty)
            # we detach it as pred and uncertainty should be removed from graph
            model_input = torch.cat([masked_images2, prev_pred, prev_uncertainty], dim=1).detach()
            prev_pred2, prev_uncertainty2 = u_net_model(model_input)
            loss_2_frames, mse_2_frames = criterion(prev_pred2, prev_uncertainty2, ground_truths)

            model_input = torch.cat([masked_images3, prev_pred2, prev_uncertainty2], dim=1).detach()
            prev_pred3, prev_uncertainty3 = u_net_model(model_input)
            loss_3_frames, mse_3_frames = criterion(prev_pred3, prev_uncertainty3, ground_truths)

            model_input = torch.cat([masked_images4, prev_pred3, prev_uncertainty3], dim=1).detach()
            prev_pred4, prev_uncertainty4 = u_net_model(model_input)

            model_input = torch.cat([masked_images5, prev_pred4, prev_uncertainty4], dim=1).detach()
            pred, uncertainty = u_net_model(model_input)
            loss, mse = criterion(pred, uncertainty, ground_truths)

            test_loss_1_frame.update(loss_1_frame.item())
            test_loss_2_frames.update(loss_2_frames.item())
            test_loss_3_frames.update(loss_3_frames.item())
            test_loss_5_frames.update(loss.item())

            mse_meter_1_frame.update(mse_1_frame.mean().item())
            mse_meter_2_frames.update(mse_2_frames.mean().item())
            mse_meter_3_frames.update(mse_3_frames.mean().item())
            mse_meter_5_frames.update(mse.mean().item())

            uncertainty_meter_1_frame.update(torch.mean(prev_uncertainty))
            uncertainty_meter_2_frames.update(torch.mean(prev_uncertainty2))
            uncertainty_meter_3_frames.update(torch.mean(prev_uncertainty3))
            uncertainty_meter_5_frames.update(torch.mean(uncertainty))

            # update vgg top1/5 accuracy loss
            top_1, top_5 = vgg_accuracy(vgg_model, pred, label)
            top_1_1_frame, top_5_1_frame = vgg_accuracy(vgg_model, prev_pred, label)
            top_1_2_frames, top_5_2_frames = vgg_accuracy(vgg_model, prev_pred2, label)
            top_1_3_frames, top_5_3_frames = vgg_accuracy(vgg_model, prev_pred3, label)

            top_1_gt, top_5_gt = vgg_accuracy(vgg_model, ground_truths, label)
            top_1_msk, top_5_msk = vgg_accuracy(vgg_model, masked_images, label)

            top_5_acc_1_frame.update(top_5_1_frame)
            top_5_acc_2_frames.update(top_5_2_frames)
            top_5_acc_3_frames.update(top_5_3_frames)
            top_5_acc_5_frames.update(top_5)

            top_1_acc_1_frame.update(top_1_1_frame)
            top_1_acc_5_frames.update(top_1)
            top_1_acc_2_frames.update(top_1_2_frames)
            top_1_acc_3_frames.update(top_1_3_frames)

            top_5_acc_gt.update(top_5_gt)
            top_1_acc_gt.update(top_1_gt)

            top_1_acc_msk.update(top_1_msk)
            top_5_acc_msk.update(top_5_msk)

            # log.txt info
            if i % print_freq == 0:
                path = os.path.join(root, str(round(density, 1)))
                os.makedirs(path, exist_ok=True)
                path1 = os.path.join(path, "fig" + str(i) + "prev1.png")
                path2 = os.path.join(path, "fig" + str(i) + "prev2.png")
                path3 = os.path.join(path, "fig" + str(i) + "prev3.png")
                path4 = os.path.join(path, "fig" + str(i) + "prev4.png")
                path5 = os.path.join(path, "fig" + str(i) + ".png")

                show_uncertainty_result(masked_images[0], prev_pred[0], ground_truths[0], prev_uncertainty[0],
                                        path1)
                show_uncertainty_result(masked_images2[0], prev_pred2[0], ground_truths[0], prev_uncertainty2[0],
                                        path2)
                show_uncertainty_result(masked_images3[0], prev_pred3[0], ground_truths[0], prev_uncertainty3[0],
                                        path3)
                show_uncertainty_result(masked_images4[0], prev_pred4[0], ground_truths[0], prev_uncertainty4[0],
                                        path4)
                show_uncertainty_result(masked_images5[0], pred[0], ground_truths[0], uncertainty[0], path5)

                logger.info('Density: [{0:.1f}][{1}/{2}]\t'
                            'Test Loss {loss.val:.5f} ({loss.avg:.5f})\t'
                            'Uncertainty {uncertainty.avg:.5f}\t'
                            'MSE {mse.avg:.5f}\t'
                            'top1 {top_1.val:.5f} ({top_1.avg:.5f})\t'
                            'top5 {top_5.val:.5f} ({top_5.avg:.5f})\t'
                            'top1 (gt) {top_1_gt.val:.5f} ({top_1_gt.avg:.5f})\t'
                            'top5 (gt) {top_5_gt.val:.5f} ({top_5_gt.avg:.5f})\t'.format(density, i,
                                                                                         len(test_loader),
                                                                                         loss=test_loss_5_frames,
                                                                                         uncertainty=uncertainty_meter_5_frames,
                                                                                         mse=mse_meter_5_frames,
                                                                                         top_1=top_1_acc_5_frames,
                                                                                         top_5=top_5_acc_5_frames,
                                                                                         top_1_gt=top_1_acc_gt,
                                                                                         top_5_gt=top_5_acc_gt))

        result_1_frame = [test_loss_1_frame.avg, top_5_acc_1_frame.avg, top_1_acc_1_frame.avg,
                          top_5_acc_gt.avg, top_1_acc_gt.avg, top_5_acc_msk.avg, top_1_acc_msk.avg,
                          mse_meter_1_frame.avg, uncertainty_meter_1_frame.avg]
        result_2_frames = [test_loss_2_frames.avg, top_5_acc_2_frames.avg, top_1_acc_2_frames.avg,
                           top_5_acc_gt.avg, top_1_acc_gt.avg, top_5_acc_msk.avg, top_1_acc_msk.avg,
                           mse_meter_2_frames.avg, uncertainty_meter_2_frames.avg]
        result_3_frames = [test_loss_3_frames.avg, top_5_acc_3_frames.avg, top_1_acc_3_frames.avg,
                           top_5_acc_gt.avg, top_1_acc_gt.avg, top_5_acc_msk.avg, top_1_acc_msk.avg,
                           mse_meter_3_frames.avg, uncertainty_meter_3_frames.avg]
        result_5_frames = [test_loss_5_frames.avg, top_5_acc_5_frames.avg, top_1_acc_5_frames.avg,
                           top_5_acc_gt.avg, top_1_acc_gt.avg, top_5_acc_msk.avg, top_1_acc_msk.avg,
                           mse_meter_5_frames.avg, uncertainty_meter_5_frames.avg]

        return result_1_frame, result_2_frames, result_3_frames, result_5_frames


def generate_uncertainty_evaluation_report(result_by_density, path):
    results = [[] for _ in range(len(next(iter(result_by_density.items()))[1]))]
    dict_items = list(result_by_density.items())
    densities = result_by_density.keys()

    for i in range(len(result_by_density)):
        density, result = dict_items[i]
        for j in range(len(results)):
            results[j].append(result[j])

    # plot loss graph [0]
    plot_graph("uncertainty loss", "density", "loss", [densities], [results[0]], [""],
               os.path.join(path, "uncertainty_loss.png"))
    np.save(os.path.join(path, "uncertainty_loss.npy"), np.array(results[0]))

    # plot acc vs gt vs mask top 5 graph [1] + [3] + [5]
    plot_graph("top-5 vgg accuracy", "density", "accuracy", [densities] * 3, [results[1], results[3], results[5]],
               ["model acc", "ground truth acc", "masked acc"], os.path.join(path, "top_5_vgg_acc.png"))
    np.save(os.path.join(path, "top_5_model_acc.npy"), np.array(results[1]))
    np.save(os.path.join(path, "top_5_ground_truth_acc.npy"), np.array(results[3]))
    np.save(os.path.join(path, "top_5_masked_acc.npy"), np.array(results[5]))

    # plot acc vs gt vs mask top 1 graph [2] + [4] + [6]
    plot_graph("top-1 vgg accuracy", "density", "accuracy", [densities] * 3, [results[2], results[4], results[6]],
               ["model acc", "ground truth acc", "masked acc"], os.path.join(path, "top_1_vgg_acc.png"))
    np.save(os.path.join(path, "top_1_model_acc.npy"), np.array(results[2]))
    np.save(os.path.join(path, "top_1_ground_truth_acc.npy"), np.array(results[4]))
    np.save(os.path.join(path, "top_1_masked_acc.npy"), np.array(results[6]))

    # plot mse graph [7]
    plot_graph("MSE loss", "density", "loss", [densities], [results[7]], [""], os.path.join(path, "mse.png"))
    np.save(os.path.join(path, "mse_loss.npy"), np.array(results[7]))

    # plot uncertainty graph [8]
    plot_graph("Uncertainty val", "density", "uncertainty", [densities], [results[8]], [""],
               os.path.join(path, "uncertainty.png"))
    np.save(os.path.join(path, "uncertainty.npy"), np.array(results[8]))

    return results
