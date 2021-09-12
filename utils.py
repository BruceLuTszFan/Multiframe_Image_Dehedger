import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from random import getrandbits
import torch
import logging
import urllib.error
import urllib3.exceptions


def parse_args():
    parser = argparse.ArgumentParser(description='Train u-net')
    # pretrained
    parser.add_argument('--pretrained-MSE-unet', type=str,
                        default="records/actual_results/0_MSE_UNet/new_model/complex/checkpoint/ep250(lr=3e-3)/BEST_checkpoint.tar",
                        help='pretrained single frame MSE unet')
    parser.add_argument('--pretrained-uncertainty-u-net', type=str,
                        default="records/actual_results/1_Uncertainty_UNet/new_model/single_frame_clamp_uncertainty_network/checkpoints/ep400(lr=3e-3)/BEST_checkpoint.tar",
                        help='pretrained uncertainty models for first frame')
    # dataset related
    parser.add_argument('--dataset', type=str, default='static dataset', help='specify dynamic/static dataset')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size in each context')
    parser.add_argument('--num-workers', type=int, default=2, help='number of cpu workers')
    # optimizer related
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--end-epoch', type=int, default=300, help='training epoch size.')
    parser.add_argument('--lr', type=float, default=1e-5, help='start learning rate')
    parser.add_argument('--change-lr', type=bool, default=False, help='checkpoint')
    parser.add_argument('--clip-val', type=float, default=None,
                        help='gradient clip value to prevent gradient explosion')
    # image related
    parser.add_argument('--image-size', type=tuple, default=(128, 128), help='input image size')
    parser.add_argument('--min-density', type=float, default=0.0, help='max value for hedge density')
    parser.add_argument('--max-density', type=float, default=0.9, help='max value for hedge density')
    parser.add_argument('--sample-per-image', type=int, default=32, help='sample per image')
    # checkpoint related
    parser.add_argument('--unet-checkpoint', type=str, default=None, help='1 frame MSE unet checkpoint')
    parser.add_argument('--single-uncertainty-checkpoint', type=str,
                        default=None,
                        help='1 frame uncertainty unet checkpoint')
    parser.add_argument('--recurrent-uncertainty-checkpoint', type=str, default="records/actual_results/2_recurrent_frame_uncertainty_u_net/new_model/checkpoint/ep250(lr=1e-4)/checkpoint.tar",
                        help='recurrent uncertainty unet checkpoint')

    args = parser.parse_args()
    return args


# ------------------------------------ plot/images related ------------------------------------

def plot_graph(title, x_label, y_label, data_batch_x, data_batch_y, legend, path, x_lim=None, y_lim=None):
    plt.figure(figsize=(12, 7))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if x_lim:
        plt.xlim(x_lim)
        plt.ylim(y_lim)
    for data_x, data_y, legend in zip(data_batch_x, data_batch_y, legend):
        plt.plot(data_x, data_y, label=legend)
    plt.legend()
    plt.savefig(path)
    plt.show()



def imshow(img, title="", clip=False, colorbar=False):
    np_img = img.numpy()
    min_val = np.min(np_img)
    max_val = np.max(np_img)

    if clip:
        np_img = np.clip(np_img, 0, 1)
    else:
        if np.ptp(np_img) != 0:
            np_img = (np_img - np.min(np_img)) / np.ptp(np_img)
        else:
            np_img = (np_img - np.min(np_img))

    if np_img.shape[0] == 1 or len(np_img.shape) == 2:
        plt.imshow(np_img[0], cmap="gray", vmin=0, vmax=1)
    else:
        plt.imshow(np.transpose(np_img, (1, 2, 0)), vmin=0, vmax=1)

    plt.title(title)
    if colorbar:
        plt.colorbar()
    plt.text(0, 150, "{} range = [{min_val:.3f}, {max_val:.3f}]".format(title, min_val=min_val, max_val=max_val))


def show_uncertainty_result(masked_image, pred, ground_truth, uncertainty, path=None):
    img_dict = get_img_dict(masked_image, pred, ground_truth, uncertainty)
    plot_img_dicts(img_dict, path)


def get_img_dict(masked_image, pred, ground_truth, uncertainty):
    """ get a dict containing [key=name, val=[np images] ] to show images """
    img_dict = dict()

    # turn argument directly into numpy arrays
    masked_image = masked_image.detach().cpu()
    pred = pred.detach().cpu()
    ground_truth = ground_truth.detach().cpu()
    uncertainty = uncertainty.detach().cpu()
    abs = torch.sqrt((pred - ground_truth) ** 2)

    # form exp uncertainty by getting exp(uncertainty)
    sigma = np.exp(uncertainty / 2)

    # calculated based on loss function
    loss_map = np.exp(-uncertainty) * (pred - ground_truth) ** 2 + uncertainty

    # z-score image = abs/sig and we clip all negative values to 0
    z_score = abs / sigma
    z_score = np.clip(z_score, 0, torch.max(z_score))

    # store results in image dict so we could store the title as key and images as value
    add_images_to_dict(img_dict, masked_image, "masked image", add_rgb=False)
    add_images_to_dict(img_dict, pred, "clipped prediction", add_rgb=False)
    add_images_to_dict(img_dict, ground_truth, "ground truth", add_rgb=False)
    add_images_to_dict(img_dict, loss_map, "loss map", add_rgb=False)
    add_images_to_dict(img_dict, uncertainty, "uncertainty map", add_rgb=True)
    add_images_to_dict(img_dict, sigma, "sigma map", add_rgb=True)
    add_images_to_dict(img_dict, abs, "abs map", add_rgb=True)
    add_images_to_dict(img_dict, z_score, "z score", add_rgb=True)

    return img_dict


def add_images_to_dict(img_dict, image, name, add_rgb=False):
    img_dict[name] = image
    if add_rgb:
        channel = ["(R)", "(G)", "(B)"]
        for i, add_rgb in enumerate(channel):
            img_dict[name + add_rgb] = np.transpose(image, (1, 2, 0))[:, :, i].reshape(1, 128, 128)


def plot_img_dicts(img_dict, path):
    try:
        plt.figure(figsize=(20, 15))
        for i, (name, img) in enumerate(img_dict.items()):
            plt.subplot(int(np.ceil(len(img_dict) / 4)), 4, i + 1)
            imshow(img, name, name.__contains__("clipped"))

        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        if path:
            plt.savefig(path)
        plt.show()
    except urllib3.exceptions.HTTPError and urllib.error.HTTPError:
        print("error: please close plots to free memory")


def show_tensor_images(tensor_imgs, title=None, path=None):
    try:
        plt.figure(figsize=(len(tensor_imgs) * 5, 5))
        for i, tensor_img in enumerate(tensor_imgs):
            plt.subplot(1, len(tensor_imgs), i + 1)
            if title:
                imshow(tensor_img.detach().cpu(), title[i])
            else:
                imshow(tensor_img.detach().cpu())
        if path:
            plt.savefig(path)
        plt.show()
    except urllib3.exceptions.HTTPError and urllib.error.HTTPError:
        print("error: please close plots to free memory")


class AverageMeter(object):
    """
    used to store and update the loss value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ------------------------------------ checkpoint related ------------------------------------

def save_checkpoint(epoch, epochs_since_improvement, u_net_model, optimizer, best_loss, is_best):
    state = {'epoch': epoch,
             'epoch_since_improvement': epochs_since_improvement,
             'u_net_model': u_net_model,
             'optimizer': optimizer,
             'loss': best_loss,
             'is_best': is_best}

    file_name = 'checkpoint.tar'
    torch.save(state, file_name)
    if is_best:
        torch.save(state, 'BEST_checkpoint.tar')
    print("check point saved")


def load_checkpoint_model(checkpoint):
    checkpoint = torch.load(checkpoint)
    return checkpoint['u_net_model']


def choose_best_checkpoint(path):
    """ get the best checkpoint based on validation loss from each epoch segments"""
    min_loss = float('inf')
    best_path = ""
    for ep in os.listdir(path):
        ep_path = os.path.join(path, ep, "BEST_checkpoint.tar")
        checkpoint = torch.load(ep_path)
        if checkpoint["loss"] < min_loss:
            min_loss = checkpoint["loss"]
            best_path = ep_path

    return best_path

# ----------------------------------------- logger -----------------------------------------

def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ----------------------------------------- Others -----------------------------------------

def make_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        print("OSError when creating directory {}".format(path))


def random_bool():
    return bool(getrandbits(1))
