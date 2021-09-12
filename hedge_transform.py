import os
import random
import numpy as np
from collections import defaultdict
import torch, torchvision
from PIL import Image

class HedgeTransform(object):
    def __init__(self, density_list, frame_no, img_size, hedge_path):
        self.setDensity(density_list)
        self.setFrames(frame_no)
        self.setImgSize(img_size)
        self.to_tensor = torchvision.transforms.ToTensor()

        self.hedge_masks_dict = defaultdict(list)

        # all the hedge masks are prepared in png format (RGBA), have to load them into a dictionary
        # hedge_path = os.path.join('Hedge_masks')
        for density in os.listdir(hedge_path):
            subdir_path = os.path.join(hedge_path, density)
            for hedge_masks in os.listdir(subdir_path):
                image_path = os.path.join(subdir_path, hedge_masks)
                self.hedge_masks_dict[float(density)].append(image_path)

    def setDensity(self, densities):
        self.densities = densities

    def setFrames(self, n_frames):
        self.n_frames = n_frames

    def setImgSize(self, img_size):
        self.img_size = img_size

    def __call__(self, input_img):
        # TODO make the crop size dynamic / customizable
        crop_width = crop_height = 1000
        input_img = input_img.copy()
        input_img = input_img.resize((crop_width, crop_height))

        masked_backgrounds, hedge_masks = None, None
        # generate n frames using same background pictures + different hedge masks
        for _ in range(self.n_frames):
            # get the background and hedge image
            current_background = input_img.copy()
            density = random.choice(self.densities)
            density = round(density, 1)
            current_hedge = Image.open(random.choice(self.hedge_masks_dict[density]))

            # make sure the cropping size is smaller than the hedge
            hedge_width, hedge_height = current_hedge.size
            if min(hedge_width, hedge_height) < max(crop_width, crop_height):
                crop_width = crop_height = min(hedge_width, hedge_height)

            # find a random point to crop
            left = np.random.randint(0, hedge_width - crop_width + 1)
            top = np.random.randint(0, hedge_height - crop_height + 1)

            current_hedge_mask = current_hedge.crop((left, top, left + crop_width, top + crop_height))

            # make sure background and hedge mask have same width/height before pasting
            current_background = current_background.resize((crop_width, crop_height))
            current_background.paste(current_hedge_mask, (0, 0), mask=current_hedge_mask)

            # Downsample the image so that the edge of the hedge wouldn't stand out
            current_background = current_background.resize(self.img_size)
            current_hedge_mask = current_hedge_mask.resize(self.img_size)

            if masked_backgrounds is None:
                masked_backgrounds = self.to_tensor(current_background)
            else:
                masked_backgrounds = torch.cat([masked_backgrounds, self.to_tensor(current_background)], dim=0)
            if hedge_masks is None:
                hedge_masks = self.to_tensor(current_hedge_mask)
            else:
                hedge_masks = torch.cat([hedge_masks, self.to_tensor(current_hedge_mask)], dim=0)

        # get the ground truth
        input_img = self.to_tensor(input_img.resize(self.img_size))

        # we return the ground truth, hedged image and the mask
        return input_img, masked_backgrounds, hedge_masks
