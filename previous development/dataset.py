import random

import numpy as np
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision


class StaticHedgedDataset(Dataset):

    def __init__(self, root_dir, density_range):
        self.root_dir = root_dir
        self.density_range = density_range

        self.label_no = -1
        self.image_no = -1
        self.density_no = len(density_range)
        self.sample_no = -1
        self.get_image_count()

        self.image_path = []
        self.path_to_ans = dict()
        self.read_static_images()

        self.dataset_validation()
        self.to_tensor = torchvision.transforms.ToTensor()

    def __len__(self):
        # train = 1000 * 32 = 32000
        # val = 1000 * 8 = 8000
        return self.label_no * self.image_no

    def __getitem__(self, idx):
        """
        self.image_path would be 2 dimensional with shape [label, images]

        we have to decide the density as well as the sample randomly
        """
        current_image_path = self.image_path[random.randint(0, self.label_no - 1)][random.randint(0, self.image_no - 1)]
        label = int(current_image_path.split("\\")[-2])
        density = str(round(self.density_range[idx%len(self.density_range)], 2))  # uniform density across dataset
        sample = str(random.randint(0, self.sample_no - 1))
        full_ground_truth_path = os.path.join(current_image_path, current_image_path.split("\\")[-1] + ".JPEG")
        full_masked_image_path = os.path.join(current_image_path, density, sample + ".jpeg")
        # full_hedge_path = os.path.join(current_image_path, density, sample + "_hedge.png")

        # if can't find try another one (just in case if some file is missing)
        try:
            masked_background = self.to_tensor(Image.open(full_masked_image_path).convert("RGB"))
            input_img = self.to_tensor(Image.open(full_ground_truth_path).convert("RGB"))
        except FileNotFoundError:
            print("{} not found".format(full_masked_image_path))
            return self.__getitem__(idx)
        return masked_background, input_img, label

    def get_image_count(self):
        self.label_no = len(os.listdir(self.root_dir))
        label_path = os.path.join(self.root_dir, os.listdir(self.root_dir)[0])
        self.image_no = len(os.listdir(label_path))
        image_path = os.path.join(label_path, os.listdir(label_path)[0])
        density_path = os.path.join(image_path, os.listdir(image_path)[0])
        self.sample_no = int(len(os.listdir(density_path)) / 2)  # hedged background and hedge mask so / 2

    def read_static_images(self):
        """
        We store the path for each image in a multi-dimension list, and the range of each dimension is recorded so that
        at inference time the indices can be randomly generated and the path can be retrieved without further effort

        For the labels, we would store the path-to-[label, ground truth] relationship using a dict(), so given a path,
        its label would and ground truth image would be retrieved

        :return:
        """

        for label in os.listdir(self.root_dir):
            current_label = []
            label_dir = os.path.join(self.root_dir, label)
            for image in os.listdir(label_dir):
                # current_image = []
                image_dir = os.path.join(label_dir, image)
                # we know the ground truth would have the same name with the folder with an extra '.JPEG' as extension
                current_ground_truth = os.path.join(image_dir, image + ".JPEG")
                current_label.append(image_dir)
            self.image_path.append(current_label)

    def dataset_validation(self):
        """ make sure all files are loaded correctly """
        assert len(self.image_path) == self.label_no
        for i in range(self.label_no):
            assert len(self.image_path[i]) == self.image_no

class RecurrentStaticHedgedDataset(StaticHedgedDataset):
    """
    inherit the StaticHedgeDataset, just need to perform some augmentation on how the data is loaded
    """

    def __init__(self, root_dir, density_range, sample_per_class, batch_size):
        super().__init__(root_dir, density_range)
        # currently there are 5 * 9 = 45 samples per image, recommended sample_per_class = 32
        self.sample_per_class = sample_per_class
        self.batch_size = batch_size
        self.current_count = float('inf')
        self.current_labels = None
        self.current_images = None

    def __getitem__(self, idx):
        """
        self.image_path would be 2 dimensional with shape [label, images]
        self.current_labels and self.current_images would be lists containing idx where their size = self.batch_size
        this means for each consequence batch, its ith item would have the same background
        and for each self.sample_per_class time, self.current_labels and self.current_image would be refreshed

        """
        if self.current_count >= self.sample_per_class * self.batch_size:
            # decide a random label + image
            self.refresh_images()
            self.current_count = 0

        label_idx = self.current_labels[self.current_count % self.batch_size]
        image_idx = self.current_images[self.current_count % self.batch_size]
        current_image_path = self.image_path[label_idx][image_idx]
        label = int(current_image_path.split("\\")[-2])
        # random density + sample chosen from dataset
        density = str(round(self.density_range[idx%len(self.density_range)], 2))  # uniform density across dataset
        sample = str(random.randint(0, self.sample_no - 1))
        full_ground_truth_path = os.path.join(current_image_path, current_image_path.split("\\")[-1] + ".JPEG")
        full_masked_image_path = os.path.join(current_image_path, density, sample + ".jpeg")

        # if can't find try another one (just in case if some file is missing)
        try:
            masked_background = self.to_tensor(Image.open(full_masked_image_path).convert("RGB"))
            input_img = self.to_tensor(Image.open(full_ground_truth_path).convert("RGB"))
        except FileNotFoundError:
            print("{} not found".format(full_masked_image_path))
            return self.__getitem__(idx)

        self.current_count += 1
        return masked_background, input_img, label

    def refresh_images(self):
        self.current_labels = torch.randint(0, self.label_no - 1, size=(self.batch_size,), requires_grad=False)
        self.current_images = torch.randint(0, self.image_no - 1, size=(self.batch_size,), requires_grad=False)


class DynamicHedgedDataset(Dataset):

    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.images_paths, self.labels = self.read_images()

    def __len__(self):
        return len(self.images_paths)

    def read_images(self):
        """
        :return: size of dataset, all the image paths and its labels
        """
        try:
            images = []
            labels = []
            for label in os.listdir(self.root_dir):
                for image in os.listdir(os.path.join(self.root_dir, label)):
                    path = os.path.join(self.root_dir, label, image)
                    images.append(path)
                    labels.append(int(label) - 1)
            return images, labels
        except OSError:
            print("error on reading {}".format(self.root_dir))

    def __getitem__(self, idx):
        image = Image.open(self.images_paths[idx])
        image = image.convert("RGB")
        label = self.labels[idx]
        input_img, masked_background, hedge_mask = self.transform(image)

        return masked_background, input_img, hedge_mask, label


class StaticHedgedPairsDataset(StaticHedgedDataset):
    """
    inherit the StaticHedgeDataset, just need to perform some augmentation on how the data is loaded
    """

    def __init__(self, root_dir, density_range):
        super().__init__(root_dir, density_range)

    def __getitem__(self, idx):
        """
        each time we have to return a pair of masked image with same background
        """
        current_image_path = self.image_path[random.randint(0, self.label_no - 1)][random.randint(0, self.image_no - 1)]
        label = int(current_image_path.split("\\")[-2])
        density = str(round(self.density_range[idx%len(self.density_range)], 2))  # uniform density across dataset
        samples = np.random.choice(range(self.sample_no), size=2, replace=False)
        # sample = str(random.randint(0, self.sample_no - 1))
        density_2 = str(round(random.choice(self.density_range), 1))
        # sample_2 = str(random.randint(0, self.sample_no - 1))
        full_ground_truth_path = os.path.join(current_image_path, current_image_path.split("\\")[-1] + ".JPEG")
        full_masked_image_path = os.path.join(current_image_path, density, str(samples[0]) + ".jpeg")
        # full_masked_image_path_2 = os.path.join(current_image_path, density_2, sample_2 + ".jpeg")
        full_masked_image_path_2 = os.path.join(current_image_path, density_2, str(samples[1]) + ".jpeg")
        # full_hedge_path = os.path.join(current_image_path, density, sample + "_hedge.png")

        # if can't find try another one (just in case if some file is missing)
        try:
            masked_background = self.to_tensor(Image.open(full_masked_image_path).convert("RGB"))
            masked_background_2 = self.to_tensor(Image.open(full_masked_image_path_2).convert("RGB"))
            input_img = self.to_tensor(Image.open(full_ground_truth_path).convert("RGB"))
        except FileNotFoundError:
            print("{} not found".format(full_masked_image_path))
            return self.__getitem__(idx)
        return masked_background, masked_background_2, input_img, label

class StaticHedgedQuintupletDataset(StaticHedgedDataset):
    """
    inherit the StaticHedgeDataset, just need to perform some augmentation on how the data is loaded
    """

    def __init__(self, root_dir, density_range):
        super().__init__(root_dir, density_range)

    def __getitem__(self, idx):
        """
        each time we have to return a pair of masked image with same background
        """
        current_image_path = self.image_path[random.randint(0, self.label_no - 1)][random.randint(0, self.image_no - 1)]
        label = int(current_image_path.split("\\")[-2])
        density = str(round(self.density_range[idx%len(self.density_range)], 2))  # uniform density across dataset
        density_2 = str(round(random.choice(self.density_range), 1))
        density_3 = str(round(random.choice(self.density_range), 1))
        density_4 = str(round(random.choice(self.density_range), 1))
        density_5 = str(round(random.choice(self.density_range), 1))
        full_ground_truth_path = os.path.join(current_image_path, current_image_path.split("\\")[-1] + ".JPG")
        full_masked_image_path = os.path.join(current_image_path, density, "0.jpg")
        full_masked_image_path_2 = os.path.join(current_image_path, density_2, "1.jpg")
        full_masked_image_path_3 = os.path.join(current_image_path, density_3, "2.jpg")
        full_masked_image_path_4 = os.path.join(current_image_path, density_4, "3.jpg")
        full_masked_image_path_5 = os.path.join(current_image_path, density_5, "4.jpg")
        # full_hedge_path = os.path.join(current_image_path, density, sample + "_hedge.png")

        # if can't find try another one (just in case if some file is missing)
        try:
            masked_background = self.to_tensor(Image.open(full_masked_image_path).convert("RGB"))
            masked_background_2 = self.to_tensor(Image.open(full_masked_image_path_2).convert("RGB"))
            masked_background_3 = self.to_tensor(Image.open(full_masked_image_path_3).convert("RGB"))
            masked_background_4 = self.to_tensor(Image.open(full_masked_image_path_4).convert("RGB"))
            masked_background_5 = self.to_tensor(Image.open(full_masked_image_path_5).convert("RGB"))
            input_img = self.to_tensor(Image.open(full_ground_truth_path).convert("RGB"))
        except FileNotFoundError:
            print("{} not found".format(full_masked_image_path))
            return self.__getitem__(idx)
        return masked_background, masked_background_2, masked_background_3, masked_background_4, masked_background_5, input_img, label


# density_range = np.arange(0.0, 0.9, 0.1)
# train_dataset = StaticHedgedDataset(os.path.join('..', 'data', 'synthetic_data', 'val'), density_range)
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
