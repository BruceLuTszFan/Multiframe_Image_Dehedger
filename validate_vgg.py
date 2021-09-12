import os
import numpy as np

from torch.utils.data import DataLoader
from dataset import StaticHedgedDataset
from VGGClass import VGGClass
from config import device
from utils import AverageMeter
from loss_functions import vgg_accuracy


vgg_model = VGGClass().to(device)
vgg_model.eval()

density_range = np.arange(0.0, 0.9, 0.1)
val_dataset = StaticHedgedDataset(os.path.join('..', 'data', 'synthetic_data', 'val'), density_range)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)
top_1_acc = AverageMeter()
top_5_acc = AverageMeter()

# top 1 should be about 0.45, top 5 should be around 0.7
for i, (_, ground_truths, _, label) in enumerate(val_loader):
    ground_truths = ground_truths.to(device)
    label = label.to(device)
    # validate vgg
    top_1, top_5 = vgg_accuracy(vgg_model, ground_truths, label)
    top_5_acc.update(top_5)
    top_1_acc.update(top_1)

    if i % 50 == 0:
        print("{}/{} top 1 {}, top 5 {}".format(i, len(val_loader), top_1_acc.avg, top_5_acc.avg))