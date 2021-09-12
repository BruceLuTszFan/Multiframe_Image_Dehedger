import numpy as np
import torch
import torch.nn.functional as F


def vgg_accuracy(model, images, labels):
    # there is 64 labels and images coming in
    model.eval()
    pred = model(images)
    label = labels.reshape(labels.shape[0], 1)

    # top-1
    _, predicted = torch.topk(pred.data, k=1, dim=1)
    top_1_result = torch.sum(predicted == label).item()

    # top-5
    _, predicted = torch.topk(pred.data, k=5, dim=1)
    top_5_result = np.sum([label[i] in predicted[i] for i in range(labels.shape[0])])

    size = labels.shape[0]
    return top_1_result / size, top_5_result / size


def uncertainty_MSE_loss(pred, uncertainty, target):
    mse = F.mse_loss(pred, target, reduction="none")
    loss = torch.mean(torch.exp(-uncertainty) * mse + uncertainty)
    return loss, mse.detach().mean()
