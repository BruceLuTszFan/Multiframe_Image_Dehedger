import torch
import torchvision
from torchvision import transforms


class VGGClass(torch.nn.Module):
    def __init__(self):
        super(VGGClass, self).__init__()
        self.vgg_net = torchvision.models.vgg16(pretrained=True)
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, input_img):
        input_img = torch.clamp(input_img, 0, 1)
        input_img = self.preprocess(input_img)
        vgg_class = self.vgg_net(input_img)
        return vgg_class
