import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, non_linearity=nn.LeakyReLU()):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, padding=1, padding_mode='reflect'),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(out_channel, affine=True),
                                   nn.Conv2d(out_channel, out_channel, 3, padding=1, padding_mode='reflect'),
                                   non_linearity,
                                   nn.BatchNorm2d(out_channel, affine=True))

        self.block.apply(self.weights_init)

    def forward(self, x):
        return self.block(x)

    def weights_init(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList([Block(in_channels[0], in_channels[1])])
        self.down_samples = nn.ModuleList([])

        for i in range(1, len(in_channels)-1):
            self.blocks.append(Block(in_channels[i+1], in_channels[i+1]))
            self.down_samples.append(nn.Conv2d(in_channels[i], in_channels[i+1], 3, stride=2, padding=1, padding_mode='reflect'))

    def forward(self, x):
        connections = [x]
        for i in range(len(self.down_samples)):
            x = self.blocks[i](x)
            connections.append(x)
            x = self.down_samples[i](x)
        x = self.blocks[-1](x)
        return x, connections


class Decoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.up_samples = nn.ModuleList([])

        for i in range(0, len(out_channels)-2):
            self.up_samples.append(nn.ConvTranspose2d(out_channels[i], out_channels[i+1], 3, stride=2, padding=1, output_padding=1))
            self.blocks.append(Block(out_channels[i], out_channels[i+1]))

        self.blocks.append(Block(out_channels[-2], out_channels[-1]))


    def forward(self, x, connections):
        for i in range(len(self.up_samples)):
            x = self.up_samples[i](x)
            x = torch.cat([x, connections[i]], dim=1)
            x = self.blocks[i](x)
        x = self.blocks[-1](x)
        return x

class UncertaintyUNet(nn.Module):
    def __init__(self, c_in, c_out, u_out):
        super().__init__()
        in_channels = [c_in, 32, 64, 128, 256]
        out_channels = [256, 128, 64, 32, c_out]
        uncertainty_channels = [256, 128, 64, 32, u_out]

        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels)
        self.uncertainty_decoder = Decoder(uncertainty_channels)

    def forward(self, x):
        x, connections = self.encoder(x)
        output_pred = self.decoder(x, connections[::-1])
        uncertainty_pred = self.uncertainty_decoder(x, connections[::-1])
        # TODO test on torch.clamp on prediction to force net output stays in range [0, 1]
        output_pred = torch.clamp(output_pred, 0, 1)
        return output_pred, uncertainty_pred


class UNet(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        in_channels = [c_in, 32, 64, 128, 256]
        out_channels = [256, 128, 64, 32, c_out]

        # in_channels = [c_in, 18, 36, 64, 128]
        # out_channels = [128, 64, 32, 16, c_out]
        # uncertainty_channels = [128, 64, 32, 16, c_out]

        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels)

    def forward(self, x):
        x, connections = self.encoder(x)
        output_pred = self.decoder(x, connections[::-1])
        return output_pred