import torch
import numpy as np
from torch.nn import functional as F

class ResNet2D_block(torch.nn.Module):
    def __init__(self, channels=128, rate=1):
        super(ResNet2D_block, self).__init__()
        self.inorm1 = torch.nn.InstanceNorm2d(num_features=channels, eps=1e-6, affine=True)
        self.inorm2 = torch.nn.InstanceNorm2d(num_features=channels, eps=1e-6, affine=True)
        self.conv1 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, dilation=rate, padding=rate)
        self.net = torch.nn.Sequential(
            self.inorm1,
            torch.nn.ELU(inplace=True),
            self.conv1,
            self.inorm2,
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout(0.2),
            self.conv2,
        )
    def forward(self, x):
        out = self.net(x)
        return x + out
D

class RNAranknet1(torch.nn.Module):
    def __init__(self):
        super(RNAranknet1, self).__init__()
        self.CNN3D = torch.nn.Sequential(
            torch.nn.InstanceNorm3d(3, eps=1e-6, affine=True),
            torch.nn.Conv3d(3, 8, 5, padding=0, bias=False),
            torch.nn.Dropout(0.2),
            torch.nn.ELU(inplace=True),
            torch.nn.InstanceNorm3d(8, eps=1e-6, affine=True),
            torch.nn.Conv3d(8, 16, 5, padding=0, bias=True),
            torch.nn.Dropout(0.2),
            torch.nn.ELU(inplace=True),
            torch.nn.MaxPool3d(2, 2, padding=0),
            torch.nn.InstanceNorm3d(16, eps=1e-6, affine=True),
            torch.nn.Conv3d(16, 32, 3, padding=0, bias=True),
            torch.nn.Dropout(0.2),
            torch.nn.ELU(inplace=True),
            torch.nn.InstanceNorm3d(32, eps=1e-6, affine=True),
            torch.nn.Conv3d(32, 64, 3, padding=0, bias=True),
            torch.nn.Dropout(0.2),
            torch.nn.ELU(inplace=True)
        )

        self.Fully_Connection_1 = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64 * 8 * 8 * 8, 128),
            torch.nn.ELU(inplace=True),
        )

        self.Conv1D_2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=200, out_channels=128, kernel_size=3, stride=1, padding=1),  # 输入通道为1
        )

        self.Conv1D_3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=200, out_channels=128, kernel_size=3, stride=1, padding=1),  # 输入通道为1
        )

        self.CNN2D_1 = torch.nn.Sequential(
            ResNet2D_block(channels=104, rate=1),
            ResNet2D_block(channels=104, rate=2),
            ResNet2D_block(channels=104, rate=4),
            ResNet2D_block(channels=104, rate=8),
            ResNet2D_block(channels=104, rate=16),
            torch.nn.InstanceNorm2d(104, eps=1e-06, affine=True),
            torch.nn.ELU(inplace=True),
            torch.nn.Conv2d(104, 128, 1),
            ResNet2D_block(channels=128, rate=1),
        )

        self.ResNetBlock_backbone = torch.nn.Sequential(*[ResNet2D_block(rate=r) for r in [1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8,16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4,8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16]])
        self.ResNetBlock_deviation = torch.nn.Sequential(*[ResNet2D_block(rate=r) for r in [1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8,16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4,8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16]])
        self.ResNetBlock_contact = torch.nn.Sequential(*[ResNet2D_block(rate=r) for r in [1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8,16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4,8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16]])

        self.left_conv2d_lDDT = torch.nn.Sequential(
            ResNet2D_block(channels=128, rate=1),
            ResNet2D_block(channels=128, rate=2),
            ResNet2D_block(channels=128, rate=4),
            ResNet2D_block(channels=128, rate=8),
            ResNet2D_block(channels=128, rate=16),
            torch.nn.InstanceNorm2d(128, eps=1e-06, affine=True),
            torch.nn.ELU(inplace=True),
            torch.nn.Conv2d(128, 5, 1),
            ResNet2D_block(channels=5, rate=1),
        )

        self.right_conv2d_2 = torch.nn.Sequential(
            ResNet2D_block(channels=128, rate=1),
            ResNet2D_block(channels=128, rate=2),
            ResNet2D_block(channels=128, rate=4),
            ResNet2D_block(channels=128, rate=8),
            ResNet2D_block(channels=128, rate=16),
            torch.nn.InstanceNorm2d(128, eps=1e-06, affine=True),
            torch.nn.ELU(inplace=True),
            torch.nn.Conv2d(128, 1, 1),
            ResNet2D_block(channels=1, rate=1),
        )


    def forward(self, pixels, fea_1d, fea_2d):
        n = fea_1d.shape[0]
        pixels = self.CNN3D(pixels)
        x = torch.flatten(pixels, start_dim=1, end_dim=-1)
        x = self.Fully_Connection_1(x)
        x = torch.cat([x, fea_1d], dim=1)
        x_1 = self.Conv1D_2(x.unsqueeze(-1)).permute(2, 1, 0)
        x_2 = self.Conv1D_3(x.unsqueeze(-1)).permute(2, 1, 0)
        temp1 = tile(x_1.unsqueeze(3), 3, n)
        temp2 = tile(x_2.unsqueeze(2), 2, n)
        fea_2d = self.CNN2D_1(fea_2d)
        out2 = temp1 +temp2 + fea_2d
        backbone_out = self.ResNetBlock_backbone(out2)
        deviation = self.ResNetBlock_deviation(backbone_out)
        deviation = self.left_conv2d_lDDT(deviation)
        deviation = (deviation + deviation.permute(0, 1, 3, 2)) / 2
        deviation_prediction = F.softmax(deviation, dim=1)[0]
        contact = self.ResNetBlock_contact(backbone_out)
        contact = self.right_conv2d_2(contact)[:, 0, :, :]
        contact = (contact + contact.permute(0, 2, 1)) / 2
        contact_prediction = torch.sigmoid(contact)[0]
        lDDT_prediction = calculate_LDDT(deviation_prediction, contact_prediction)
        return lDDT_prediction, deviation_prediction, contact_prediction, (deviation, contact)

class RNAranknet2(torch.nn.Module):
    def __init__(self):
        super(RNAranknet2, self).__init__()
        self.CNN3D = torch.nn.Sequential(
            torch.nn.InstanceNorm3d(3, eps=1e-6, affine=True),
            torch.nn.Conv3d(3, 8, 5, padding=0, bias=False),
            torch.nn.Dropout(0.2),
            torch.nn.ELU(inplace=True),
            torch.nn.InstanceNorm3d(8, eps=1e-6, affine=True),
            torch.nn.Conv3d(8, 16, 5, padding=0, bias=True),
            torch.nn.Dropout(0.2),
            torch.nn.ELU(inplace=True),
            torch.nn.MaxPool3d(2, 2, padding=0),
            torch.nn.InstanceNorm3d(16, eps=1e-6, affine=True),
            torch.nn.Conv3d(16, 32, 3, padding=0, bias=True),
            torch.nn.Dropout(0.2),
            torch.nn.ELU(inplace=True),
            torch.nn.InstanceNorm3d(32, eps=1e-6, affine=True),
            torch.nn.Conv3d(32, 64, 3, padding=0, bias=True),
            torch.nn.Dropout(0.2),
            torch.nn.ELU(inplace=True)
        )

        self.Fully_Connection_1 = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64 * 8 * 8 * 8, 128),
            torch.nn.ELU(inplace=True),
        )

        self.Fully_Connection_2 = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(195, 128),
            torch.nn.ELU(inplace=True),
        )

        self.Fully_Connection_3 = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(195, 128),
            torch.nn.ELU(inplace=True),
        )

        self.CNN2D_1 = torch.nn.Sequential(
            torch.nn.InstanceNorm2d(58, eps=1e-06, affine=True),
            torch.nn.ELU(inplace=True),
            torch.nn.Conv2d(58, 58, 1),
            ResNet2D_block(channels=58, rate=1),
            torch.nn.InstanceNorm2d(58, eps=1e-06, affine=True),
            torch.nn.ELU(inplace=True),
            torch.nn.Conv2d(58, 128, 1),
            ResNet2D_block(channels=128, rate=1),
        )

        self.ResNetBlock_backbone = torch.nn.Sequential(*[ResNet2D_block(rate=r) for r in [1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16]])
        self.ResNetBlock_deviation = torch.nn.Sequential(*[ResNet2D_block(rate=r) for r in [1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16]])
        self.ResNetBlock_contact = torch.nn.Sequential(*[ResNet2D_block(rate=r) for r in [1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16]])

        self.left_conv2d_lDDT = torch.nn.Sequential(
            torch.nn.InstanceNorm2d(128, eps=1e-06, affine=True),
            torch.nn.ELU(inplace=True),
            torch.nn.Conv2d(128, 128, 1),
            ResNet2D_block(channels=128, rate=1),
            torch.nn.InstanceNorm2d(128, eps=1e-06, affine=True),
            torch.nn.ELU(inplace=True),
            torch.nn.Conv2d(128, 5, 1),
            ResNet2D_block(channels=5, rate=1),
        )

        self.right_conv2d_2 = torch.nn.Sequential(
            torch.nn.InstanceNorm2d(128, eps=1e-06, affine=True),
            torch.nn.ELU(inplace=True),
            torch.nn.Conv2d(128, 128, 1),
            ResNet2D_block(channels=128, rate=1),
            torch.nn.InstanceNorm2d(128, eps=1e-06, affine=True),
            torch.nn.ELU(inplace=True),
            torch.nn.Conv2d(128, 1, 1),
            ResNet2D_block(channels=1, rate=1),
        )


    def forward(self, pixels, fea_1d, fea_2d):
        n = fea_1d.shape[0]
        pixels = self.CNN3D(pixels)
        x = torch.flatten(pixels, start_dim=1, end_dim=-1)
        x = self.Fully_Connection_1(x)
        x = torch.cat([x, fea_1d], dim=1)
        x_1 = self.Fully_Connection_2(x).unsqueeze(0).permute(0, 2, 1)
        x_2 = self.Fully_Connection_3(x).unsqueeze(0).permute(0, 2, 1)
        temp1 = tile(x_1.unsqueeze(3), 3, n)
        temp2 = tile(x_2.unsqueeze(2), 2, n)
        fea_2d = self.CNN2D_1(fea_2d)
        out2 = temp1 +temp2 + fea_2d
        backbone_out = self.ResNetBlock_backbone(out2)
        deviation = self.ResNetBlock_deviation(backbone_out)
        deviation = self.left_conv2d_lDDT(deviation)
        deviation = (deviation + deviation.permute(0, 1, 3, 2)) / 2
        deviation_prediction = F.softmax(deviation, dim=1)[0]
        contact = self.ResNetBlock_contact(backbone_out)
        contact = self.right_conv2d_2(contact)[:, 0, :, :]
        contact = (contact + contact.permute(0, 2, 1)) / 2
        contact_prediction = torch.sigmoid(contact)[0]
        lDDT_prediction = calculate_LDDT(deviation_prediction, contact_prediction)
        return lDDT_prediction, deviation_prediction, contact_prediction, (deviation, contact)

def calculate_LDDT(deviation, mask):
    device = deviation.device
    nres = mask.shape[-1]
    mask = torch.mul(mask, torch.ones((nres, nres)).to(device) - torch.eye(nres).to(device))
    masked = torch.mul(deviation, mask)
    p0 = (masked[0]).sum(axis=0)
    p1 = (masked[1]).sum(axis=0) + p0
    p2 = (masked[2]).sum(axis=0) + p1
    p3 = (masked[3]).sum(axis=0) + p2
    p4 = mask.sum(axis=0)
    return 0.25 * (p0 + p1 + p2 + p3) / p4

def tile(a, dim, n_tile):
    device = a.device
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
    return torch.index_select(a, dim, order_index)
