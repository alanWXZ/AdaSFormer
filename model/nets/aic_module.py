import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Bottleneck3D(nn.Module):

    def __init__(self, inplanes, planes, norm_layer, stride=1, dilation=[1, 1, 1], expansion=4, downsample=None,
                 fist_dilation=1, multi_grid=1,kernel=(3, 5, 7),
                 bn_momentum=0.0003):
        super(Bottleneck3D, self).__init__()
        # often锛宲lanes = inplanes // 4
        norm_layer=nn.BatchNorm3d
        self.expansion = expansion
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 1, 3), stride=(1, 1, stride),
                               dilation=(1, 1, dilation[0]), padding=(0, 0, dilation[0]), bias=False)
        self.bn2 = norm_layer(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 1), stride=(1, stride, 1),
                               dilation=(1, dilation[1], 1), padding=(0, dilation[1], 0), bias=False)
        self.bn3 = norm_layer(planes, momentum=bn_momentum)
        self.conv4 = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1),
                               dilation=(dilation[2], 1, 1), padding=(dilation[2], 0, 0), bias=False)
        self.bn4 = norm_layer(planes, momentum=bn_momentum)
        self.conv5 = nn.Conv3d(planes, planes * self.expansion, kernel_size=(1, 1, 1), bias=False)
        self.bn5 = norm_layer(planes * self.expansion, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        #self.downsample = None

        self.dilation = dilation
        self.stride = stride

        self.downsample2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, stride, 1), stride=(1, stride, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.conv_1x1xk = nn.ModuleList()
        self.conv_1xkx1 = nn.ModuleList()
        self.conv_kx1x1 = nn.ModuleList()
        self.n = len(kernel)
        self.avp_z=nn.AvgPool3d(kernel_size=(1, 1, stride), stride=(1, 1, stride))
        self.avp_x = nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1))
        self.avp_y = nn.AvgPool3d(kernel_size=(1, stride, 1), stride=(1, stride, 1))
        for _idx in range(self.n):
            k = kernel[_idx]
            d = dilation[_idx]
            p = k // 2 * d
            self.conv_1x1xk.append(
                nn.Conv3d(planes, planes, (1, 1, k), stride=(1, 1, stride), padding=(0, 0, p), bias=True, dilation=(1, 1, d)))
            self.conv_1xkx1.append(nn.Conv3d(planes, planes, (1, k, 1), stride=(1, stride, 1), padding=(0, p, 0), bias=True, dilation=(1, d, 1)))
            self.conv_kx1x1.append(nn.Conv3d(planes, planes, (k, 1, 1), stride=(stride, 1, 1), padding=(p, 0, 0), bias=True, dilation=(d, 1, 1)))
        self.conv_mx = nn.Conv3d(inplanes, 3 * self.n, (1, 1, 1), stride=1, padding=0, bias=False, dilation=1)
        self.softmax = nn.Softmax(dim=2)
        self.planes=planes

    def forward(self, x):
        residual = x
        mx = self.conv_mx(x)  # (BS, 3n, D, H, W)
        _bs, _tn, _d, _h, _w = mx.size()
        mx = mx.view(_bs, 3, -1, _d, _h, _w)  # (BS, 3, n, D, H, W)

        # print("After 'view', mx.size() is: {}".format(mx.size()))
        mx = self.softmax(mx)  # dim=2

        mx_c = torch.unsqueeze(mx, dim=3)  # (BS, 3, n, 1, D, H, W)
        mx_c = mx_c.expand(-1, -1, -1, self.planes, -1, -1, -1)  # (BS, 3, n, c, D, H, W)
        # mx1, mx2, mx3 = torch.split(mx_c, 1, dim=2)  # n x (BS, 3, 1, c, D, H, W)
        mx_list = torch.split(mx_c, 1, dim=2)  # n x (BS, 3, 1, c, D, H, W)
        mx_z_list = []
        mx_y_list = []
        mx_x_list = []
        for i in range(self.n):
            # mx_list[i] = torch.squeeze(mx_list[i], dim=2)  # (BS, 3, c, D, H, W)
            # mx_z, mx_y, mx_x = torch.split(mx_list[i], 1, dim=1)  # 3 x (BS, 1, c, D, H, W)
            mx_z, mx_y, mx_x = torch.split(torch.squeeze(mx_list[i], dim=2), 1, dim=1)  # 3 x (BS, 1, c, D, H, W)
            mx_z_list.append(self.avp_z(torch.squeeze(mx_z, dim=1)))  # (BS, c, D, H, W)
            mx_y_list.append(self.avp_y(self.avp_z(torch.squeeze(mx_y, dim=1))))  # (BS, c, D, H, W)
            mx_x_list.append(self.avp_x(self.avp_y(self.avp_z(torch.squeeze(mx_x, dim=1)))))  # (BS, c, D, H, W)
        x_in = self.relu(self.bn1(self.conv1(x)))
        y_x = None
        for _idx in range(self.n):
            y1_x = self.conv_1x1xk[_idx](x_in)
            y1_x = F.relu(y1_x, inplace=True)
            y1_x = torch.mul(mx_z_list[_idx], y1_x)
            y_x = y1_x if y_x is None else y_x + y1_x


        y_x = self.bn2(y_x)
        y_x_relu = self.relu(y_x)
        y_y = None
        for _idx in range(self.n):
            y1_y = self.conv_1xkx1[_idx](y_x_relu)
            y1_y = F.relu(y1_y, inplace=True)
            y1_y = torch.mul(mx_y_list[_idx], y1_y)
            y_y = y1_y if y_y is None else y_y + y1_y
        y_y = self.bn3(y_y)

        if self.stride != 1:
            y_x = self.downsample2(y_x)
        y_y = y_x + y_y
        y_y_relu = self.relu(y_y)

        y_z = None
        for _idx in range(self.n):
            y1_z = self.conv_kx1x1[_idx](y_y_relu)
            y1_z = F.relu(y1_z, inplace=True)
            y1_z = torch.mul(mx_x_list[_idx], y1_z)
            y_z = y1_z if y_z is None else y_z + y1_z
        y_z = self.bn4(y_z)
        if self.stride != 1:
            y_x = self.downsample3(y_x)
            y_y = self.downsample4(y_y)


        y_z=y_x+y_y+y_z
        y_z_relu = self.relu(y_z)



        out5 = self.bn5(self.conv5(y_z_relu))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out5 + residual
        out_relu = self.relu(out)


        return out_relu


class BasicAIC3d(nn.Module):
    def __init__(self, channel, kernel=(3, 5, 7), dilation=(1, 1, 1), residual=True):
        super(BasicAIC3d, self).__init__()
        self.channel = channel
        self.residual = residual
        self.n = len(kernel)  # number of kernels
        self.conv_mx = nn.Conv3d(channel, 3 * self.n, (1, 1, 1), stride=1, padding=0, bias=False, dilation=1)
        self.softmax = nn.Softmax(dim=2)  # Applies the Softmax function in each axis

        # ---- Convs of each axis
        self.conv_1x1xk = nn.ModuleList()
        self.conv_1xkx1 = nn.ModuleList()
        self.conv_kx1x1 = nn.ModuleList()

        c = channel
        for _idx in range(self.n):
            k = kernel[_idx]
            d = dilation[_idx]
            p = k // 2 * d
            self.conv_1x1xk.append(nn.Conv3d(c, c, (1, 1, k), stride=1, padding=(0, 0, p), bias=True, dilation=(1, 1, d)))
            self.conv_1xkx1.append(nn.Conv3d(c, c, (1, k, 1), stride=1, padding=(0, p, 0), bias=True, dilation=(1, d, 1)))
            self.conv_kx1x1.append(nn.Conv3d(c, c, (k, 1, 1), stride=1, padding=(p, 0, 0), bias=True, dilation=(d, 1, 1)))

    def forward(self, x):
        mx = self.conv_mx(x)  # (BS, 3n, D, H, W)
        _bs, _tn, _d, _h, _w = mx.size()
        mx = mx.view(_bs, 3, -1, _d, _h, _w)  # (BS, 3, n, D, H, W)

        # print("After 'view', mx.size() is: {}".format(mx.size()))
        mx = self.softmax(mx)  # dim=2

        mx_c = torch.unsqueeze(mx, dim=3)  # (BS, 3, n, 1, D, H, W)
        mx_c = mx_c.expand(-1, -1, -1, self.channel, -1, -1, -1)  # (BS, 3, n, c, D, H, W)
        # mx1, mx2, mx3 = torch.split(mx_c, 1, dim=2)  # n x (BS, 3, 1, c, D, H, W)
        mx_list = torch.split(mx_c, 1, dim=2)  # n x (BS, 3, 1, c, D, H, W)

        mx_z_list = []
        mx_y_list = []
        mx_x_list = []
        for i in range(self.n):
            # mx_list[i] = torch.squeeze(mx_list[i], dim=2)  # (BS, 3, c, D, H, W)
            # mx_z, mx_y, mx_x = torch.split(mx_list[i], 1, dim=1)  # 3 x (BS, 1, c, D, H, W)
            mx_z, mx_y, mx_x = torch.split(torch.squeeze(mx_list[i], dim=2), 1, dim=1)  # 3 x (BS, 1, c, D, H, W)
            mx_z_list.append(torch.squeeze(mx_z, dim=1))  # (BS, c, D, H, W)
            mx_y_list.append(torch.squeeze(mx_y, dim=1))  # (BS, c, D, H, W)
            mx_x_list.append(torch.squeeze(mx_x, dim=1))  # (BS, c, D, H, W)

        # ------ x ------
        y_x = None
        for _idx in range(self.n):
            y1_x = self.conv_1x1xk[_idx](x)
            y1_x = F.relu(y1_x, inplace=True)
            y1_x = torch.mul(mx_x_list[_idx], y1_x)
            y_x = y1_x if y_x is None else y_x + y1_x

        # ------ y ------
        y_y = None
        for _idx in range(self.n):
            y1_y = self.conv_1xkx1[_idx](y_x)
            y1_y = F.relu(y1_y, inplace=True)
            y1_y = torch.mul(mx_y_list[_idx], y1_y)
            y_y = y1_y if y_y is None else y_y + y1_y

        # ------ z ------
        y_z = None
        for _idx in range(self.n):
            y1_z = self.conv_kx1x1[_idx](y_y)
            y1_z = F.relu(y1_z, inplace=True)
            y1_z = torch.mul(mx_z_list[_idx], y1_z)
            y_z = y1_z if y_z is None else y_z + y1_z

        y = F.relu(y_z + x, inplace=True) if self.residual else F.relu(y_z, inplace=True)
        return y
