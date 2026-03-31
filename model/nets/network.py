# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import partial
from collections import OrderedDict
from config import config

from .arch_blocks import Bottleneck3D, Feature2DTo3D, SimpleRB
from .unet2d import UNet2D

from transformer.pt_transformer import *
from .gumbel_softmax import *




class STAGE2(nn.Module):
    def __init__(self, class_num, norm_layer, backbone_feat_ch=2048, feature=512, ThreeDinit=True,
                 bn_momentum=0.1, eval=False, freeze_bn=False):
        super(STAGE2, self).__init__()
        self.business_layer = []
        norm1_layer=nn.BatchNorm2d
        norm_layer=nn.BatchNorm3d
        if eval:
            self.downsample = nn.Sequential(
                nn.Conv2d(backbone_feat_ch, feature, kernel_size=1, bias=False),
                nn.BatchNorm2d(feature, momentum=bn_momentum),
                nn.ReLU()
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(backbone_feat_ch, feature, kernel_size=1, bias=False),
                norm1_layer(feature, momentum=bn_momentum),
                nn.ReLU()
            )
        self.business_layer.append(self.downsample)

        self.backbone_feat_ch = backbone_feat_ch
        self.feature = feature
        self.ThreeDinit = ThreeDinit

        self.classify_semantic = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=3, stride=2, padding=1, dilation=1,
                                   output_padding=1),
                norm1_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                norm1_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.Dropout3d(.1),
                nn.Conv3d(128, class_num, kernel_size=1, bias=True)
            )]
        )

        self.feature_2d_to_3d = Feature2DTo3D()


        self.semantic_layer1 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=4, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature, momentum=bn_momentum),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
        )

        self.semantic_layer2 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=8, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature * 2,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature * 2, momentum=bn_momentum),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
        )
        self.business_layer.append(self.semantic_layer2)
        self.classify_semantic1 = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=3, stride=2, padding=1, dilation=1,
                                   output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.Dropout3d(.1),
                nn.Conv3d(feature, class_num, kernel_size=1, bias=True)
            ),
            nn.Sequential(
                nn.Dropout3d(.1),
                nn.Conv3d(feature, 2, kernel_size=1, bias=True)
            )]
        )
        self.business_layer.append(self.classify_semantic1)

        self.mask_net = nn.Sequential(nn.Conv3d(12,12,kernel_size=3, padding=1),
                                      nn.BatchNorm3d(12),
                                      nn.ReLU(),
                                      nn.Dropout3d(.1),
                                      nn.Conv3d(12,2,kernel_size=3, padding=1))
        self.business_layer.append(self.mask_net)

        self.enc = SerializedAttention(
            channels=128,
            num_heads=8,  # 可以根据需要调整
            patch_size=512,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            enable_flash=False
        )
        self.business_layer.append(self.enc)

        self.block = Block(
            channels = 128,
            num_heads = 8,
            enable_flash=False,
            patch_size = 512,
            adaLN=True,
            shift=0,
        )
        self.block1 = Block(
            channels=128,
            num_heads=8,
            enable_flash=False,
            patch_size=512,
            adaLN=True,
            shift=0,
        )
        self.block2 = Block(
            channels=128,
            num_heads=8,
            enable_flash=False,
            patch_size=512,
            adaLN=True,
            shift=0,
        )

        self.shift_l = GumbelSoftmaxParameter(num_bins=32, tau_init=3.0, tau_min=1, anneal_rate=0.03)

        self.shift_l1 = GumbelSoftmaxParameter(num_bins=32, tau_init=3.0, tau_min=1, anneal_rate=0.03)

        self.shift_l2 = GumbelSoftmaxParameter(num_bins=32, tau_init=3.0, tau_min=1, anneal_rate=0.03)


    def forward(self, feature2d, depth_mapping_3d, epoch):
        feature3d = self.feature_2d_to_3d(feature2d, depth_mapping_3d)


        b, c, _, _, _ = feature3d.shape

        shift = self.shift_l()
        self.shift_l.update_tau(epoch)

        shift = shift.argmax().item()

        voxel_point = VoxelPoint(feature3d)
        voxel_point.serialization(order=["z"], shuffle_orders=True,shift=16*shift)
        voxel_point.sparsify()
        voxel_point.feat = voxel_point.non_empty_features
        voxel_point = self.block(voxel_point,feature3d)
        shift1 = self.shift_l1()
        self.shift_l1.update_tau(epoch)

        shift1 = shift1.argmax().item()

        voxel_point.serialization(order=["z"], shuffle_orders=True, shift=16*shift1)
        voxel_point = self.block1(voxel_point,feature3d)


        voxel_output = voxel_point.get_voxel_output(voxel_point)

        semantic1 = self.semantic_layer1(voxel_output)

        voxel_point1 = VoxelPoint(semantic1)

        shift2 = self.shift_l2()
        self.shift_l2.update_tau(epoch)

        shift2 = shift2.argmax().item()
        voxel_point1.serialization(order=["z"], shuffle_orders=True,shift=16*shift2)
        voxel_point1.sparsify()
        voxel_point1.feat = voxel_point1.non_empty_features
        voxel_point1 = self.block2(voxel_point1,semantic1)

        voxel_output1 = voxel_point1.get_voxel_output(voxel_point1)

        semantic2 = self.semantic_layer2(voxel_output1)

        up_sem1 = self.classify_semantic1[0](semantic2)
        up_sem1 = up_sem1 + semantic1
        up_sem2 = self.classify_semantic1[1](up_sem1)
        up_sem2 = up_sem2 + F.interpolate(up_sem1, size=[60, 60, 36], mode="trilinear", align_corners=True)


        input_feature = up_sem2
        h_feature = input_feature
        pred_ssc = self.classify_semantic[2](h_feature)
        ssc_prediction = [pred_ssc]
        return ssc_prediction, []


'''
main network
'''
class Network(nn.Module):
    def __init__(self, class_num, norm_layer, backbone_feat_ch=2048, feature=512, ThreeDinit=True,
                 bn_momentum=0.1, eval=False, freeze_bn=False):
        super(Network, self).__init__()
        self.business_layer = []

        self.backbone = UNet2D.build(out_feature=feature, use_decoder=True, frozen_encoder=True)

        self.stage2 = STAGE2(class_num, norm_layer, backbone_feat_ch=backbone_feat_ch, feature=feature, ThreeDinit=ThreeDinit,
                             bn_momentum=bn_momentum, eval=eval, freeze_bn=freeze_bn)
        self.business_layer += self.stage2.business_layer

        self.oper_tsdf = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(32, 128, kernel_size=3, padding=1, bias=False),
            norm_layer(128, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.business_layer.append(self.oper_tsdf)



        self.semantic_layer1 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=4, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature, momentum=bn_momentum),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.semantic_layer1)
        self.semantic_layer2 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=8, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature * 2,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature * 2, momentum=bn_momentum),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.semantic_layer2)







    def forward(self, rgb, depth_mapping_3d, tsdf, epoch):
        feature2d = self.backbone(rgb)
        # tsdf_feature=self.oper_tsdf(tsdf)

        feature2d = feature2d['1_16']

        pred_semantic, _ = self.stage2(feature2d, depth_mapping_3d, epoch)

        if self.training:
            return pred_semantic, _
        return pred_semantic, _

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


if __name__ == '__main__':
    model = Network(class_num=12, norm_layer=nn.BatchNorm3d, feature=128, eval=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    left = torch.rand(1, 3, 480, 640).cuda()
    right = torch.rand(1, 3, 480, 640).cuda()
    depth_mapping_3d = torch.from_numpy(np.ones((1, 129600)).astype(np.int64)).long().cuda()
    tsdf = torch.rand(1, 1, 60, 36, 60).cuda()

    out = model(left, depth_mapping_3d, tsdf, 0)