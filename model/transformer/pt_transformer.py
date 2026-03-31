from .modules import PointModule, PointSequential
import torch.nn as nn
import spconv.pytorch as spconv
from .voxel_point import SerializedAttention, MLP, VoxelPoint
import torch
class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        adaLN=False,
        shift=0
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            shift=0
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.adaLN = adaLN
        self.shift = nn.Parameter(torch.ones(1))
        self.scale = nn.Parameter(torch.ones(1))

        self.mlp1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 2*channels, bias=True)
        )
        #self.shift=shift

    def forward(self, point, voxel_pre=None):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)

        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        if self.adaLN:

            shift,scale = self.adaLN_modulation(voxel_pre,point)
            
            shift = shift * self.shift
            scale = scale * self.scale
            point.feat = self.modulate(point.feat,shift,scale)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point

    def modulate(self, x, shift, scale):
        return x * (1 + scale) + shift

    def adaLN_modulation(self,x, y):
        out = x.mean(dim=(2, 3, 4), keepdim=True)
        out = out.view(out.size(0),out.size(1))
        pool = self.mlp1(out)

        shift,scale = pool.chunk(2, dim=1)

        shift_expanded = self._expand_by_batch_counts(shift, y.batch_counts)
        scale_expanded = self._expand_by_batch_counts(scale, y.batch_counts)

        return shift_expanded,scale_expanded

    
    def _expand_by_batch_counts(self, params, batch_counts):
        """
        根据每个batch的点数扩展参数
        Args:
            params: [batch_size, channels] 形状的参数
            batch_counts: 每个batch中点的数量列表
        Returns:
            expanded_params: [total_points, channels] 形状的参数
        """
        batch_counts_tensor = torch.tensor(batch_counts, device=params.device, dtype=torch.long)
        result = params.repeat_interleave(batch_counts_tensor, dim=0)

        return result  # [total_points, channels]