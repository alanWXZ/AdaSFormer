import torch
import numpy as np
from typing import Dict, List, Tuple
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import spconv.pytorch as spconv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from model.serialization import encode
from model.serialization import *


# 添加RPE类
class RPE(nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        # 简化的RPE实现
        self.rpe_net = nn.Linear(3, num_heads)

    def forward(self, rel_pos):
        # 简化的相对位置编码
        return self.rpe_net(rel_pos)


# 添加flash_attn检查
try:
    import flash_attn
except ImportError:
    flash_attn = None


class VoxelPoint:
    def __init__(self, voxel_features: torch.Tensor):
        """
        初始化体素点云类
        Args:
            voxel_features: 体素特征张量 [B, C, D, H, W]
                B: batch size (2)
                C: 特征通道数 (128)
                D, H, W: 体素网格尺寸 (60, 36, 60)
        """
        self.voxel_features = voxel_features
        self.batch_size, self.num_channels, self.depth,  self.width, self.height= voxel_features.shape
        # 添加字典接口
        self._data = {}

        # 生成体素坐标网格
        self._generate_voxel_coords()

        # 获取非空体素
        self._extract_non_empty_voxels()

        # 计算batch和offset信息
        self._compute_batch_info()
        self.grid_size = 1.0

    def keys(self):
        """返回所有可用的键"""
        return self._data.keys()

    def __getitem__(self, key):
        """获取属性"""
        if hasattr(self, key):
            return getattr(self, key)
        elif key in self._data:
            return self._data[key]
        else:
            raise KeyError(f"Key '{key}' not found")

    def __setitem__(self, key, value):
        """设置属性"""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self._data[key] = value

    def __contains__(self, key):
        """检查键是否存在"""
        return hasattr(self, key) or key in self._data

    def _generate_voxel_coords(self):
        """生成体素坐标网格"""
        # 创建3D坐标网格
        x_coords = torch.arange(self.depth, device=self.voxel_features.device)
        y_coords = torch.arange(self.width, device=self.voxel_features.device)
        z_coords = torch.arange(self.height, device=self.voxel_features.device)

        # 生成网格坐标
        try:
            # 新版本PyTorch
            grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        except TypeError:
            # 老版本PyTorch
            grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords)
            # # 手动调整顺序以匹配 'ij' 索引
            # grid_x = grid_x.permute(1, 0, 2)
            # grid_y = grid_y.permute(1, 0, 2)
            # grid_z = grid_z.permute(1, 0, 2)

        # 重塑为 [D*H*W, 3] 的坐标
        self.voxel_coords_grid = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)

    def _extract_non_empty_voxels(self):
        """提取非空体素"""
        non_empty_voxels = []
        non_empty_coords = []
        batch_indices = []

        for batch_idx in range(self.batch_size):
            # 获取当前batch的特征
            batch_features = self.voxel_features[batch_idx]  # [C, D, H, W]

            # 计算每个体素的特征强度（例如：L2范数）
            voxel_intensity = torch.norm(batch_features, dim=0)  # [D, H, W]

            # 找到非空体素（强度大于阈值）
            threshold = 0.01  # 可以调整阈值
            non_empty_mask = voxel_intensity > threshold

            # 获取非空体素的索引
            non_empty_indices = torch.nonzero(non_empty_mask, as_tuple=False)

            if len(non_empty_indices) > 0:
                # 获取非空体素的特征
                batch_non_empty_features = batch_features[:, non_empty_mask]  # [C, N]
                batch_non_empty_features = batch_non_empty_features.T  # [N, C]

                # 获取非空体素的坐标
                batch_non_empty_coords = self.voxel_coords_grid[non_empty_mask.flatten()]

                # 添加到列表
                non_empty_voxels.append(batch_non_empty_features)
                non_empty_coords.append(batch_non_empty_coords)
                batch_indices.extend([batch_idx] * len(batch_non_empty_features))

        # 合并所有batch的非空体素
        if non_empty_voxels:
            self.non_empty_features = torch.cat(non_empty_voxels, dim=0)  # [N_total, C]
            self.non_empty_coords = torch.cat(non_empty_coords, dim=0)  # [N_total, 3]
            self.batch_indices = torch.tensor(batch_indices, dtype=torch.long).to(self.voxel_features.device)
            self.feat = self.non_empty_features
        else:
            # 如果没有非空体素，创建空张量
            self.non_empty_features = torch.empty((0, self.num_channels))
            self.non_empty_coords = torch.empty((0, 3))
            self.batch_indices = torch.empty((0,), dtype=torch.long).to(self.voxel_features.device)

    def _compute_batch_info(self):
        """计算batch和offset信息"""
        if len(self.batch_indices) == 0:
            self.offset = torch.tensor([0])
            return

        # 计算每个batch的体素数量
        unique_batches = torch.unique(self.batch_indices)
        batch_counts = []

        for batch_idx in unique_batches:
            count = torch.sum(self.batch_indices == batch_idx)
            batch_counts.append(count.item())

        # 计算offset
        self.offset = torch.cumsum(torch.tensor(batch_counts), dim=0)
        self.batch_counts = batch_counts

    def serialization(self, order="z", shift=0, depth=None, shuffle_orders=False):
        """
        点云序列化
        Args:
            order: 序列化顺序，可以是单个字符串或字符串列表
            depth: 序列化深度，如果为None则自动计算
            shuffle_orders: 是否打乱序列化顺序
        """
        # 设置序列化参数
        self.order = order
        assert len(self.batch_indices) > 0, "没有非空体素，无法进行序列化"

        grid_coord = self.non_empty_coords

        # 自动计算深度
        if depth is None:
            depth = int(grid_coord.max() + 1).bit_length()
        self.serialized_depth = depth

        # 检查深度限制
        assert depth * 3 + len(self.offset).bit_length() <= 63
        assert depth <= 16

        # 处理序列化顺序
        if isinstance(order, str):
            order = [order]

        self.grid_coord = grid_coord.to(self.voxel_features.device)

        # 生成序列化代码
        code = []
        for order_ in order:
            code.append(encode(self.grid_coord, self.batch_indices.to(self.voxel_features.device), depth, order=order_))
        code = torch.stack(code)

        # 排序
        order_indices = torch.argsort(code)
        
        inverse = torch.zeros_like(order_indices).scatter_(
            dim=1,
            index=order_indices,
            src=torch.arange(0, code.shape[1], device=order_indices.device).repeat(
                code.shape[0], 1
            ),
        )
        # if shift != 0:
        #     shift = shift % code.shape[1]  # 防止超过长度
        #     code = torch.roll(code, shifts=shift, dims=1)
        #     order_indices = torch.roll(order_indices, shifts=shift, dims=1)
        #     inverse = torch.roll(inverse, shifts=shift, dims=1)
        if shift != 0:
            # 对每个batch分别进行roll操作
            _offset = torch.cat([torch.tensor([0], device=code.device), self.offset.to(code.device)])
            for i in range(len(self.offset)):
                start_idx = _offset[i]
                end_idx = _offset[i + 1]
                batch_size = end_idx - start_idx

                if batch_size > 0:
                    # 计算当前batch的shift，防止超过batch长度
                    batch_shift = shift

                    # 对当前batch的数据进行roll
                    code[:, start_idx:end_idx] = torch.roll(code[:, start_idx:end_idx], shifts=batch_shift, dims=1)
                    order_indices[:, start_idx:end_idx] = torch.roll(order_indices[:, start_idx:end_idx],
                                                                     shifts=batch_shift, dims=1)
                    inverse[:, start_idx:end_idx] = torch.roll(inverse[:, start_idx:end_idx], shifts=batch_shift,
                                                               dims=1)
        # 打乱顺序（可选）
        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order_indices = order_indices[perm]
            inverse = inverse[perm]
        a = inverse.max()
        b = inverse.min()
        inverse = inverse.contiguous()
        # 存储序列化结果
        self.serialized_code = code
        self.serialized_order = order_indices
        self.serialized_inverse = inverse


        # print(f"序列化完成: 深度={depth}, 顺序={order}, 代码形状={code.shape}")

    def sparsify(self, pad=96):
        """
        Point Cloud Serialization

        Point cloud is sparse, here we use "sparsify" to specifically refer to
        preparing "spconv.SparseConvTensor" for SpConv.

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]

        pad: padding sparse for sparse shape.
        """
        # assert {"feat", "batch_counts"}.issubset(self.keys())
        # if "grid_coord" not in self.keys():
        assert hasattr(self, 'feat') and hasattr(self, 'batch_counts'), "缺少必要的属性: feat 或 batch_counts"

        if not hasattr(self, 'grid_coord'):
            # 如果你不在数据增强中进行 GridSampling，
            # 请在 pipeline 中添加以下增强：
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # （根据需要调整 grid_size）
            assert hasattr(self, 'grid_size') and hasattr(self, 'non_empty_coords'), "缺少 grid_size 或 non_empty_coords"
            self.grid_coord = ((self.non_empty_coords - self.non_empty_coords.min(0)[0]) // self.grid_size).int()

        if hasattr(self, 'sparse_shape'):
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.tensor([60,60,36]).tolist()
            self.sparse_shape = sparse_shape

        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat(
                [self.batch_indices.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch_indices[-1].tolist() + 1,
        )

        self.sparse_conv_feat = sparse_conv_feat

    def get_non_empty_voxel_indices(self) -> torch.Tensor:
        """获取非空体素的索引"""
        return torch.arange(len(self.non_empty_features))

    def get_voxel_batch_info(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取每个体素所属的batch和offset信息"""
        return self.batch_indices, self.offset

    def get_point_representation(self) -> Dict:
        """获取点云表示"""
        return {
            "coord": self.non_empty_coords,
            "feat": self.non_empty_features,
            "batch": self.batch_indices,
            "offset": self.offset
        }

    def get_voxel_representation(self) -> Dict:
        """获取体素表示"""
        return {
            "voxel_features": self.voxel_features,
            "non_empty_features": self.non_empty_features,
            "non_empty_coords": self.non_empty_coords,
            "batch_indices": self.batch_indices,
            "offset": self.offset
        }

    def get_serialized_representation(self) -> Dict:
        """获取序列化后的表示"""
        if not hasattr(self, 'serialized_code'):
            raise ValueError("请先调用 serialization() 方法")

        return {
            "serialized_code": self.serialized_code,
            "serialized_order": self.serialized_order,
            "serialized_inverse": self.serialized_inverse,
            "serialized_depth": self.serialized_depth,
            "order": self.order
        }

    def point_to_voxel(self, point_features: torch.Tensor) -> torch.Tensor:
        """
        将点云特征映射回体素网格中对应位置
        Args:
            point_features: 点云特征 [N, C]
        Returns:
            voxel_tensor: [B, C, D, H, W]
        """
        assert point_features.shape[0] == self.non_empty_coords.shape[0], \
            "point_features 和 non_empty_coords 数量不一致"

        B, C, D, H, W = self.batch_size, point_features.shape[1], self.depth, self.height, self.width
        device = point_features.device

        voxel_tensor = torch.zeros((B, C, D, H, W), device=device, dtype=point_features.dtype)

        coords = self.non_empty_coords.long()  # [N, 3]
        batch_ids = self.batch_indices.long()  # [N]

        # 展平目标张量，以便 scatter
        flat_tensor = voxel_tensor.view(B, C, -1)  # [B, C, D*H*W]
        linear_idx = coords[:, 0] * (H * W) + coords[:, 1] * H + coords[:, 2]  # [N]

        #linear_idx = coords[:, 0] * (H * W) + coords[:, 2] * H + coords[:, 1]  # [N]

        # linear_idx = coords[:, 2] * (D * H) + coords[:, 2] * W + coords[:, 1]  # [N]

        # 展开点特征 [N, C] → [B, C, D*H*W]，用 scatter 逐个通道放入对应位置

        flat_tensor[batch_ids, :, linear_idx] = point_features

        voxel_feat = flat_tensor.view(B, C, D, W, H)

        return voxel_feat

    def get_voxel_output(self, output_point) -> torch.Tensor:
        """
        从输出点对象获取体素特征
        Args:
            output_point: 注意力层的输出点对象
        Returns:
            voxel_features: 体素特征 [B, C, D, H, W]
        """
        out = self.point_to_voxel(output_point.feat)
        return out


class SerializedAttention(nn.Module):
    def __init__(
            self,
            channels,
            num_heads,
            patch_size,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            order_index=0,
            enable_rpe=False,
            enable_flash=True,
            upcast_attention=True,
            upcast_softmax=True,
            shift=0
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        self.shift = shift

        if enable_flash:
            assert (
                    enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                    upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                    upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
                pad_key not in point.keys()
                or unpad_key not in point.keys()
                or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)

            # 修复版本兼容性问题
            try:
                # 新版本PyTorch
                bincount_pad = (
                        torch.div(
                            bincount + self.patch_size - 1,
                            self.patch_size,
                            rounding_mode="trunc",
                        )
                        * self.patch_size
                )
            except TypeError:
                # 老版本PyTorch
                bincount_pad = (
                        torch.floor(
                            (bincount + self.patch_size - 1) / self.patch_size
                        )
                        * self.patch_size
                )
            # 确保数据类型为Long
            bincount_pad = bincount_pad.long()

            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))

            # 确保所有offset都是Long类型
            _offset = _offset.long()
            _offset_pad = _offset_pad.long()

            pad = torch.arange(_offset_pad[-1], device=offset.device, dtype=torch.long)
            unpad = torch.arange(_offset[-1], device=offset.device, dtype=torch.long)
            cu_seqlens = []

            for i in range(len(offset)):
                unpad[_offset[i]: _offset[i + 1]] += _offset_pad[i] - _offset[i]

                if bincount[i] != bincount_pad[i]:
                    pad[
                    _offset_pad[i + 1]
                    - self.patch_size
                    + (bincount[i] % self.patch_size): _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size): _offset_pad[i + 1]
                                                           - self.patch_size
                        ]
                pad[_offset_pad[i]: _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )

            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.cat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )

        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def get_padding_and_inverse_shift(self, point, shift=0):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
                pad_key not in point.keys()
                or unpad_key not in point.keys()
                or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)

            # 修复版本兼容性问题
            try:
                # 新版本PyTorch
                bincount_pad = (
                        (torch.div(
                            bincount + self.patch_size - 1,
                            self.patch_size,
                            rounding_mode="trunc",
                        ) + 1)
                        * (self.patch_size)
                )
            except TypeError:
                # 老版本PyTorch
                bincount_pad = (
                        (torch.floor(
                            (bincount + self.patch_size - 1) / self.patch_size
                        ) + 1)
                        * self.patch_size
                )

            # 确保数据类型为Long
            bincount_pad = bincount_pad.long()

            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))

            # 确保所有offset都是Long类型
            _offset = _offset.long()
            _offset_pad = _offset_pad.long()

            pad = torch.arange(_offset_pad[-1], device=offset.device, dtype=torch.long)
            unpad = torch.arange(_offset[-1], device=offset.device, dtype=torch.long)
            cu_seqlens = []

            for i in range(len(offset)):
                unpad[_offset[i]: _offset[i + 1]] += _offset_pad[i] - _offset[i]

                if bincount[i] != bincount_pad[i]:
                    # pad[
                    # _offset_pad[i + 1]
                    # - self.patch_size
                    # + (bincount[i] % self.patch_size): _offset_pad[i + 1]
                    # ] = pad[
                    #     _offset_pad[i + 1]
                    #     - 2 * self.patch_size
                    #     + (bincount[i] % self.patch_size): _offset_pad[i + 1]
                    #                                        - self.patch_size
                    #     ]

                    pad[(_offset_pad[i + 1] - (self.patch_size - shift)): _offset_pad[i + 1]] = \
                        pad[_offset_pad[i]: (_offset_pad[i] + (self.patch_size - shift))]

                    pad[(_offset_pad[i + 1] - self.patch_size):(_offset_pad[i + 1] - self.patch_size + shift)] = \
                        pad[(_offset_pad[i + 1] - 2 * self.patch_size + (bincount[i] % self.patch_size) - shift):(
                                    _offset_pad[i + 1] - 2 * self.patch_size + (bincount[i] % self.patch_size))]

                    pad[(_offset_pad[i + 1] - 2 * self.patch_size + (bincount[i] % self.patch_size) - shift):(
                                _offset_pad[i + 1] - self.patch_size)] = \
                        pad[(_offset_pad[i + 1] - 3 * self.patch_size):(_offset_pad[i + 1] - 3 * self.patch_size + (
                                    self.patch_size - (bincount[i] % self.patch_size) + shift))]

                pad[_offset_pad[i]: _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )

            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.cat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )

        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels
        pad, unpad, cu_seqlens = self.get_padding_and_inverse_shift(point, shift=self.shift)
        order = point.serialized_order.squeeze(0)[pad]
        inverse = unpad[point.serialized_inverse.squeeze(0)]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels=None,
            out_channels=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def example_usage():
    # 生成随机的体素特征张量 [2, 128, 60, 36, 60]
    torch.manual_seed(42)  # 设置随机种子以便复现
    voxel_features = torch.zeros(2, 1, 4, 4, 4)

    voxel_features[:, 0, 0:3, 0:2, 0] = 1.0

    voxel_features = voxel_features.cuda()

    print(f"输入体素特征形状: {voxel_features.shape}")
    print(f"Batch size: {voxel_features.shape[0]}")
    print(f"特征通道数: {voxel_features.shape[1]}")
    print(f"体素网格尺寸: {voxel_features.shape[2:]}")

    # 创建VoxelPoint对象
    voxel_point = VoxelPoint(voxel_features)

    # 获取非空体素索引
    non_empty_indices = voxel_point.get_non_empty_voxel_indices()
    print(f"\n非空体素数量: {len(non_empty_indices)}")

    voxel_point.get_voxel_representation()
    # 获取batch和offset信息
    batch_indices, offset = voxel_point.get_voxel_batch_info()
    print(f"体素batch索引形状: {batch_indices.shape}")
    print(f"Offset: {offset}")

    # 获取点云表示
    point_data = voxel_point.get_point_representation()
    print(f"\n点云坐标形状: {point_data['coord'].shape}")
    print(f"点云特征形状: {point_data['feat'].shape}")
    print(f"点云batch形状: {point_data['batch'].shape}")

    # 统计每个batch的点数
    for i in range(len(offset) - 1):
        start_idx = offset[i]
        end_idx = offset[i + 1]
        batch_points = end_idx - start_idx
        print(f"Batch {i}: {batch_points} 个点")

    # 进行序列化
    print("\n开始序列化...")
    voxel_point.serialization(order=["z"], shuffle_orders=True)
    voxel_point.sparsify(pad=96)

    # 获取序列化结果
    serialized_data = voxel_point.get_serialized_representation()
    print(f"序列化代码形状: {serialized_data['serialized_code'].shape}")
    print(f"序列化顺序形状: {serialized_data['serialized_order'].shape}")
    print(f"序列化逆序形状: {serialized_data['serialized_inverse'].shape}")

    enc = SerializedAttention(
        channels=1,
        num_heads=1,  # 可以根据需要调整
        patch_size=1024,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        enable_flash=False
    ).to('cuda:0')

    # voxel_point.feat = voxel_point.non_empty_features
    voxel_point = enc(voxel_point)

    # voxel_point.feat = voxel_point.feat * 2

    voxel_output = voxel_point.get_voxel_output(voxel_point)
    print(voxel_point.voxel_features)
    print(voxel_output)

    return voxel_output


if __name__ == "__main__":
    # torch.cuda.empty_cache()
    A = 1
    voxel_point = example_usage()