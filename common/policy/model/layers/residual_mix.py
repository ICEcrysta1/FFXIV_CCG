"""四路残差流的动态混合与 Sinkhorn 约束。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class HyperConnections(nn.Module):
    """每个子层独立生成 pre/post 和近似双随机的残差混合矩阵。"""

    norm_eps = 1e-20

    def __init__(
        self,
        d_model: int,
        *,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        mix_dim = (2 + hc_mult) * hc_mult
        self.fn = nn.Parameter(torch.empty(mix_dim, hc_mult * d_model, dtype=torch.float32))
        self.base = nn.Parameter(torch.empty(mix_dim, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """从近似平均读入、单位残差传递开始，并保留动态投影的梯度。"""
        nn.init.xavier_uniform_(self.fn)
        nn.init.constant_(self.scale, 0.01)
        with torch.no_grad():
            self.base.zero_()
            self.base[:self.hc_mult].fill_(-math.log(self.hc_mult - 1))
            self.base[2 * self.hc_mult:].view(self.hc_mult, self.hc_mult).copy_(
                4.0 * torch.eye(self.hc_mult, device=self.base.device, dtype=self.base.dtype),
            )

    def expand(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """复制输入流，首个子层只读取第一路。"""
        streams = tokens.unsqueeze(-2).expand(-1, -1, self.hc_mult, -1)
        # 首层是固定选择器；先用 bool 构造，避免 ONNX 物化 FP32 scatter 更新块。
        pre_mix = tokens.new_zeros((*tokens.shape[:2], self.hc_mult), dtype=torch.bool)
        pre_mix[..., 0] = True
        return streams, pre_mix.float()

    def mixes(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CUDA 训练融合系数生成与反向重算，避免保存 Sinkhorn 中间量。"""
        if streams.is_cuda and self.training and torch.is_grad_enabled() and not torch.onnx.is_in_onnx_export():
            # 算子注册在捕获前完成，CPU 和导出路径不加载 Triton。
            from . import hyper_connections_cuda  # noqa: F401

        return self._recompute(self._mixes, self._compiled_mixes, streams)

    def _recompute(self, operation, compiled_operation, *inputs):
        if self.training and torch.is_grad_enabled() and not torch.onnx.is_in_onnx_export():
            if inputs[0].is_cuda:
                # 编译入口也关闭 AMP，确保 AOT 反向的投影矩阵乘继续使用 FP32。
                with torch.autocast(device_type=inputs[0].device.type, enabled=False):
                    return compiled_operation(*inputs)
            return checkpoint(operation, *inputs, use_reentrant=False, preserve_rng_state=False)
        return operation(*inputs)

    @torch.compile(fullgraph=True, dynamic=True)
    def _compiled_mixes(self, streams: torch.Tensor):
        # 编译范围包含重算，反向可将重算与求导融合，投影参数仍参与 autograd。
        return checkpoint(self._mixes, streams, True, use_reentrant=False, preserve_rng_state=False)

    def _mixes(self, streams: torch.Tensor, fused: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """在 FP32 中生成系数，Sinkhorn 顺序与参考实现保持一致。"""
        with torch.autocast(device_type=streams.device.type, enabled=False):
            flat = streams.flatten(-2).float()
            inv_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + self.norm_eps)
            mixes = F.linear(flat, self.fn.float()) * inv_rms
            if fused:
                from .hyper_connections_cuda import hc_coefficients

                return hc_coefficients(mixes, self.scale, self.base, self.hc_mult,
                                       self.hc_sinkhorn_iters, self.hc_eps)
            mult = self.hc_mult
            pre = torch.sigmoid(mixes[..., :mult] * self.scale[0] + self.base[:mult]) + self.hc_eps
            post = 2 * torch.sigmoid(mixes[..., mult:2 * mult] * self.scale[1] + self.base[mult:2 * mult])
            logits = (mixes[..., 2 * mult:] * self.scale[2] + self.base[2 * mult:]).unflatten(-1, (mult, mult))
            comb = logits.softmax(dim=-1) + self.hc_eps
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
            for _ in range(self.hc_sinkhorn_iters - 1):
                comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
                comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        return pre, post, comb

    def collapse(self, streams: torch.Tensor, pre_mix: torch.Tensor, norm=None) -> torch.Tensor:
        """融合读流与可选归一化，重算时只保存原始流和 pre。"""
        # CUDA AMP 的 RMSNorm 会提升到 FP32；保留读流先转回原 dtype 的舍入。
        norm_dtype = torch.float32 if norm is not None and streams.is_cuda and torch.is_autocast_enabled("cuda") else streams.dtype
        return self._recompute(self._collapse, self._compiled_collapse, streams, pre_mix, norm, norm_dtype)

    # 避免 joint graph 把 FP32→低精度→FP32 的必要舍入当作冗余转换删除。
    @torch.compile(fullgraph=True, dynamic=True,
                   options={"emulate_precision_casts": True, "use_joint_graph_passes": False})
    def _compiled_collapse(self, streams: torch.Tensor, pre_mix: torch.Tensor, norm, norm_dtype):
        return checkpoint(self._collapse, streams, pre_mix, norm, norm_dtype, True,
                          use_reentrant=False, preserve_rng_state=False)

    def _collapse(self, streams: torch.Tensor, pre_mix: torch.Tensor, norm=None, norm_dtype=None,
                  fused: bool = False) -> torch.Tensor:
        """使用上一子层传出的 pre 加权读取残差流。"""
        with torch.autocast(device_type=streams.device.type, enabled=False):
            if fused:
                # 小流轴展开成逐元素乘加，读流可直接进入同一核的特征维归约。
                output = sum(weight.unsqueeze(-1) * stream.float() for weight, stream in
                             zip(pre_mix.unbind(-1), streams.unbind(-2), strict=True))
            else:
                output = (pre_mix.unsqueeze(-1) * streams.float()).sum(dim=-2)
        output = output.to(streams.dtype)
        if norm is None:
            return output
        output = output.to(norm_dtype)
        if fused:
            # 原生融合 RMSNorm 会物化输入；展开同一公式才能与加权读流融合。
            values = output if output.dtype == torch.float64 else output.float()
            eps = norm.eps if norm.eps is not None else torch.finfo(values.dtype).eps
            values = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
            if norm.weight is not None:
                values = values * norm.weight
            return values.to(output.dtype)
        return norm(output)

    def combine(self, output: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        """重算 FP32 混合副本，训练只保存原始流、子层输出和系数。"""
        return self._recompute(self._combine, self._compiled_combine, output, residual, post, comb)

    @torch.compile(fullgraph=True, dynamic=True)
    def _compiled_combine(self, output: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        return checkpoint(self._combine, output, residual, post, comb, True, use_reentrant=False, preserve_rng_state=False)

    def _combine(
        self, output: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor,
        fused: bool = False,
    ) -> torch.Tensor:
        """按源流到目标流混合旧残差，并分配当前子层输出。"""
        with torch.autocast(device_type=output.device.type, enabled=False):
            if fused:
                # 编译器融合乘加、归约和 cast，不物化广播的五维 FP32 乘积。
                mixed = (comb.unsqueeze(-1) * residual.float().unsqueeze(-2)).sum(dim=-3)
            else:
                mixed = torch.matmul(comb.transpose(-1, -2), residual.float())
            mixed = mixed + post.unsqueeze(-1) * output.float().unsqueeze(-2)
        return mixed.to(output.dtype)
