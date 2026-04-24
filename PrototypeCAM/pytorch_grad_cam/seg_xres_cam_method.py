"""
Seg-XRes-CAM: Explaining Spatially Local Regions in Image Segmentation
Based on CVPRW 2023 paper by Hasany et al.

Seg-XRes-CAM (Eq.5):
  L^c_{XRes-CAM} = ReLU( Σ_k Up[Pool[∂Y^c/∂A^k]] ⊙ A^k )

  - pool_size=1: equivalent to Seg-HiResCAM (Eq.4), no pooling
  - pool_size>1: Seg-XRes-CAM with coarser gradient pooling

Note: Seg-Grad-CAM (Eq.3) already等价于现有的 GradCAM + SemanticSegmentationTarget，
      不需要单独实现。
"""

import numpy as np
import cv2
import skimage.measure
import skimage.transform


def _save_gradient(tensor, gradients_list):
    """Register a backward hook on the tensor to capture gradients."""
    tensor.register_hook(lambda grad: gradients_list.append(grad))


class SegXResCAM:
    """
    Seg-XRes-CAM implementation (Hasany et al., CVPRW 2023).

    Computes CAM by:
    1. Forward pass → capture activations at target layer
    2. Backward pass from masked target region → capture gradients
    3. (Optional) Pool gradients with window of pool_size × pool_size, then upsample
    4. Element-wise (Hadamard) product of gradients and activations
    5. Sum over channels → localization map
    6. ReLU → retain positive contributions only

    Parameters:
        model: The neural network model (wrapped if needed)
        target_layers: List of target layers to extract features from
        pool_size: Pooling window size for gradient matrix
                   - 1: No pooling (equivalent to Seg-HiResCAM, Eq.4)
                   - >1: Seg-XRes-CAM with h×w pooling window (Eq.5)
        pool_mode: Pooling function - 'mean' or 'max'
    """

    def __init__(self, model, target_layers, pool_size=1, pool_mode='max'):
        self.model = model
        self.target_layers = target_layers
        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.activations = []
        self.gradients = []
        self.last_weights = None
        self.batch_size = 32  # 兼容属性

    def __call__(self, input_tensor, targets, **kwargs):
        self.activations = []
        self.gradients = []

        # 在第一个目标层上注册钩子
        target_layer = self.target_layers[0]

        handle_fwd = target_layer.register_forward_hook(
            lambda module, inp, out: self.activations.append(out)
        )
        handle_grad = target_layer.register_forward_hook(
            lambda module, inp, out: _save_gradient(out, self.gradients)
        )

        # 前向传播
        self.model.zero_grad()
        output = self.model(input_tensor)

        # 通过 targets 计算 loss (SemanticSegmentationTarget)
        loss = sum([target(output[0]) for target in targets])
        loss.backward()

        # 提取激活和梯度: shape (C, H_feat, W_feat)
        activations = self.activations[0][0].detach().cpu().numpy()
        gradients = self.gradients[0][0].detach().cpu().numpy()

        # 对梯度做池化 (Eq.5 中的 Pool + Up)
        if self.pool_size is not None and self.pool_size > 1:
            if self.pool_mode == 'mean':
                pool_fn = np.mean
            else:  # 'max'
                pool_fn = np.max

            # Block reduce: (C, H, W) → (C, H//pool, W//pool)
            pooled = skimage.measure.block_reduce(
                gradients, (1, self.pool_size, self.pool_size), pool_fn
            )
            # 转置后上采样回原始特征图大小
            pooled = np.transpose(pooled, (1, 2, 0))
            gradients = skimage.transform.resize(
                pooled,
                (gradients.shape[1], gradients.shape[2]),
                order=0,  # 最近邻插值
                preserve_range=True
            )
            gradients = np.transpose(gradients, (2, 0, 1))

        # Hadamard 逐元素乘积后求和 (Eq.5 核心)
        grayscale_cam = (gradients * activations).sum(axis=0)

        # ReLU
        grayscale_cam = np.maximum(grayscale_cam, 0)

        # 缩放到输入空间尺寸
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        grayscale_cam = cv2.resize(
            grayscale_cam.astype(np.float32), (input_w, input_h)
        )

        # 归一化到 [0, 1]
        max_val, min_val = grayscale_cam.max(), grayscale_cam.min()
        if max_val - min_val > 1e-8:
            grayscale_cam = (grayscale_cam - min_val) / (max_val - min_val)
        else:
            grayscale_cam = np.zeros_like(grayscale_cam)

        self.last_weights = grayscale_cam.copy()

        # 清除钩子
        handle_fwd.remove()
        handle_grad.remove()

        return grayscale_cam[np.newaxis, :]  # (1, H, W)

    def __del__(self):
        pass