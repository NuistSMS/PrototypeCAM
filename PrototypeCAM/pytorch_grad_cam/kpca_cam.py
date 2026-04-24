from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.covariance_pca import get_2d_projection_kernel_fast


class KPCA_CAM(BaseCAM):
    """
    KPCA-CAM - 使用Kernel PCA的第一主成分作为权重

    版本更新：
    - 使用优化后的KernelPCA实现

    参数：
    - kernel: 核函数类型 ('rbf', 'poly', 'sigmoid', 'linear')
    - gamma: 核函数系数（None则自动计算）
    """

    def __init__(self, model, target_layers,
                 reshape_transform=None, kernel='rbf', gamma=None):
        super(KPCA_CAM, self).__init__(
            model,
            target_layers,
            reshape_transform,
            uses_gradients=False
        )
        self.kernel = kernel
        self.gamma = gamma

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        """
        计算KPCA-CAM

        Args:
            activations: numpy array, shape [batch, channels, height, width]
            其他参数：兼容BaseCAM接口

        Returns:
            cam: numpy array, shape [batch, height, width]
        """
        # 使用优化后的Kernel PCA方法
        return get_2d_projection_kernel_fast(
            activations,
            kernel=self.kernel,
            gamma=self.gamma
        )


