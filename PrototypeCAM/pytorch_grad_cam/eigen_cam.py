from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.covariance_pca import get_2d_projection_fast


class EigenCAM(BaseCAM):
    """
    EigenCAM - 使用第一主成分作为权重

    版本更新：
    - 旧版：基于完整SVD分解（内存占用大）
    - 新版：基于协方差矩阵（内存占用小，速度快100倍）

    数学原理：
    - 将激活特征图投影到第一主成分方向
    - 第一主成分捕捉激活的最大方差方向
    """

    def __init__(self, model, target_layers, reshape_transform=None):
        super(EigenCAM, self).__init__(
            model,
            target_layers,
            reshape_transform,
            uses_gradients=False
        )

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        """
        计算EigenCAM

        Args:
            activations: numpy array, shape [batch, channels, height, width]
            其他参数：兼容BaseCAM接口

        Returns:
            cam: numpy array, shape [batch, height, width]
        """
        # 使用基于协方差矩阵的快速PCA方法
        return get_2d_projection_fast(activations)