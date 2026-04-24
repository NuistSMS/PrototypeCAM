import argparse
import os
import sys
import warnings
import cv2
import numpy as np
import torch
import pandas as pd
from PIL import Image
from scipy.ndimage import zoom
from datetime import datetime
from collections import defaultdict
from time import time
import tqdm
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    EigenCAM, LayerCAM, GradCAMElementWise,
    KPCA_CAM, FinerCAM, ShapleyCAM, SegXResCAM, PrototypeCAM
)
from pytorch_grad_cam.utils.image import show_cam_on_image

# 导入UNet模型
sys.path.append('./model')
from model.unet import U_Net

# 忽略警告信息
warnings.filterwarnings('ignore')

# 全局缓存变量
_original_image_cache = {}


def parse_int_list(string):
    """解析逗号分隔的整数列表"""
    if string is None or string == '':
        return None
    try:
        return [int(x.strip()) for x in string.split(',')]
    except:
        return None


def parse_float_list(string):
    """解析逗号分隔的浮点数列表"""
    if string is None or string == '': # 是否为None或空字符串
        return None
    try:
        return [float(x.strip()) for x in string.split(',')]
    except:
        return None


def get_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation CAM Analysis')
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='Torch device to use')
    # 输入数据根目录
    parser.add_argument('--data-root', type=str, default='./data/Duke',
                        help='Data root directory containing images, label, and edge_label folders')
    # 模型权重路径
    parser.add_argument('--model-path', type=str, default='./model/Duke.pth',
                        help='Path to model weights')
    # 分析模式选择
    parser.add_argument('--analysis-mode', type=str, default='prediction',
                        choices=['misclassification', 'prediction', 'whynot'],
                        help='Analysis mode: misclassification (错误分类分析), prediction (预测类别分析), or whynot (为什么不是分析)')
    # 目标类别（用于prediction和whynot模式）
    parser.add_argument('--target-category', type=int, default=None,
                        help='Target category for prediction/whynot mode (if None, analyze all predicted classes)')
    # 负样本类别（仅用于whynot模式）
    parser.add_argument('--negative-category', type=int, default=3,
                        help='Negative category for whynot mode (the class to contrast with)')
    parser.add_argument('--num-classes', type=int, default=8,
                        help='总类别数 (默认8)')
    # 批量处理开关
    parser.add_argument('--process-all', action='store_true', default=True,
                        help='Process all images in directory')
    # 尺寸还原开关
    parser.add_argument('--restore-size', action='store_true', default=True,
                        help='Restore output images to original size')
    # CAM方法选择
    parser.add_argument('--method', type=str, default='prototypecam',
                        choices=[
                            'gradcam', 'hirescam', 'gradcam++',
                            'scorecam','eigencam', 'layercam',
                            'gradcamelementwise', 'kpcacam',
                            'shapleycam', 'finercam', 'segxrescam', 'prototypecam'
                        ],
                        help='CAM method')
    parser.add_argument('--xrescam-pool-size', type=int, default=1,
                        help='Seg-XRes-CAM gradient pooling window size '
                             '(1=HiResCAM, >1=XRes-CAM with pooling, default: 1)')
    parser.add_argument('--xrescam-pool-mode', type=str, default='max',
                        choices=['mean', 'max'],
                        help='Seg-XRes-CAM gradient pooling mode (default: max)')
    # FinerCAM 特定参数
    parser.add_argument('--finer-alpha', type=float, default=0.6,
                        help='FinerCAM alpha parameter (default: 0.6, set to 0 for baseline)')
    parser.add_argument('--finer-num-comparisons', type=int, default=3,
                        help='FinerCAM: number of similar classes to compare per pixel (default: 3)')
    # PrototypeCAM 参数
    parser.add_argument('--n-prototypes', type=int, default=6,
                        help='PrototypeCAM的原型数量 (默认1)')
    parser.add_argument('--proto-iterations', type=int, default=20,
                        help='PrototypeCAM的迭代次数 (默认5)')
    parser.add_argument('--proto-gamma', type=float, default=0.9,
                        help='PrototypeCAM的momentum系数 (默认0.9)')
    parser.add_argument('--proto-temperature', type=float, default=0.1,
                        help='PrototypeCAM的温度参数 (默认0.1)')
    parser.add_argument('--proto-use-sinkhorn', action='store_true', default=True,
                        help='PrototypeCAM是否使用Sinkhorn算法')
    parser.add_argument('--proto-normalize', action='store_true', default=True,
                        help='PrototypeCAM是否归一化特征')
    parser.add_argument('--proto-use-contrastive', action='store_true', default=True,
                        help='PrototypeCAM是否使用对比学习')
    parser.add_argument('--proto-contrastive-weight', type=float, default=0.5,
                        help='PrototypeCAM对比学习的负权重α (默认0.5)')
    parser.add_argument('--proto-eta-attract', type=float, default=0.2,
                        help='PrototypeCAM类内吸引力学习率 (默认0.05)')
    parser.add_argument('--proto-eta-repel', type=float, default=2.2,
                        help='PrototypeCAM类间排斥力学习率 (默认0.01)')
    parser.add_argument('--proto-reg-lambda', type=float, default=0.01,
                        help='PrototypeCAM协方差正则化 (默认0.01)')
    # 目标层选择
    parser.add_argument('--target-layer', type=str, default='Up_conv2',
                        choices=['Conv5', 'Up_conv5', 'Up_conv4', 'Up_conv3', 'Up_conv2'],
                        help='Target layer for visualization')
    # 输出根目录
    parser.add_argument('--output-dir', type=str, default='cam_pre_xianyan',
                        help='Output root directory')
    # 模型名称
    parser.add_argument('--model-name', type=str, default='UNet',
                        help='Model name for output directory structure')
    # 边界标签颜色
    parser.add_argument('--edge-color', type=str, default='white',
                        choices=['white', 'black', 'red', 'green', 'blue'],
                        help='Color for edge overlay')
    # 保存图像类型
    parser.add_argument('--save-type', type=int, default=2,
                        choices=[1, 2, 3, 4, 5, 6],
                        help='Save image type: 1=combined (4张拼接), 2=cam_edge (CAM+边界), 3=cam (CAM叠加原图), 4=pure_cam (纯CAM热图), 5=pure_cam_npy (CAM原始数值npy), 6=pure_cam_edge (纯CAM热图+边界)')

    # 保存Excel
    parser.add_argument('--save-excel', action='store_true', default=False,
                        help='Save weights to excel files')
    # 保存分割结果
    parser.add_argument('--save-seg', action='store_true', default=True,
                        help='Save segmentation result images (mask with red edge) to seg folder')
    args = parser.parse_args()

    # 根据save_type确定文件夹名称
    save_type_names = {1: 'combined', 2: 'cam_edge', 3: 'cam', 4: 'pure_cam', 5: 'pure_cam_npy', 6: 'pure_cam_edge'}
    args.save_folder_name = save_type_names[args.save_type]

    # 参数验证
    if args.analysis_mode == 'whynot':
        if args.target_category is None or args.negative_category is None:
            parser.error("whynot mode requires both --target-category and --negative-category")
        if args.target_category == args.negative_category:
            parser.error("target-category and negative-category must be different")

    print(f'\n{"=" * 60}')
    if args.analysis_mode == 'misclassification':
        print(f'Mode: Misclassification Analysis')
    elif args.analysis_mode == 'prediction':
        print(f'Mode: Prediction Analysis')
        if args.target_category is not None:
            print(f'   Target: Class {args.target_category}')
        else:
            print(f'   Target: All predicted classes')
    else:  # whynot
        print(f'Mode: WhyNot Analysis')
        print(f'   Target: Class {args.target_category} (vs Class {args.negative_category})')

    print(f'Device: {args.device} | Method: {args.method} | Layer: {args.target_layer}')
    if args.method == 'prototypecam':
        contrastive_info = " | Contrastive" if args.proto_use_contrastive else ""
        print(
            f'Prototypes: {args.n_prototypes} | Iterations: {args.proto_iterations} | Gamma: {args.proto_gamma}{contrastive_info}')
        if args.proto_use_contrastive:
            print(f'Contrastive Weight: {args.proto_contrastive_weight}')
    print(f'Data Root: {args.data_root}')
    print(f'Edge Color: {args.edge_color}')
    print(f'Save Type: {args.save_type} ({args.save_folder_name})')
    print(f'Save Segmentation: {args.save_seg}')
    print(f'{"=" * 60}\n')
    if args.method == 'finercam':
        print(f'FinerCAM alpha: {args.finer_alpha} | Comparisons: {args.finer_num_comparisons}')

    return args


def load_image(image_path, data_root):
    """加载并预处理灰度图像"""
    global _original_image_cache

    image = np.array(Image.open(image_path).convert('L'))
    x, y = image.shape[0], image.shape[1]

    filename = os.path.basename(image_path)
    _original_image_cache[filename] = {
        'original_image': image.copy(),
        'original_size': (x, y)
    }

    output_size = (224, 224)
    resized_image = zoom(image, (output_size[0] / x, output_size[1] / y), order=3)
    resized_image = np.clip(resized_image, 0, 255).astype(np.uint8)

    return resized_image.astype(np.float32)


def load_label(image_path, data_root):
    """加载对应的 label 图像"""
    filename = os.path.basename(image_path)
    label_path = os.path.join(data_root, 'label', filename)

    if not os.path.exists(label_path):
        return np.zeros((224, 224), dtype=np.uint8)

    label_image = np.array(Image.open(label_path))

    if len(label_image.shape) == 3:
        label_image = label_image[:, :, 0]

    x, y = label_image.shape[0], label_image.shape[1]
    output_size = (224, 224)
    resized_label = zoom(label_image, (output_size[0] / x, output_size[1] / y), order=0)
    resized_label = np.clip(resized_label, 0, 255).astype(np.uint8)

    if filename in _original_image_cache:
        _original_image_cache[filename]['original_label'] = label_image.copy()
    return resized_label


def load_edge_label(image_path, data_root):
    """加载对应的 edge_label 图像（只缓存原始，不resize）"""
    filename = os.path.basename(image_path)
    edge_label_path = os.path.join(data_root, 'edge_label', filename)

    if not os.path.exists(edge_label_path):
        if filename in _original_image_cache:
            original_size = _original_image_cache[filename]['original_size']
            edge_label_image = np.zeros(original_size, dtype=np.uint8)
        else:
            edge_label_image = np.zeros((224, 224), dtype=np.uint8)
    else:
        edge_label_image = np.array(Image.open(edge_label_path))
        if len(edge_label_image.shape) == 3:
            edge_label_image = edge_label_image[:, :, 0]

    if filename in _original_image_cache:
        _original_image_cache[filename]['original_edge_label'] = edge_label_image.copy()

    return None


def preprocess_image_gray(image):
    """将灰度图像预处理为模型输入格式"""
    input_tensor = torch.from_numpy(image).float()
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    return input_tensor


def resize_to_original(image, original_size):
    """将图像缩放回原始尺寸"""
    height, width = original_size
    current_height, current_width = image.shape[:2]
    scale_h = height / current_height
    scale_w = width / current_width
    if len(image.shape) == 2:
        resized_image = zoom(image, (scale_h, scale_w), order=0)
    elif len(image.shape) == 3:
        resized_image = zoom(image, (scale_h, scale_w, 1), order=3)

    return resized_image


def apply_color_map(label_image):
    """将单通道标签图像转换为彩色图像"""
    color_map = {
        0: [0, 0, 0],
        1: [173, 216, 230],
        2: [0, 255, 255],
        3: [0, 128, 0],
        4: [255, 255, 0],
        5: [255, 165, 0],
        6: [255, 0, 0],
        7: [139, 0, 0]
    }

    h, w = label_image.shape
    colored_image = np.zeros((h, w, 3), dtype=np.uint8)
    if label_image.max() > 7:
        label_image_normalized = np.round(label_image / (label_image.max() / 7)).astype(np.uint8)
        label_image_normalized = np.clip(label_image_normalized, 0, 7)
    else:
        label_image_normalized = label_image

    for class_id, color in color_map.items():
        mask = label_image_normalized == class_id
        colored_image[mask] = color

    colored_image_bgr = cv2.cvtColor(colored_image, cv2.COLOR_RGB2BGR)
    return colored_image_bgr


def overlay_mask_with_red_edge(mask, edge_label):
    """
    专门为黑白掩码图叠加红色边界，白色mask覆盖边界
    """
    h, w = mask.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)

    edge_mask = edge_label > 0
    result[edge_mask] = [0, 0, 255]  # 红色边界

    target_mask = mask > 127
    result[target_mask] = [255, 255, 255]  # 白色目标区域覆盖一切

    return result


def overlay_edge_on_image(image_bgr, edge_label, edge_color='white'):
    """
    在图像上叠加边界标签（普通模式：边界覆盖图像）
    """
    color_map = {
        'white': [255, 255, 255],
        'black': [0, 0, 0],
        'red': [0, 0, 255],
        'green': [0, 255, 0],
        'blue': [255, 0, 0]
    }

    edge_bgr_color = color_map.get(edge_color, [255, 255, 255])
    result = image_bgr.copy()
    edge_mask = edge_label > 0
    result[edge_mask] = edge_bgr_color

    return result

class FinerSemanticSegmentationTarget:
    """
    FinerCAM 专用分割 Target - 逐像素动态排序版本
    """

    def __init__(self, main_category: int, mask: np.ndarray,
                 num_comparisons: int = 3, alpha: float = 1.0, device='cpu'):
        self.main_category = main_category
        self.num_comparisons = num_comparisons
        self.alpha = alpha
        self.mask = torch.from_numpy(mask).to(device)

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        C, H, W = model_output.shape

        target_scores = model_output[self.main_category, :, :]
        pixel_comparison_scores = torch.zeros((H, W), device=model_output.device)
        mask_indices = torch.nonzero(self.mask, as_tuple=True)

        for h, w in zip(mask_indices[0], mask_indices[1]):
            pixel_scores = model_output[:, h, w]
            target_score = pixel_scores[self.main_category]

            differences = torch.abs(pixel_scores - target_score)
            differences[self.main_category] = float('inf')

            sorted_indices = torch.argsort(differences)
            top_similar_indices = sorted_indices[:self.num_comparisons]

            similar_scores = pixel_scores[top_similar_indices]
            comparison_mean = similar_scores.mean()

            pixel_comparison_scores[h, w] = comparison_mean

        target_sum = (target_scores * self.mask).sum()
        comparison_sum = (pixel_comparison_scores * self.mask).sum()

        return target_sum - self.alpha * comparison_sum

class UNetOutputWrapper(torch.nn.Module):
    """UNet模型输出包装器"""

    def __init__(self, model):
        super(UNetOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class SemanticSegmentationTarget:
    """语义分割任务的目标类"""

    def __init__(self, category, mask, device='cpu'):
        self.category = category
        self.mask = torch.from_numpy(mask).to(device)

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


def get_target_layer(model, layer_name):
    """根据层名称获取对应的网络层对象"""
    layer_map = {
        'Conv5': model.model.Conv5,
        'Up_conv5': model.model.Up_conv5,
        'Up_conv4': model.model.Up_conv4,
        'Up_conv3': model.model.Up_conv3,
        'Up_conv2': model.model.Up_conv2
    }
    return [layer_map[layer_name]]


def create_output_dir(base_dir, method, model_name, layer_name, mode, save_folder_name, *args):
    """创建输出目录"""
    if mode == 'misclassification':
        true_class, pred_class = args
        output_path = os.path.join(
            base_dir, method, model_name, layer_name,
            f'class_{true_class}', f'misclassified_as_{pred_class}', save_folder_name
        )
    elif mode == 'prediction':
        target_class = args[0]
        output_path = os.path.join(
            base_dir, method, model_name, layer_name,
            f'class_{target_class}', save_folder_name
        )
    else:  # whynot
        target_class, negative_class = args
        output_path = os.path.join(
            base_dir, method, model_name, layer_name,
            f'class_{target_class}_vs_{negative_class}', save_folder_name
        )

    os.makedirs(output_path, exist_ok=True)
    return output_path


def create_seg_output_dir(base_dir, model_name, layer_name, mode, *args):
    """创建分割结果输出目录"""
    if mode == 'misclassification':
        true_class, pred_class = args
        output_path = os.path.join(
            base_dir, 'seg', model_name, layer_name,
            f'class_{true_class}', f'misclassified_as_{pred_class}'
        )
    elif mode == 'prediction':
        target_class = args[0]
        output_path = os.path.join(
            base_dir, 'seg', model_name, layer_name,
            f'class_{target_class}'
        )
    else:  # whynot
        target_class, negative_class = args
        output_path = os.path.join(
            base_dir, 'seg', model_name, layer_name,
            f'class_{target_class}_vs_{negative_class}'
        )

    os.makedirs(output_path, exist_ok=True)
    return output_path


def get_misclassification_mask(label, prediction, true_class, pred_class):
    """获取特定错误分类的掩码"""
    mask = ((label == true_class) & (prediction == pred_class)).astype(np.float32)
    pixel_count = int(mask.sum())
    return mask, pixel_count


def get_prediction_mask(prediction, target_class):
    """获取预测类别的掩码"""
    mask = (prediction == target_class).astype(np.float32)
    pixel_count = int(mask.sum())
    return mask, pixel_count


class WeightsAggregator:
    """权重聚合器 - 按空间位置(224x224)保存权重"""

    def __init__(self):
        self.weights_dict = defaultdict(dict)
        self.pos_weights_dict = defaultdict(dict)
        self.neg_weights_dict = defaultdict(dict)
        self.combined_weights_dict = defaultdict(dict)

    def add_weights(self, key, image_name, weights):
        """添加权重- 保存为空间矩阵"""
        if isinstance(weights, np.ndarray):
            self.weights_dict[key][image_name] = weights.copy()
        elif isinstance(weights, torch.Tensor):
            self.weights_dict[key][image_name] = weights.cpu().detach().numpy().copy()
        else:
            self.weights_dict[key][image_name] = np.array(weights).copy()

    def add_prototype_weights(self, key, image_name, pos_weights, neg_weights, combined_weights):
        """添加prototypecam的三种权重（224x224空间矩阵）"""
        if isinstance(pos_weights, np.ndarray):
            self.pos_weights_dict[key][image_name] = pos_weights.copy()
        elif isinstance(pos_weights, torch.Tensor):
            self.pos_weights_dict[key][image_name] = pos_weights.cpu().detach().numpy().copy()
        else:
            self.pos_weights_dict[key][image_name] = np.array(pos_weights).copy()

        if isinstance(neg_weights, np.ndarray):
            self.neg_weights_dict[key][image_name] = neg_weights.copy()
        elif isinstance(neg_weights, torch.Tensor):
            self.neg_weights_dict[key][image_name] = neg_weights.cpu().detach().numpy().copy()
        else:
            self.neg_weights_dict[key][image_name] = np.array(neg_weights).copy()

        if isinstance(combined_weights, np.ndarray):
            self.combined_weights_dict[key][image_name] = combined_weights.copy()
        elif isinstance(combined_weights, torch.Tensor):
            self.combined_weights_dict[key][image_name] = combined_weights.cpu().detach().numpy().copy()
        else:
            self.combined_weights_dict[key][image_name] = np.array(combined_weights).copy()

    def _save_spatial_weights_to_excel(self, weights_data, excel_path, sheet_name='Weights'):
        """将空间权重(224x224)保存到Excel"""
        if len(weights_data) == 0:
            return

        first_weights = list(weights_data.values())[0]
        if len(first_weights.shape) != 2:
            print(f"Warning: weights shape is not 2D: {first_weights.shape}")
            return

        height, width = first_weights.shape

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for image_name, weights_matrix in sorted(weights_data.items()):
                df = pd.DataFrame(
                    weights_matrix,
                    index=[f'row_{i}' for i in range(height)],
                    columns=[f'col_{j}' for j in range(width)]
                )
                sheet_name_img = image_name[:31]
                df.to_excel(writer, sheet_name=sheet_name_img)

    def save_all_to_excel(self, base_dir, method, model_name, layer_name, mode):
        """保存所有权重到Excel文件（按224x224空间位置）"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for key, images_weights in self.weights_dict.items():
            if len(images_weights) == 0:
                continue

            if mode == 'misclassification':
                true_class, pred_class = key
                excel_dir = os.path.join(
                    base_dir, method, model_name, layer_name,
                    f'class_{true_class}', f'misclassified_as_{pred_class}'
                )
            elif mode == 'prediction':
                target_class = key[0] if isinstance(key, tuple) else key
                excel_dir = os.path.join(
                    base_dir, method, model_name, layer_name, f'class_{target_class}'
                )
            else:  # whynot
                target_class, negative_class = key
                excel_dir = os.path.join(
                    base_dir, method, model_name, layer_name,
                    f'class_{target_class}_vs_{negative_class}'
                )

            os.makedirs(excel_dir, exist_ok=True)
            excel_filename = f'weights_spatial_{timestamp}.xlsx'
            excel_path = os.path.join(excel_dir, excel_filename)
            self._save_spatial_weights_to_excel(images_weights, excel_path)

        for key in set(self.pos_weights_dict.keys()) | set(self.neg_weights_dict.keys()) | set(
                self.combined_weights_dict.keys()):
            if mode == 'misclassification':
                true_class, pred_class = key
                excel_dir = os.path.join(
                    base_dir, method, model_name, layer_name,
                    f'class_{true_class}', f'misclassified_as_{pred_class}'
                )
            elif mode == 'prediction':
                target_class = key[0] if isinstance(key, tuple) else key
                excel_dir = os.path.join(
                    base_dir, method, model_name, layer_name, f'class_{target_class}'
                )
            else:  # whynot
                target_class, negative_class = key
                excel_dir = os.path.join(
                    base_dir, method, model_name, layer_name,
                    f'class_{target_class}_vs_{negative_class}'
                )

            os.makedirs(excel_dir, exist_ok=True)

            if key in self.pos_weights_dict and len(self.pos_weights_dict[key]) > 0:
                excel_filename = f'pos_weights_spatial_{timestamp}.xlsx'
                excel_path = os.path.join(excel_dir, excel_filename)
                self._save_spatial_weights_to_excel(self.pos_weights_dict[key], excel_path)

            if key in self.neg_weights_dict and len(self.neg_weights_dict[key]) > 0:
                excel_filename = f'neg_weights_spatial_{timestamp}.xlsx'
                excel_path = os.path.join(excel_dir, excel_filename)
                self._save_spatial_weights_to_excel(self.neg_weights_dict[key], excel_path)

            if key in self.combined_weights_dict and len(self.combined_weights_dict[key]) > 0:
                excel_filename = f'combined_weights_spatial_{timestamp}.xlsx'
                excel_path = os.path.join(excel_dir, excel_filename)
                self._save_spatial_weights_to_excel(self.combined_weights_dict[key], excel_path)


def compute_cam_and_visualize(image_resized, input_tensor, model, target_layers,
                              target_class, mask, device, args, cam_algorithm,
                              label_resized=None, pred_mask=None, true_class=None,
                              negative_class=None):
    rgb_img = np.stack([image_resized / 255.0] * 3, axis=-1).astype(np.float32)

    if args.method == 'finercam':
        targets = [FinerSemanticSegmentationTarget(
            main_category=target_class,
            num_comparisons=args.finer_num_comparisons,
            mask=mask,
            alpha=args.finer_alpha,
            device=device
        )]
    else:
        targets = [SemanticSegmentationTarget(target_class, mask, device)]

    cam_weights = None
    pos_weights = None
    neg_weights = None
    combined_weights = None

    if args.method == 'prototypecam':
        negative_mask = None
        pred_mask_np = None

        if args.proto_use_contrastive:
            if args.analysis_mode == 'misclassification':
                if label_resized is not None and pred_mask is not None and true_class is not None:
                    negative_mask = ((label_resized == true_class) &
                                     (pred_mask == true_class)).astype(np.float32)

            elif args.analysis_mode in ['prediction', 'whynot']:
                if pred_mask is not None:
                    pred_mask_np = pred_mask.astype(np.float32)

        cam = cam_algorithm(
            model=model,
            target_layers=target_layers,
            n_prototypes=args.n_prototypes,
            n_iterations=args.proto_iterations,
            gamma=args.proto_gamma,
            temperature=args.proto_temperature,
            use_sinkhorn=args.proto_use_sinkhorn,
            normalize=args.proto_normalize,
            use_contrastive=args.proto_use_contrastive,
            contrastive_weight=args.proto_contrastive_weight,
            num_classes=args.num_classes,
            eta_attract=args.proto_eta_attract,
            eta_repel=args.proto_eta_repel,
            reg_lambda=args.proto_reg_lambda
        )

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
            negative_mask=negative_mask,
            pred_mask_np=pred_mask_np,
            target_class=target_class,
            analysis_mode=args.analysis_mode,
            negative_class=negative_class
        )
        grayscale_cam = grayscale_cam[0, :]

        if hasattr(cam, 'last_weights'):
            cam_weights = cam.last_weights

        if hasattr(cam, 'last_pos_weights') and cam.last_pos_weights is not None:
            pos_weights = cam.last_pos_weights
        if hasattr(cam, 'last_neg_weights') and cam.last_neg_weights is not None:
            neg_weights = cam.last_neg_weights
        if hasattr(cam, 'last_combined_weights') and cam.last_combined_weights is not None:
            combined_weights = cam.last_combined_weights

        del cam
    elif args.method == 'segxrescam':
        cam = cam_algorithm(
            model=model,
            target_layers=target_layers,
            pool_size=args.xrescam_pool_size,
            pool_mode=args.xrescam_pool_mode
        )
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        if hasattr(cam, 'last_weights'):
            cam_weights = cam.last_weights
        del cam
    elif args.method == 'finercam':
        cam = cam_algorithm(model=model, target_layers=target_layers)
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
            eigen_smooth=getattr(args, 'eigen_smooth', False)
        )
        grayscale_cam = grayscale_cam[0, :]

        if hasattr(cam, 'last_weights'):
            cam_weights = cam.last_weights
    else:
        cam = cam_algorithm(model=model, target_layers=target_layers)
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        if hasattr(cam, 'last_weights'):
            cam_weights = cam.last_weights
        del cam

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return cam_weights, cam_image, pos_weights, neg_weights, combined_weights, grayscale_cam


def process_single_image(image_path, model, target_layers, device, args,
                         cam_algorithm, weights_aggregator):
    """处理单张图像（支持whynot模式）"""

    start_time = time()

    image_resized = load_image(image_path, args.data_root)
    label_resized = load_label(image_path, args.data_root)
    load_edge_label(image_path, args.data_root)
    input_tensor = preprocess_image_gray(image_resized).to(device)

    filename = os.path.basename(image_path)
    filename_without_ext = os.path.splitext(filename)[0]
    original_size = _original_image_cache[filename]['original_size']

    with torch.no_grad():
        output = model(input_tensor)
    pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    pattern_count = 0

    if args.analysis_mode == 'misclassification':
        tasks = []
        for true_class in range(args.num_classes):
            for pred_class in range(args.num_classes):
                if true_class == pred_class:
                    continue
                mask, pixel_count = get_misclassification_mask(
                    label_resized, pred_mask, true_class, pred_class
                )
                if pixel_count > 0:
                    tasks.append(('mis', true_class, pred_class, None, mask, pixel_count))

    elif args.analysis_mode == 'prediction':
        if args.target_category is not None:
            target_classes = [args.target_category]
        else:
            target_classes = np.unique(pred_mask).tolist()

        tasks = []
        for target_class in target_classes:
            mask, pixel_count = get_prediction_mask(pred_mask, target_class)
            if pixel_count > 0:
                tasks.append(('pred', target_class, None, None, mask, pixel_count))

    else:  # whynot mode
        target_class = args.target_category
        negative_class = args.negative_category

        mask, pixel_count = get_prediction_mask(pred_mask, target_class)

        tasks = []
        if pixel_count > 0:
            tasks.append(('whynot', target_class, None, negative_class, mask, pixel_count))

    for task in tasks:
        pattern_count += 1

        if task[0] == 'mis':  # misclassification
            _, true_class, pred_class, _, mask, pixel_count = task
            target_class = pred_class
            negative_class_for_cam = None

            output_dir = create_output_dir(
                args.output_dir, args.method, args.model_name,
                args.target_layer, 'misclassification', args.save_folder_name,
                true_class, pred_class
            )

            if args.save_seg:
                seg_output_dir = create_seg_output_dir(
                    args.output_dir, args.model_name,
                    args.target_layer, 'misclassification',
                    true_class, pred_class
                )

            key = (true_class, pred_class)

        elif task[0] == 'pred':  # prediction
            _, target_class, _, _, mask, pixel_count = task
            negative_class_for_cam = None

            output_dir = create_output_dir(
                args.output_dir, args.method, args.model_name,
                args.target_layer, 'prediction', args.save_folder_name,
                target_class
            )

            if args.save_seg:
                seg_output_dir = create_seg_output_dir(
                    args.output_dir, args.model_name,
                    args.target_layer, 'prediction',
                    target_class
                )

            key = (target_class,)

        else:  # whynot
            _, target_class, _, negative_class, mask, pixel_count = task
            negative_class_for_cam = negative_class

            output_dir = create_output_dir(
                args.output_dir, args.method, args.model_name,
                args.target_layer, 'whynot', args.save_folder_name,
                target_class, negative_class
            )

            if args.save_seg:
                seg_output_dir = create_seg_output_dir(
                    args.output_dir, args.model_name,
                    args.target_layer, 'whynot',
                    target_class, negative_class
                )

            key = (target_class, negative_class)

        cam_weights, cam_image, pos_weights, neg_weights, combined_weights, grayscale_cam = compute_cam_and_visualize(
            image_resized, input_tensor, model, target_layers,
            target_class, mask, device, args, cam_algorithm,
            label_resized=label_resized,
            pred_mask=pred_mask,
            true_class=true_class if task[0] == 'mis' else None,
            negative_class=negative_class_for_cam
        )

        if args.method == 'prototypecam':
            if pos_weights is not None and neg_weights is not None and combined_weights is not None:
                weights_aggregator.add_prototype_weights(key, filename_without_ext,
                                                         pos_weights, neg_weights, combined_weights)
        else:
            if cam_weights is not None:
                weights_aggregator.add_weights(key, filename_without_ext, cam_weights)

        if args.save_seg:
            mask_uint8 = (mask * 255).astype(np.uint8)
            original_edge_label = _original_image_cache[filename]['original_edge_label']

            if args.restore_size:
                mask_uint8_resized = zoom(mask_uint8,
                                          (original_size[0] / mask_uint8.shape[0],
                                           original_size[1] / mask_uint8.shape[1]),
                                          order=0).astype(np.uint8)
                edge_label_for_seg = original_edge_label
            else:
                edge_label_for_seg = zoom(original_edge_label,
                                          (224 / original_edge_label.shape[0],
                                           224 / original_edge_label.shape[1]),
                                          order=0).astype(np.uint8)
                mask_uint8_resized = mask_uint8

            seg_result_image = overlay_mask_with_red_edge(mask_uint8_resized, edge_label_for_seg)

            seg_output_filename = f'{filename_without_ext}.png'
            seg_output_path = os.path.join(seg_output_dir, seg_output_filename)
            cv2.imwrite(seg_output_path, seg_result_image)

        cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        original_edge_label = _original_image_cache[filename]['original_edge_label']

        if args.save_type == 1:
            # 类型1：四张图拼接
            mask_uint8 = (mask * 255).astype(np.uint8)

            original_rgb = cv2.cvtColor(image_resized.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            label_colored = apply_color_map(label_resized)

            if args.restore_size:
                mask_uint8_resized = zoom(mask_uint8,
                                          (original_size[0] / mask_uint8.shape[0],
                                           original_size[1] / mask_uint8.shape[1]),
                                          order=0).astype(np.uint8)

                original_rgb = resize_to_original(original_rgb, original_size).astype(np.uint8)
                cam_image_bgr = resize_to_original(cam_image_bgr, original_size).astype(np.uint8)

                if filename in _original_image_cache and 'original_label' in _original_image_cache[filename]:
                    original_label = _original_image_cache[filename]['original_label']
                    label_colored = apply_color_map(original_label)
                else:
                    label_colored = resize_to_original(label_colored, original_size).astype(np.uint8)

                edge_label_to_use = original_edge_label
            else:
                edge_label_to_use = zoom(original_edge_label,
                                         (224 / original_edge_label.shape[0],
                                          224 / original_edge_label.shape[1]),
                                         order=0).astype(np.uint8)
                mask_uint8_resized = mask_uint8

            cam_with_edge = overlay_edge_on_image(cam_image_bgr, edge_label_to_use, args.edge_color)
            mask_with_red_edge = overlay_mask_with_red_edge(mask_uint8_resized, edge_label_to_use)

            top_row = np.hstack([original_rgb, cam_with_edge])
            bottom_row = np.hstack([mask_with_red_edge, label_colored])
            output_image = np.vstack([top_row, bottom_row])

        elif args.save_type == 2:
            # 类型2：CAM图+边界标签
            if args.restore_size:
                cam_image_bgr = resize_to_original(cam_image_bgr, original_size).astype(np.uint8)
                edge_label_to_use = original_edge_label
            else:
                edge_label_to_use = zoom(original_edge_label,
                                         (224 / original_edge_label.shape[0],
                                          224 / original_edge_label.shape[1]),
                                         order=0).astype(np.uint8)

            output_image = overlay_edge_on_image(cam_image_bgr, edge_label_to_use, args.edge_color)

        elif args.save_type == 3:
            # 类型3：纯CAM图，不带边界标签
            if args.restore_size:
                cam_image_bgr = resize_to_original(cam_image_bgr, original_size).astype(np.uint8)

            output_image = cam_image_bgr

        elif args.save_type == 4:
            # 类型4：纯CAM热图，不叠加原图
            import matplotlib.pyplot as plt

            colored_cam = plt.cm.jet(grayscale_cam)[:, :, :3]
            colored_cam = (colored_cam * 255).astype(np.uint8)
            colored_cam_bgr = cv2.cvtColor(colored_cam, cv2.COLOR_RGB2BGR)

            if args.restore_size:
                output_image = resize_to_original(colored_cam_bgr, original_size).astype(np.uint8)
            else:
                output_image = colored_cam_bgr

        elif args.save_type == 5:
            # 类型5：纯CAM原始数值，保存为npy格式
            if args.restore_size:
                cam_data = resize_to_original(grayscale_cam, original_size)
            else:
                cam_data = grayscale_cam

            output_filename = f'{filename_without_ext}.npy'
            output_path = os.path.join(output_dir, output_filename)
            np.save(output_path, cam_data)
            continue  # 跳过后面的图像保存步骤

        elif args.save_type == 6:
            # 类型6：纯CAM热图（不叠加原图） + 边界标签
            import matplotlib.pyplot as plt

            # 将 grayscale_cam (0-1) 转换为 jet 彩色热图
            colored_cam = plt.cm.jet(grayscale_cam)[:, :, :3]  # 只取RGB，去掉alpha通道
            colored_cam = (colored_cam * 255).astype(np.uint8)
            colored_cam_bgr = cv2.cvtColor(colored_cam, cv2.COLOR_RGB2BGR)

            if args.restore_size:
                colored_cam_bgr = resize_to_original(colored_cam_bgr, original_size).astype(np.uint8)
                edge_label_to_use = original_edge_label
            else:
                edge_label_to_use = zoom(original_edge_label,
                                         (224 / original_edge_label.shape[0],
                                          224 / original_edge_label.shape[1]),
                                         order=0).astype(np.uint8)

            # 在纯CAM热图上叠加边界
            output_image = overlay_edge_on_image(colored_cam_bgr, edge_label_to_use, args.edge_color)

        # 保存图像
        output_filename = f'{filename_without_ext}.png'
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, output_image)

    elapsed_time = time() - start_time

    return pattern_count, elapsed_time

if __name__ == '__main__':
    args = get_args()

    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise,
        "kpcacam": KPCA_CAM,
        "finercam": FinerCAM,
        "shapleycam": ShapleyCAM,
        "segxrescam": SegXResCAM,
        "prototypecam": PrototypeCAM
    }

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    model = U_Net(in_ch=3, out_ch=8)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    model = UNetOutputWrapper(model)
    target_layers = get_target_layer(model, args.target_layer)
    cam_algorithm = methods[args.method]

    image_folder = os.path.join(args.data_root, 'images')
    if not os.path.exists(image_folder):
        print(f"Error: Image folder not found at {image_folder}")
        exit(1)

    image_files = [f for f in os.listdir(image_folder)
                   if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    if not image_files:
        print(f"Error: No image files found in {image_folder}")
        exit(1)

    if args.process_all:
        image_paths = [os.path.join(image_folder, f) for f in image_files]
    else:
        target_image = 'patient_10_B_scan_7.png'
        image_paths = [os.path.join(image_folder, target_image)]

    weights_aggregator = WeightsAggregator()

    total_patterns = 0
    total_time = 0

    pbar = tqdm.tqdm(image_paths, desc="Progress", ncols=80)

    for image_path in pbar:
        pattern_count, elapsed_time = process_single_image(
            image_path, model, target_layers, device, args,
            cam_algorithm, weights_aggregator
        )
        total_patterns += pattern_count
        total_time += elapsed_time

        pbar.write(f"✓ {os.path.basename(image_path)}: {pattern_count} patterns, {elapsed_time:.1f}s")

    if args.save_excel:
        weights_aggregator.save_all_to_excel(
            args.output_dir, args.method, args.model_name, args.target_layer,
            args.analysis_mode
        )

    avg_time = total_time / len(image_paths) if len(image_paths) > 0 else 0
    print(f"Images: {len(image_paths)} | Patterns: {total_patterns}")
    print(f"Time: {total_time:.1f}s (avg: {avg_time:.1f}s/image)")
    print(f"Output: {args.output_dir}")