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

# Import UNet model
sys.path.append('./model')
from model.unet import U_Net

# Ignore warnings
warnings.filterwarnings('ignore')

# Global cache variable
_original_image_cache = {}


def parse_int_list(string):
    """Parse comma-separated integer list"""
    if string is None or string == '':
        return None
    try:
        return [int(x.strip()) for x in string.split(',')]
    except:
        return None


def parse_float_list(string):
    """Parse comma-separated float list"""
    if string is None or string == '':  # Check None or empty string
        return None
    try:
        return [float(x.strip()) for x in string.split(',')]
    except:
        return None


def get_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation CAM Analysis')
    # Device parameter
    parser.add_argument('--device', type=str, default='cuda',
                        help='Torch device to use')
    # Input data root directory
    parser.add_argument('--data-root', type=str, default='./data/Duke',
                        help='Data root directory containing images, label, and edge_label folders')
    # Model weights path
    parser.add_argument('--model-path', type=str, default='./model/Duke.pth',
                        help='Path to model weights')
    # Analysis mode selection
    parser.add_argument('--analysis-mode', type=str, default='prediction',
                        choices=['misclassification', 'prediction'],
                        help='Analysis mode: misclassification or prediction')
    # Target category (for prediction mode)
    parser.add_argument('--target-category', type=int, default=None,
                        help='Target category for prediction mode (if None, analyze all predicted classes)')
    parser.add_argument('--num-classes', type=int, default=8,
                        help='Total number of classes (default 8)')
    # Batch processing switch
    parser.add_argument('--process-all', action='store_true', default=True,
                        help='Process all images in directory')
    # Size restoration switch
    parser.add_argument('--restore-size', action='store_true', default=True,
                        help='Restore output images to original size')
    # CAM method selection
    parser.add_argument('--method', type=str, default='prototypecam',
                        choices=[
                            'gradcam', 'hirescam', 'gradcam++',
                            'scorecam', 'eigencam', 'layercam',
                            'gradcamelementwise', 'kpcacam',
                            'shapleycam', 'finercam', 'segxrescam', 'prototypecam'
                        ],
                        help='CAM method')
    # Seg-XRes-CAM specific parameters
    parser.add_argument('--xrescam-pool-size', type=int, default=1,
                        help='Seg-XRes-CAM gradient pooling window size '
                             '(1=HiResCAM, >1=XRes-CAM with pooling, default: 1)')
    parser.add_argument('--xrescam-pool-mode', type=str, default='max',
                        choices=['mean', 'max'],
                        help='Seg-XRes-CAM gradient pooling mode (default: max)')
    # FinerCAM specific parameters
    parser.add_argument('--finer-alpha', type=float, default=0.6,
                        help='FinerCAM alpha parameter (default: 0.6, set to 0 for baseline)')
    parser.add_argument('--finer-num-comparisons', type=int, default=3,
                        help='FinerCAM: number of similar classes to compare per pixel (default: 3)')
    # PrototypeCAM parameters
    parser.add_argument('--n-prototypes', type=int, default=6,
                        help='Number of prototypes for PrototypeCAM (default: 1)')
    parser.add_argument('--proto-iterations', type=int, default=20,
                        help='Number of iterations for PrototypeCAM (default: 5)')
    parser.add_argument('--proto-gamma', type=float, default=0.9,
                        help='Momentum coefficient for PrototypeCAM (default: 0.9)')
    parser.add_argument('--proto-temperature', type=float, default=0.1,
                        help='Temperature parameter for PrototypeCAM (default: 0.1)')
    parser.add_argument('--proto-use-sinkhorn', action='store_true', default=True,
                        help='Whether to use Sinkhorn algorithm in PrototypeCAM')
    parser.add_argument('--proto-normalize', action='store_true', default=True,
                        help='Whether to normalize features in PrototypeCAM')
    parser.add_argument('--proto-use-contrastive', action='store_true', default=True,
                        help='Whether to use contrastive learning in PrototypeCAM')
    parser.add_argument('--proto-contrastive-weight', type=float, default=0.5,
                        help='Negative weight alpha for contrastive learning in PrototypeCAM (default: 0.5)')
    parser.add_argument('--proto-eta-attract', type=float, default=0.2,
                        help='Intra-class attraction learning rate for PrototypeCAM (default: 0.05)')
    parser.add_argument('--proto-eta-repel', type=float, default=2.2,
                        help='Inter-class repulsion learning rate for PrototypeCAM (default: 0.01)')
    parser.add_argument('--proto-reg-lambda', type=float, default=0.01,
                        help='Covariance regularization for PrototypeCAM (default: 0.01)')
    # Target layer selection
    parser.add_argument('--target-layer', type=str, default='Up_conv2',
                        choices=['Conv5', 'Up_conv5', 'Up_conv4', 'Up_conv3', 'Up_conv2'],
                        help='Target layer for visualization')
    # Output root directory
    parser.add_argument('--output-dir', type=str, default='cam_mis',
                        help='Output root directory')
    # Model name
    parser.add_argument('--model-name', type=str, default='UNet',
                        help='Model name for output directory structure')
    # Edge label color
    parser.add_argument('--edge-color', type=str, default='white',
                        choices=['white', 'black', 'red', 'green', 'blue'],
                        help='Color for edge overlay')
    # Save image type
    parser.add_argument('--save-type', type=int, default=2,
                        choices=[1, 2, 3, 4, 5, 6],
                        help='Save image type: 1=combined (4 images tiled), 2=cam_edge (CAM+edge), 3=cam (CAM overlay on image), 4=pure_cam (pure CAM heatmap), 5=pure_cam_npy (raw CAM values as npy), 6=pure_cam_edge (pure CAM heatmap + edge)')

    # Save Excel
    parser.add_argument('--save-excel', action='store_true', default=False,
                        help='Save weights to excel files')
    # Save segmentation results
    parser.add_argument('--save-seg', action='store_true', default=True,
                        help='Save segmentation result images (mask with red edge) to seg folder')
    args = parser.parse_args()

    # Determine folder name based on save_type
    save_type_names = {1: 'combined', 2: 'cam_edge', 3: 'cam', 4: 'pure_cam', 5: 'pure_cam_npy', 6: 'pure_cam_edge'}
    args.save_folder_name = save_type_names[args.save_type]

    return args


def load_image(image_path, data_root):
    """Load and preprocess grayscale image"""
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
    """Load corresponding label image"""
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
    """Load corresponding edge_label image (cache original only, no resize)"""
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
    """Preprocess grayscale image to model input format"""
    input_tensor = torch.from_numpy(image).float()
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    return input_tensor


def resize_to_original(image, original_size):
    """Resize image back to original size"""
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
    """Convert single-channel label image to color image"""
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
    Overlay red edge on black-and-white mask image; white mask covers edges
    """
    h, w = mask.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)

    edge_mask = edge_label > 0
    result[edge_mask] = [0, 0, 255]  # Red edge

    target_mask = mask > 127
    result[target_mask] = [255, 255, 255]  # White target area covers everything

    return result


def overlay_edge_on_image(image_bgr, edge_label, edge_color='white'):
    """
    Overlay edge label on image (normal mode: edge covers image)
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
    FinerCAM dedicated segmentation Target - per-pixel dynamic sorting version
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
    """UNet model output wrapper"""

    def __init__(self, model):
        super(UNetOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class SemanticSegmentationTarget:
    """Target class for semantic segmentation task"""

    def __init__(self, category, mask, device='cpu'):
        self.category = category
        self.mask = torch.from_numpy(mask).to(device)

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


def get_target_layer(model, layer_name):
    """Get the network layer object corresponding to the layer name"""
    layer_map = {
        'Conv5': model.model.Conv5,
        'Up_conv5': model.model.Up_conv5,
        'Up_conv4': model.model.Up_conv4,
        'Up_conv3': model.model.Up_conv3,
        'Up_conv2': model.model.Up_conv2
    }
    return [layer_map[layer_name]]


def create_output_dir(base_dir, method, model_name, layer_name, mode, save_folder_name, *args):
    """Create output directory"""
    if mode == 'misclassification':
        true_class, pred_class = args
        output_path = os.path.join(
            base_dir, method, model_name, layer_name,
            f'class_{true_class}', f'misclassified_as_{pred_class}', save_folder_name
        )
    else:  # prediction
        target_class = args[0]
        output_path = os.path.join(
            base_dir, method, model_name, layer_name,
            f'class_{target_class}', save_folder_name
        )

    os.makedirs(output_path, exist_ok=True)
    return output_path


def create_seg_output_dir(base_dir, model_name, layer_name, mode, *args):
    """Create segmentation result output directory"""
    if mode == 'misclassification':
        true_class, pred_class = args
        output_path = os.path.join(
            base_dir, 'seg', model_name, layer_name,
            f'class_{true_class}', f'misclassified_as_{pred_class}'
        )
    else:  # prediction
        target_class = args[0]
        output_path = os.path.join(
            base_dir, 'seg', model_name, layer_name,
            f'class_{target_class}'
        )

    os.makedirs(output_path, exist_ok=True)
    return output_path


def get_misclassification_mask(label, prediction, true_class, pred_class):
    """Get mask for specific misclassification"""
    mask = ((label == true_class) & (prediction == pred_class)).astype(np.float32)
    pixel_count = int(mask.sum())
    return mask, pixel_count


def get_prediction_mask(prediction, target_class):
    """Get mask for predicted class"""
    mask = (prediction == target_class).astype(np.float32)
    pixel_count = int(mask.sum())
    return mask, pixel_count


class WeightsAggregator:
    """Weights aggregator - save weights by spatial position (224x224)"""

    def __init__(self):
        self.weights_dict = defaultdict(dict)
        self.pos_weights_dict = defaultdict(dict)
        self.neg_weights_dict = defaultdict(dict)
        self.combined_weights_dict = defaultdict(dict)

    def add_weights(self, key, image_name, weights):
        """Add weights - save as spatial matrix"""
        if isinstance(weights, np.ndarray):
            self.weights_dict[key][image_name] = weights.copy()
        elif isinstance(weights, torch.Tensor):
            self.weights_dict[key][image_name] = weights.cpu().detach().numpy().copy()
        else:
            self.weights_dict[key][image_name] = np.array(weights).copy()

    def add_prototype_weights(self, key, image_name, pos_weights, neg_weights, combined_weights):
        """Add three types of weights for prototypecam (224x224 spatial matrices)"""
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
        """Save spatial weights (224x224) to Excel"""
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
        """Save all weights to Excel files (by 224x224 spatial position)"""
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
            else:  # prediction
                target_class = key[0] if isinstance(key, tuple) else key
                excel_dir = os.path.join(
                    base_dir, method, model_name, layer_name, f'class_{target_class}'
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
            else:  # prediction
                target_class = key[0] if isinstance(key, tuple) else key
                excel_dir = os.path.join(
                    base_dir, method, model_name, layer_name, f'class_{target_class}'
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
                              label_resized=None, pred_mask=None, true_class=None):
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

            elif args.analysis_mode == 'prediction':
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
            negative_class=None
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
    """Process a single image"""

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
                    tasks.append(('mis', true_class, pred_class, mask, pixel_count))

    else:  # prediction
        if args.target_category is not None:
            target_classes = [args.target_category]
        else:
            target_classes = np.unique(pred_mask).tolist()

        tasks = []
        for target_class in target_classes:
            mask, pixel_count = get_prediction_mask(pred_mask, target_class)
            if pixel_count > 0:
                tasks.append(('pred', target_class, None, mask, pixel_count))

    for task in tasks:
        pattern_count += 1

        if task[0] == 'mis':  # misclassification
            _, true_class, pred_class, mask, pixel_count = task
            target_class = pred_class

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

        else:  # prediction
            _, target_class, _, mask, pixel_count = task

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

        cam_weights, cam_image, pos_weights, neg_weights, combined_weights, grayscale_cam = compute_cam_and_visualize(
            image_resized, input_tensor, model, target_layers,
            target_class, mask, device, args, cam_algorithm,
            label_resized=label_resized,
            pred_mask=pred_mask,
            true_class=true_class if task[0] == 'mis' else None
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
            # Type 1: four images tiled
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
            # Type 2: CAM image + edge label
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
            # Type 3: pure CAM image without edge label
            if args.restore_size:
                cam_image_bgr = resize_to_original(cam_image_bgr, original_size).astype(np.uint8)

            output_image = cam_image_bgr

        elif args.save_type == 4:
            # Type 4: pure CAM heatmap, no original image overlay
            import matplotlib.pyplot as plt

            colored_cam = plt.cm.jet(grayscale_cam)[:, :, :3]
            colored_cam = (colored_cam * 255).astype(np.uint8)
            colored_cam_bgr = cv2.cvtColor(colored_cam, cv2.COLOR_RGB2BGR)

            if args.restore_size:
                output_image = resize_to_original(colored_cam_bgr, original_size).astype(np.uint8)
            else:
                output_image = colored_cam_bgr

        elif args.save_type == 5:
            # Type 5: pure CAM raw values, saved as npy format
            if args.restore_size:
                cam_data = resize_to_original(grayscale_cam, original_size)
            else:
                cam_data = grayscale_cam

            output_filename = f'{filename_without_ext}.npy'
            output_path = os.path.join(output_dir, output_filename)
            np.save(output_path, cam_data)
            continue  # Skip the image saving below

        elif args.save_type == 6:
            # Type 6: pure CAM heatmap (no original image overlay) + edge label
            import matplotlib.pyplot as plt

            # Convert grayscale_cam (0-1) to jet color heatmap
            colored_cam = plt.cm.jet(grayscale_cam)[:, :, :3]  # Take only RGB, drop alpha channel
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

            # Overlay edge on pure CAM heatmap
            output_image = overlay_edge_on_image(colored_cam_bgr, edge_label_to_use, args.edge_color)

        # Save image
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
        "eigencam": EigenCAM,
        "layercam": LayerCAM,
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

    if args.save_excel:
        weights_aggregator.save_all_to_excel(
            args.output_dir, args.method, args.model_name, args.target_layer,
            args.analysis_mode
        )
