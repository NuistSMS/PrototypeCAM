import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom


def kmeans_plus_plus_init(features, n_prototypes, device='cpu'):
    N, C = features.shape

    if not isinstance(features, torch.Tensor):
        features = torch.from_numpy(features).float().to(device)

    centers = [features[torch.randint(N, (1,), device=device).item()]]

    for _ in range(1, n_prototypes):
        centers_tensor = torch.stack(centers)  # [k, C]

        distances = torch.cdist(features.unsqueeze(0), centers_tensor.unsqueeze(0)).squeeze(0)
        min_distances = distances.min(dim=1).values  # [N]

        if min_distances.sum() < 1e-10:
            next_center_idx = torch.randint(N, (1,), device=device).item()
        else:
            probabilities = min_distances / min_distances.sum()
            probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)

            if probabilities.sum() > 0:
                probabilities = probabilities / probabilities.sum()
            else:
                probabilities = torch.ones(N, device=device) / N

            next_center_idx = torch.multinomial(probabilities, 1).item()

        centers.append(features[next_center_idx])

    prototypes = torch.stack(centers)

    return prototypes  # [N.D]


def sinkhorn_knopp(out, n_iterations=3, epsilon=0.05):
    """Sinkhorn-Knopp for soft assignment"""
    L = torch.exp(out / epsilon).t()
    K, N = L.shape
    sum_L = torch.sum(L)
    L /= sum_L

    for _ in range(n_iterations):
        L /= torch.sum(L, dim=1, keepdim=True)
        L /= K
        L /= torch.sum(L, dim=0, keepdim=True)
        L /= N

    L *= N
    L = L.t()
    indices = torch.argmax(L, dim=1)

    return L, indices


def estimate_covariance(features, reg_lambda=0.01):
    N, D = features.shape

    mu = features.mean(dim=0)  # [D]

    centered = features - mu.unsqueeze(0)  # [N, D]

    cov = (centered.t() @ centered) / N + reg_lambda * torch.eye(D, device=features.device)
    return cov, mu


def compute_mahalanobis_force_vectorized(prototypes, targets, cov_inv, epsilon=0.1, max_force=5.0):
    K, D = prototypes.shape
    M = targets.shape[0]

    diff = prototypes.unsqueeze(1) - targets.unsqueeze(0)  # [K, M, D]

    diff_transformed = diff @ cov_inv  # [K, M, D]
    mahal_dist_sq = torch.sum(diff_transformed * diff, dim=2)  # [K, M]

    force_magnitudes = torch.clamp(
        1.0 / (mahal_dist_sq + epsilon),
        max=max_force
    )  # [K, M]

    forces = (diff_transformed * force_magnitudes.unsqueeze(2)).mean(dim=1)  # [K, D]

    return forces


# =====================================================================
# NEW: Bayesian prior helper functions
# =====================================================================

def compute_gaussian_log_likelihood(features, prototypes, cov, reg_lambda=0.01):
    """
    Compute log p(x | c) using max-component approximation.

    For each class c with K prototypes, the class-conditional likelihood is
    approximated as:
        p(x | c) ≈ max_k  p(x | p_k^c, Sigma_k^c)

    where p(x | p_k, Sigma) is a multivariate Gaussian density.

    This corresponds to Eq. (class_likelihood_approx) in the paper.

    Args:
        features:   [N, D] spatial feature tokens
        prototypes: [K, D] class prototypes
        cov:        [D, D] class covariance matrix (shared across prototypes)

    Returns:
        max_log_lik: [N] max log-likelihood across prototypes for each token
    """
    D = features.shape[1]
    cov_inv = torch.linalg.inv(cov)
    log_det = torch.logdet(cov)

    # Constant term: -D/2 * log(2*pi)
    const = -0.5 * D * np.log(2 * np.pi)

    log_likelihoods = []
    for k in range(prototypes.shape[0]):
        diff = features - prototypes[k].unsqueeze(0)  # [N, D]
        mahal = (diff @ cov_inv * diff).sum(dim=1)  # [N]
        log_lik = const - 0.5 * log_det - 0.5 * mahal  # [N]
        log_likelihoods.append(log_lik)

    log_likelihoods = torch.stack(log_likelihoods, dim=1)  # [N, K]
    max_log_lik = log_likelihoods.max(dim=1).values  # [N]

    return max_log_lik


def compute_class_prior(pred_mask_flat, classes):
    """
    Estimate class prior pi_c from pixel frequencies in the prediction mask.

    This corresponds to Eq. (class_prior) in the paper:
        pi_c = N_c / sum_{c'} N_{c'}

    Args:
        pred_mask_flat: [HW] flattened prediction mask with class labels
        classes:        list of class indices to consider

    Returns:
        priors: dict {class_idx: pi_c}
    """
    counts = {}
    total = 0
    for cls in classes:
        n_cls = (pred_mask_flat == cls).sum().item()
        counts[cls] = n_cls
        total += n_cls

    priors = {}
    for cls in classes:
        priors[cls] = counts[cls] / (total + 1e-8)

    return priors


def compute_multiclass_posterior(features, class_prototypes, class_covs,
                                 class_priors, target_class, reg_lambda=0.01):
    """
    Compute P(target_class | x) via Bayes' theorem over all classes.

    This corresponds to Eq. (posterior) in the paper:
        P(c | x) = pi_c * p(x|c) / sum_{c'} pi_{c'} * p(x|c')

    Implemented in log-space with softmax for numerical stability.

    Args:
        features:         [N, D] spatial feature tokens
        class_prototypes: dict {cls: [K, D] prototypes}
        class_covs:       dict {cls: [D, D] covariance matrices}
        class_priors:     dict {cls: float prior probability}
        target_class:     int, the class P for which to compute posterior

    Returns:
        posterior: [N] posterior probability P(target_class | x_i) for each token
    """
    classes = list(class_prototypes.keys())
    log_joints = []

    for cls in classes:
        log_lik = compute_gaussian_log_likelihood(
            features, class_prototypes[cls], class_covs[cls], reg_lambda
        )  # [N]
        log_prior = np.log(class_priors[cls] + 1e-8)
        log_joints.append(log_lik + log_prior)  # [N]

    log_joint_stack = torch.stack(log_joints, dim=1)  # [N, C]
    posteriors = F.softmax(log_joint_stack, dim=1)  # [N, C]

    target_idx = classes.index(target_class)
    target_posterior = posteriors[:, target_idx]  # [N]

    return target_posterior


def compute_pairwise_posterior(features, proto_P, cov_P, prior_P,
                               proto_Q, cov_Q, prior_Q, reg_lambda=0.01):
    """
    Compute pairwise posterior P(P | x; P, Q) using sigmoid form.

    This corresponds to Eq. (pairwise_sigmoid) in the paper:
        P(P|x; P,Q) = sigma( log(pi_P/pi_Q)
                             + 0.5*(M^2(x;q*,Sigma_Q) - M^2(x;p*,Sigma_P))
                             + 0.5*log(|Sigma_Q|/|Sigma_P|) )

    Args:
        features: [N, D]
        proto_P:  [K, D] prototypes of class P
        cov_P:    [D, D] covariance of class P
        prior_P:  float, pi_P
        proto_Q:  [K, D] prototypes of class Q
        cov_Q:    [D, D] covariance of class Q
        prior_Q:  float, pi_Q

    Returns:
        posterior_P: [N] pairwise posterior for class P
    """
    cov_P_inv = torch.linalg.inv(cov_P)
    cov_Q_inv = torch.linalg.inv(cov_Q)
    log_det_P = torch.logdet(cov_P)
    log_det_Q = torch.logdet(cov_Q)

    # Mahalanobis distance to best prototype of P: min_k M^2(x; p_k^P, Sigma_P)
    mahal_P_list = []
    for k in range(proto_P.shape[0]):
        diff = features - proto_P[k].unsqueeze(0)
        m2 = (diff @ cov_P_inv * diff).sum(dim=1)
        mahal_P_list.append(m2)
    mahal_P = torch.stack(mahal_P_list, dim=1).min(dim=1).values  # [N]

    # Mahalanobis distance to best prototype of Q: min_k M^2(x; q_k^Q, Sigma_Q)
    mahal_Q_list = []
    for k in range(proto_Q.shape[0]):
        diff = features - proto_Q[k].unsqueeze(0)
        m2 = (diff @ cov_Q_inv * diff).sum(dim=1)
        mahal_Q_list.append(m2)
    mahal_Q = torch.stack(mahal_Q_list, dim=1).min(dim=1).values  # [N]

    # Sigmoid argument: log(pi_P/pi_Q) + 0.5*(M^2_Q - M^2_P) + 0.5*log(|Sigma_Q|/|Sigma_P|)
    log_prior_ratio = np.log((prior_P + 1e-8) / (prior_Q + 1e-8))
    mahal_diff = 0.5 * (mahal_Q - mahal_P)
    log_det_ratio = 0.5 * (log_det_Q - log_det_P)

    logit = log_prior_ratio + mahal_diff + log_det_ratio  # [N]
    posterior_P = torch.sigmoid(logit)  # [N]

    return posterior_P


# =====================================================================


class PrototypeCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 n_prototypes=1,
                 n_iterations=5,
                 gamma=0.9,
                 temperature=0.1,
                 use_sinkhorn=True,
                 normalize=True,
                 use_contrastive=False,
                 contrastive_weight=0.5,
                 num_classes=8,
                 eta_attract=0.05,
                 eta_repel=0.01,
                 reg_lambda=0.01,
                 use_prior=False):  # NEW: flag to enable Bayesian prior
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.device = next(self.model.parameters()).device

        self.n_prototypes = n_prototypes
        self.n_iterations = n_iterations
        self.gamma = gamma
        self.temperature = temperature
        self.use_sinkhorn = use_sinkhorn
        self.normalize = normalize

        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.num_classes = num_classes

        self.eta_attract = eta_attract
        self.eta_repel = eta_repel
        self.reg_lambda = reg_lambda

        self.use_prior = use_prior  # NEW

        self.last_weights = None
        self.last_prototypes = None
        self.last_negative_prototypes = None
        self.last_all_class_prototypes = None

        self.last_pos_weights = None
        self.last_neg_weights = None
        self.last_combined_weights = None

        self.activations = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations.append(output.detach())

        for target_layer in self.target_layers:
            self.hooks.append(
                target_layer.register_forward_hook(forward_hook)
            )

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_similarity(self, features, prototypes, features_prenorm=False):
        if self.normalize:
            if not features_prenorm:
                features = F.normalize(features, p=2, dim=-1)
            prototypes = F.normalize(prototypes, p=2, dim=-1)

        if len(features.shape) == 2:
            similarity = torch.mm(features, prototypes.t())
        else:
            similarity = torch.einsum('bnc,kc->bnk', features, prototypes)

        return similarity

    def initialize_prototypes(self, target_features, device):
        if self.n_prototypes == 1:
            prototypes = target_features.mean(dim=0, keepdim=True)
        else:
            prototypes = kmeans_plus_plus_init(
                target_features,
                self.n_prototypes,
                device
            )
        return prototypes

    def update_prototypes(self, target_features, prototypes,
                          negative_features=None, negative_prototypes=None):

        target_features_orig = target_features.clone()
        prototypes_orig = prototypes.clone()

        if negative_features is not None:
            negative_features_orig = negative_features.clone()
        if negative_prototypes is not None:
            negative_prototypes_orig = negative_prototypes.clone()

        cov_pos, mu_pos = estimate_covariance(target_features_orig, self.reg_lambda)
        cov_pos_inv = torch.linalg.inv(cov_pos)

        cov_between_inv = None
        if negative_features is not None and negative_prototypes is not None:
            cov_neg, mu_neg = estimate_covariance(negative_features_orig, self.reg_lambda)

            n_pos = target_features_orig.shape[0]
            n_neg = negative_features_orig.shape[0]

            cov_pooled = (n_pos * cov_pos + n_neg * cov_neg) / (n_pos + n_neg)
            cov_between_inv = torch.linalg.inv(cov_pooled)

        if self.normalize:
            target_features_norm = F.normalize(target_features, p=2, dim=-1)
        else:
            target_features_norm = target_features

        for iteration in range(self.n_iterations):
            similarities = self.compute_similarity(
                target_features_norm,
                prototypes,
                features_prenorm=self.normalize
            )

            if self.use_sinkhorn and self.n_prototypes > 1:
                assignments, _ = sinkhorn_knopp(
                    similarities,
                    n_iterations=3,
                    epsilon=0.05
                )
            else:
                assignments = F.softmax(similarities / self.temperature, dim=-1)

            new_prototypes = []
            new_prototypes_orig = []

            for k in range(self.n_prototypes):
                p_k_orig = prototypes_orig[k]
                weights = assignments[:, k:k + 1]

                mean_k_orig = (target_features_orig * weights).sum(0) / (weights.sum() + 1e-8)
                diff_attract = mean_k_orig - p_k_orig
                force_attract = cov_pos_inv @ diff_attract

                force_repel = torch.zeros_like(p_k_orig)
                if cov_between_inv is not None and negative_prototypes_orig is not None:
                    force_repel = compute_mahalanobis_force_vectorized(
                        p_k_orig.unsqueeze(0),
                        negative_prototypes_orig,
                        cov_between_inv,
                        epsilon=0.1,
                        max_force=5.0
                    ).squeeze(0)

                p_k_new_orig = p_k_orig + self.eta_attract * force_attract + self.eta_repel * force_repel
                p_k_updated_orig = self.gamma * p_k_orig + (1 - self.gamma) * p_k_new_orig

                if self.normalize:
                    p_k_updated = F.normalize(p_k_updated_orig.unsqueeze(0), p=2, dim=-1).squeeze(0)
                else:
                    p_k_updated = p_k_updated_orig

                new_prototypes.append(p_k_updated)
                new_prototypes_orig.append(p_k_updated_orig)

            prototypes = torch.stack(new_prototypes)
            prototypes_orig = torch.stack(new_prototypes_orig)

        return prototypes

    def update_prototypes_multiclass(self, all_class_features, all_class_prototypes):
        all_class_features_orig = {cls: feat.clone() for cls, feat in all_class_features.items()}
        all_class_prototypes_orig = {cls: proto.clone() for cls, proto in all_class_prototypes.items()}

        class_covs = {}
        class_covs_inv = {}
        class_means = {}

        for cls, features in all_class_features_orig.items():
            cov, mu = estimate_covariance(features, self.reg_lambda)
            class_covs[cls] = cov
            class_covs_inv[cls] = torch.linalg.inv(cov)
            class_means[cls] = mu

        if self.normalize:
            all_class_features_norm = {
                cls: F.normalize(feat, p=2, dim=-1)
                for cls, feat in all_class_features.items()
            }
        else:
            all_class_features_norm = all_class_features

        updated_prototypes = {}
        updated_prototypes_orig = {}

        for cls, prototypes in all_class_prototypes.items():
            features_norm = all_class_features_norm[cls]
            features_orig = all_class_features_orig[cls]
            prototypes_orig = all_class_prototypes_orig[cls]
            cov_inv = class_covs_inv[cls]

            similarities = self.compute_similarity(
                features_norm,
                prototypes,
                features_prenorm=self.normalize
            )

            if self.use_sinkhorn and self.n_prototypes > 1:
                assignments, _ = sinkhorn_knopp(similarities, n_iterations=3, epsilon=0.05)
            else:
                assignments = F.softmax(similarities / self.temperature, dim=-1)

            new_prototypes = []
            new_prototypes_orig = []

            for k in range(self.n_prototypes):
                p_k_orig = prototypes_orig[k]
                weights = assignments[:, k:k + 1]

                mean_k_orig = (features_orig * weights).sum(0) / (weights.sum() + 1e-8)
                force_attract = cov_inv @ (mean_k_orig - p_k_orig)

                force_repel = torch.zeros_like(p_k_orig)

                for other_cls, other_prototypes_orig in all_class_prototypes_orig.items():
                    if other_cls == cls:
                        continue

                    n_cls = all_class_features_orig[cls].shape[0]
                    n_other = all_class_features_orig[other_cls].shape[0]

                    cov_pooled = (n_cls * class_covs[cls] + n_other * class_covs[other_cls]) / (n_cls + n_other)
                    cov_between_inv = torch.linalg.inv(cov_pooled)

                    force_repel += compute_mahalanobis_force_vectorized(
                        p_k_orig.unsqueeze(0),
                        other_prototypes_orig,
                        cov_between_inv,
                        epsilon=0.1,
                        max_force=5.0
                    ).squeeze(0)

                num_other_classes = len(all_class_prototypes) - 1
                if num_other_classes > 0:
                    force_repel = force_repel / num_other_classes

                p_k_new_orig = p_k_orig + self.eta_attract * force_attract + self.eta_repel * force_repel
                p_k_updated_orig = self.gamma * p_k_orig + (1 - self.gamma) * p_k_new_orig

                if self.normalize:
                    p_k_updated = F.normalize(p_k_updated_orig.unsqueeze(0), p=2, dim=-1).squeeze(0)
                else:
                    p_k_updated = p_k_updated_orig

                new_prototypes.append(p_k_updated)
                new_prototypes_orig.append(p_k_updated_orig)

            updated_prototypes[cls] = torch.stack(new_prototypes)
            updated_prototypes_orig[cls] = torch.stack(new_prototypes_orig)

        return updated_prototypes

    def forward(self, input_tensor, targets, negative_mask=None, pred_mask_np=None,
                target_class=None, analysis_mode='misclassification', negative_class=None):
        input_tensor = input_tensor.to(self.device)
        self.activations = []

        with torch.no_grad():
            _ = self.model(input_tensor)

        activations = self.activations[0]

        if len(targets) > 0:
            target = targets[0]
        else:
            B, C, H, W = activations.shape
            mask = torch.ones(H, W).to(self.device)

            class DummyTarget:
                def __init__(self, mask):
                    self.mask = mask

            target = DummyTarget(mask)

        device = activations.device

        if hasattr(target, 'mask'):
            mask = target.mask
        else:
            if len(activations.shape) == 4:
                B, C, H, W = activations.shape
                mask = torch.ones(H, W).to(device)
            else:
                B, C, D, H, W = activations.shape
                mask = torch.ones(D, H, W).to(device)

        is_3d = len(activations.shape) == 5

        if is_3d:
            B, C, D, H, W = activations.shape
            spatial_dims = (D, H, W)
            all_features = activations.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
        else:
            B, C, H, W = activations.shape
            spatial_dims = (H, W)
            all_features = activations.permute(0, 2, 3, 1).reshape(B, -1, C)

        if is_3d:
            mask_resized = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=spatial_dims,
                mode='nearest'
            ).squeeze()
            mask_flat = mask_resized.reshape(-1)
        else:
            mask_resized = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=spatial_dims,
                mode='nearest'
            ).squeeze()
            mask_flat = mask_resized.reshape(-1)

        if negative_mask is not None and self.use_contrastive and analysis_mode == 'misclassification':
            negative_mask_tensor = torch.from_numpy(negative_mask).float().to(device)
            if is_3d:
                neg_mask_resized = F.interpolate(
                    negative_mask_tensor.unsqueeze(0).unsqueeze(0),
                    size=spatial_dims,
                    mode='nearest'
                ).squeeze()
                neg_mask_flat = neg_mask_resized.reshape(-1)
            else:
                neg_mask_resized = F.interpolate(
                    negative_mask_tensor.unsqueeze(0).unsqueeze(0),
                    size=spatial_dims,
                    mode='nearest'
                ).squeeze()
                neg_mask_flat = neg_mask_resized.reshape(-1)
        else:
            neg_mask_flat = None

        if pred_mask_np is not None and self.use_contrastive and (
                analysis_mode == 'prediction' or analysis_mode == 'whynot'):
            pred_mask_tensor = torch.from_numpy(pred_mask_np).float().to(device)
            if is_3d:
                pred_mask_resized = F.interpolate(
                    pred_mask_tensor.unsqueeze(0).unsqueeze(0),
                    size=spatial_dims,
                    mode='nearest'
                ).squeeze()
                pred_mask_flat = pred_mask_resized.reshape(-1)
            else:
                pred_mask_resized = F.interpolate(
                    pred_mask_tensor.unsqueeze(0).unsqueeze(0),
                    size=spatial_dims,
                    mode='nearest'
                ).squeeze()
                pred_mask_flat = pred_mask_resized.reshape(-1)
        else:
            pred_mask_flat = None

        all_cams = []
        all_prototypes = []
        all_negative_prototypes = []
        all_class_prototypes_list = []

        all_pos_weights_spatial = []
        all_neg_weights_spatial = []
        all_combined_weights_spatial = []

        for b in range(B):
            features = all_features[b]

            target_mask = mask_flat > 0
            target_features = features[target_mask]

            if target_features.shape[0] == 0:
                if is_3d:
                    cam = torch.zeros(D, H, W).to(device)
                    pos_weights_spatial = torch.zeros(D, H, W).to(device)
                    neg_weights_spatial = torch.zeros(D, H, W).to(device)
                else:
                    cam = torch.zeros(H, W).to(device)
                    pos_weights_spatial = torch.zeros(H, W).to(device)
                    neg_weights_spatial = torch.zeros(H, W).to(device)
            else:
                pos_weights = None
                neg_weights = None

                # ========== misclassification mode (mean) ==========
                if analysis_mode == 'misclassification':
                    prototypes = self.initialize_prototypes(target_features, device)

                    neg_prototypes = None
                    negative_features = None
                    if self.use_contrastive and neg_mask_flat is not None:
                        neg_target_mask = neg_mask_flat > 0
                        negative_features = features[neg_target_mask]

                        if negative_features.shape[0] > 0:
                            neg_prototypes = self.initialize_prototypes(negative_features, device)

                    prototypes = self.update_prototypes(
                        target_features,
                        prototypes,
                        negative_features=negative_features,
                        negative_prototypes=neg_prototypes
                    )

                    similarities_pos = self.compute_similarity(features, prototypes)

                    if self.use_contrastive and negative_features is not None and negative_features.shape[0] > 0:
                        neg_prototypes = self.update_prototypes(
                            negative_features,
                            neg_prototypes,
                            negative_features=target_features,
                            negative_prototypes=prototypes
                        )

                        similarities_neg = self.compute_similarity(features, neg_prototypes)

                        if self.n_prototypes > 1:
                            neg_weights = similarities_neg.mean(dim=-1)
                        else:
                            neg_weights = similarities_neg.squeeze(-1)

                        all_negative_prototypes.append(neg_prototypes.cpu().detach().numpy())
                    else:
                        neg_weights = torch.zeros_like(similarities_pos[:, 0])

                    if self.n_prototypes > 1:
                        pos_weights = similarities_pos.mean(dim=-1)
                    else:
                        pos_weights = similarities_pos.squeeze(-1)

                    weights = pos_weights * (1 - self.contrastive_weight * neg_weights)
                    all_prototypes.append(prototypes.cpu().detach().numpy())

                # ========== prediction mode (max) ==========
                elif analysis_mode == 'prediction' and self.use_contrastive:
                    if pred_mask_flat is not None and target_class is not None:
                        unique_classes = torch.unique(pred_mask_flat).cpu().numpy()
                        unique_classes = [int(c) for c in unique_classes if c >= 0]

                        all_class_features_dict = {}
                        all_class_prototypes_dict = {}

                        for cls in unique_classes:
                            cls_mask = pred_mask_flat == cls
                            cls_features = features[cls_mask]

                            if cls_features.shape[0] > 0:
                                cls_prototypes = self.initialize_prototypes(cls_features, device)
                                all_class_features_dict[cls] = cls_features
                                all_class_prototypes_dict[cls] = cls_prototypes

                        if len(all_class_prototypes_dict) > 1:
                            all_class_prototypes_dict = self.update_prototypes_multiclass(
                                all_class_features_dict,
                                all_class_prototypes_dict
                            )

                        all_class_prototypes = {}
                        sim_all_classes = []

                        for cls in unique_classes:
                            if cls in all_class_prototypes_dict:
                                cls_prototypes = all_class_prototypes_dict[cls]
                                all_class_prototypes[cls] = cls_prototypes

                                sim_cls = self.compute_similarity(features, cls_prototypes)

                                if self.n_prototypes > 1:
                                    sim_cls_max = sim_cls.max(dim=-1).values
                                else:
                                    sim_cls_max = sim_cls.squeeze(-1)

                                sim_all_classes.append((cls, sim_cls_max))

                        if len(sim_all_classes) > 0:
                            class_indices = [item[0] for item in sim_all_classes]
                            sim_matrix = torch.stack([item[1] for item in sim_all_classes], dim=1)

                            if target_class in class_indices:
                                target_idx = class_indices.index(target_class)
                                sim_target = sim_matrix[:, target_idx]
                                pos_weights = sim_target

                                # =============================================
                                # NEW: Bayesian prior modulation (prediction)
                                # =============================================
                                if self.use_prior and len(all_class_features_dict) > 1:
                                    # Step 1: Estimate class priors pi_c
                                    class_priors = compute_class_prior(
                                        pred_mask_flat, list(all_class_features_dict.keys())
                                    )

                                    # Step 2: Estimate covariance for each class
                                    class_covs_for_posterior = {}
                                    for cls in all_class_features_dict:
                                        cov_cls, _ = estimate_covariance(
                                            all_class_features_dict[cls], self.reg_lambda
                                        )
                                        class_covs_for_posterior[cls] = cov_cls

                                    # Step 3: Compute posterior P(P | x_i)
                                    posterior_P = compute_multiclass_posterior(
                                        features,
                                        all_class_prototypes_dict,
                                        class_covs_for_posterior,
                                        class_priors,
                                        target_class,
                                        self.reg_lambda
                                    )

                                    # Step 4: Modulate pos_weights
                                    # w_tilde_pos = w_pos * P(P | x_i)
                                    pos_weights = pos_weights * posterior_P
                                # =============================================

                                other_indices = [
                                    i for i in range(len(class_indices))
                                    if i != target_idx and class_indices[i] != 0
                                ]
                                if len(other_indices) > 0:
                                    sim_competitors = sim_matrix[:, other_indices]
                                    sim_max_competitor = sim_competitors.max(dim=1).values
                                    neg_weights = sim_max_competitor

                                    weights = pos_weights * (1 - self.contrastive_weight * neg_weights)
                                else:
                                    neg_weights = torch.zeros_like(sim_target)
                                    weights = pos_weights
                            else:
                                prototypes = self.initialize_prototypes(target_features, device)
                                prototypes = self.update_prototypes(target_features, prototypes)
                                similarities_pos = self.compute_similarity(features, prototypes)
                                if self.n_prototypes > 1:
                                    pos_weights = similarities_pos.max(dim=-1).values
                                else:
                                    pos_weights = similarities_pos.squeeze(-1)
                                neg_weights = torch.zeros_like(pos_weights)
                                weights = pos_weights

                            all_class_prototypes_list.append(all_class_prototypes)
                        else:
                            prototypes = self.initialize_prototypes(target_features, device)
                            prototypes = self.update_prototypes(target_features, prototypes)
                            similarities_pos = self.compute_similarity(features, prototypes)
                            if self.n_prototypes > 1:
                                pos_weights = similarities_pos.max(dim=-1).values
                            else:
                                pos_weights = similarities_pos.squeeze(-1)
                            neg_weights = torch.zeros_like(pos_weights)
                            weights = pos_weights
                            all_prototypes.append(prototypes.cpu().detach().numpy())
                    else:
                        prototypes = self.initialize_prototypes(target_features, device)
                        prototypes = self.update_prototypes(target_features, prototypes)
                        similarities_pos = self.compute_similarity(features, prototypes)
                        if self.n_prototypes > 1:
                            pos_weights = similarities_pos.max(dim=-1).values
                        else:
                            pos_weights = similarities_pos.squeeze(-1)
                        neg_weights = torch.zeros_like(pos_weights)
                        weights = pos_weights
                        all_prototypes.append(prototypes.cpu().detach().numpy())

                # ========== whynot mode (mean) ==========
                elif analysis_mode == 'whynot' and self.use_contrastive:
                    if pred_mask_flat is not None and target_class is not None and negative_class is not None:
                        classes_to_process = [target_class, negative_class]

                        all_class_features_dict = {}
                        all_class_prototypes_dict = {}

                        for cls in classes_to_process:
                            cls_mask = pred_mask_flat == cls
                            cls_features = features[cls_mask]

                            if cls_features.shape[0] > 0:
                                cls_prototypes = self.initialize_prototypes(cls_features, device)
                                all_class_features_dict[cls] = cls_features
                                all_class_prototypes_dict[cls] = cls_prototypes

                        if len(all_class_prototypes_dict) == 2:
                            all_class_prototypes_dict = self.update_prototypes_multiclass(
                                all_class_features_dict,
                                all_class_prototypes_dict
                            )

                            if target_class in all_class_prototypes_dict:
                                target_prototypes = all_class_prototypes_dict[target_class]
                                sim_target = self.compute_similarity(features, target_prototypes)

                                if self.n_prototypes > 1:
                                    pos_weights = sim_target.mean(dim=-1)
                                else:
                                    pos_weights = sim_target.squeeze(-1)
                            else:
                                pos_weights = torch.zeros(features.shape[0]).to(device)

                            if negative_class in all_class_prototypes_dict:
                                negative_prototypes = all_class_prototypes_dict[negative_class]
                                sim_negative = self.compute_similarity(features, negative_prototypes)

                                if self.n_prototypes > 1:
                                    neg_weights = sim_negative.mean(dim=-1)
                                else:
                                    neg_weights = sim_negative.squeeze(-1)
                            else:
                                neg_weights = torch.zeros(features.shape[0]).to(device)

                            # =============================================
                            # NEW: Bayesian pairwise posterior (whynot)
                            # =============================================
                            if (self.use_prior
                                    and target_class in all_class_features_dict
                                    and negative_class in all_class_features_dict):
                                # Step 1: Estimate class priors for P and Q
                                class_priors = compute_class_prior(
                                    pred_mask_flat, [target_class, negative_class]
                                )
                                prior_P = class_priors[target_class]
                                prior_Q = class_priors[negative_class]

                                # Step 2: Estimate covariance for P and Q
                                cov_P, _ = estimate_covariance(
                                    all_class_features_dict[target_class], self.reg_lambda
                                )
                                cov_Q, _ = estimate_covariance(
                                    all_class_features_dict[negative_class], self.reg_lambda
                                )

                                # Step 3: Compute pairwise posterior via sigmoid
                                # P(P|x; P,Q) = sigma(log(pi_P/pi_Q)
                                #   + 0.5*(M^2_Q - M^2_P) + 0.5*log(|Sigma_Q|/|Sigma_P|))
                                posterior_P = compute_pairwise_posterior(
                                    features,
                                    all_class_prototypes_dict[target_class], cov_P, prior_P,
                                    all_class_prototypes_dict[negative_class], cov_Q, prior_Q,
                                    self.reg_lambda
                                )

                                # Step 4: Modulate pos_weights
                                # w_tilde_pos = w_pos * P(P | x; P, Q)
                                pos_weights = pos_weights * posterior_P
                            # =============================================

                            weights = pos_weights * (1 - self.contrastive_weight * neg_weights)

                            all_class_prototypes_list.append(all_class_prototypes_dict)
                        else:
                            prototypes = self.initialize_prototypes(target_features, device)
                            prototypes = self.update_prototypes(target_features, prototypes)
                            similarities_pos = self.compute_similarity(features, prototypes)
                            if self.n_prototypes > 1:
                                pos_weights = similarities_pos.mean(dim=-1)
                            else:
                                pos_weights = similarities_pos.squeeze(-1)
                            neg_weights = torch.zeros_like(pos_weights)
                            weights = pos_weights
                            all_prototypes.append(prototypes.cpu().detach().numpy())
                    else:
                        prototypes = self.initialize_prototypes(target_features, device)
                        prototypes = self.update_prototypes(target_features, prototypes)
                        similarities_pos = self.compute_similarity(features, prototypes)
                        if self.n_prototypes > 1:
                            pos_weights = similarities_pos.mean(dim=-1)
                        else:
                            pos_weights = similarities_pos.squeeze(-1)
                        neg_weights = torch.zeros_like(pos_weights)
                        weights = pos_weights
                        all_prototypes.append(prototypes.cpu().detach().numpy())

                # ========== no contrastive (mean) ==========
                else:
                    prototypes = self.initialize_prototypes(target_features, device)
                    prototypes = self.update_prototypes(target_features, prototypes)
                    similarities_pos = self.compute_similarity(features, prototypes)

                    if self.n_prototypes > 1:
                        pos_weights = similarities_pos.mean(dim=-1)
                    else:
                        pos_weights = similarities_pos.squeeze(-1)

                    neg_weights = torch.zeros_like(pos_weights)
                    weights = pos_weights

                    all_prototypes.append(prototypes.cpu().detach().numpy())

                if is_3d:
                    cam = weights.reshape(D, H, W)
                    pos_weights_spatial = pos_weights.reshape(D, H, W)
                    neg_weights_spatial = neg_weights.reshape(D, H, W)
                else:
                    cam = weights.reshape(H, W)
                    pos_weights_spatial = pos_weights.reshape(H, W)
                    neg_weights_spatial = neg_weights.reshape(H, W)

            all_cams.append(cam)
            all_pos_weights_spatial.append(pos_weights_spatial)
            all_neg_weights_spatial.append(neg_weights_spatial)
            all_combined_weights_spatial.append(cam.clone())

        all_cams = torch.stack(all_cams)
        all_pos_weights_spatial = torch.stack(all_pos_weights_spatial)
        all_neg_weights_spatial = torch.stack(all_neg_weights_spatial)
        all_combined_weights_spatial = torch.stack(all_combined_weights_spatial)

        input_size = (input_tensor.shape[-2], input_tensor.shape[-1])

        if B > 0:
            self.last_weights = all_cams[0].cpu().detach().numpy()
            if len(all_prototypes) > 0:
                self.last_prototypes = all_prototypes[0]
            if len(all_negative_prototypes) > 0:
                self.last_negative_prototypes = all_negative_prototypes[0]
            if len(all_class_prototypes_list) > 0:
                self.last_all_class_prototypes = all_class_prototypes_list[0]

            pos_w = all_pos_weights_spatial[0].cpu().detach().numpy()
            neg_w = all_neg_weights_spatial[0].cpu().detach().numpy()
            combined_w = all_combined_weights_spatial[0].cpu().detach().numpy()

            if is_3d:
                scale_d = input_size[0] / pos_w.shape[0]
                scale_h = input_size[1] / pos_w.shape[1]
                scale_w = input_size[2] / pos_w.shape[2]
                self.last_pos_weights = zoom(pos_w, (scale_d, scale_h, scale_w), order=1)
                self.last_neg_weights = zoom(neg_w, (scale_d, scale_h, scale_w), order=1)
                self.last_combined_weights = zoom(combined_w, (scale_d, scale_h, scale_w), order=1)
            else:
                scale_h = input_size[0] / pos_w.shape[0]
                scale_w = input_size[1] / pos_w.shape[1]
                self.last_pos_weights = zoom(pos_w, (scale_h, scale_w), order=1)
                self.last_neg_weights = zoom(neg_w, (scale_h, scale_w), order=1)
                self.last_combined_weights = zoom(combined_w, (scale_h, scale_w), order=1)

        cam_np = all_cams.cpu().detach().numpy()
        cam_np = np.maximum(cam_np, 0)

        for i in range(B):
            cam_i = cam_np[i]
            cam_i[cam_i <= 0.5] = 0

            non_zero_mask = cam_i > 0
            if non_zero_mask.any():
                cam_min = cam_i[non_zero_mask].min()
                cam_max = cam_i[non_zero_mask].max()
                if cam_max > cam_min:
                    cam_i[non_zero_mask] = (cam_i[non_zero_mask] - cam_min) / (cam_max - cam_min)

            cam_np[i] = cam_i

        cam_resized = []
        for i in range(B):
            if is_3d:
                scale_d = input_size[0] / cam_np.shape[1]
                scale_h = input_size[1] / cam_np.shape[2]
                scale_w = input_size[2] / cam_np.shape[3]
                cam_i_resized = zoom(cam_np[i], (scale_d, scale_h, scale_w), order=1)
            else:
                scale_h = input_size[0] / cam_np.shape[1]
                scale_w = input_size[1] / cam_np.shape[2]
                cam_i_resized = zoom(cam_np[i], (scale_h, scale_w), order=1)
            cam_resized.append(cam_i_resized)

        cam_resized = np.array(cam_resized)

        return cam_resized

    def __call__(self, input_tensor, targets=None, negative_mask=None,
                 pred_mask_np=None, target_class=None, analysis_mode='misclassification',
                 negative_class=None, aug_smooth=False, eigen_smooth=False):
        return self.forward(input_tensor, targets, negative_mask, pred_mask_np,
                            target_class, analysis_mode, negative_class)

    def __del__(self):
        self._remove_hooks()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._remove_hooks()