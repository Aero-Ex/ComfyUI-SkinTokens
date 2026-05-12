# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Dict

from ..modules.distributions import DiagonalGaussianDistribution
from ...utils.eval import compute_psnr
from ...utils import misc
import numpy as np
from copy import deepcopy


def logits_to_sdf(logits):
    return torch.sigmoid(logits) * 2 - 1

class KLNearFar(nn.Module):
    def __init__(self,
                 near_weight: float = 0.1,
                 kl_weight: float = 1.0,
                 num_near_samples: Optional[int] = None):

        super().__init__()

        self.near_weight = near_weight
        self.kl_weight = kl_weight
        self.num_near_samples = num_near_samples
        self.geo_criterion = nn.BCEWithLogitsLoss()

    def forward(self,
                posteriors: Optional[DiagonalGaussianDistribution],
                logits: torch.FloatTensor,
                labels: torch.FloatTensor,
                split: Optional[str] = "train", **kwargs) -> Tuple[torch.FloatTensor, Dict[str, float]]:

        """

        Args:
            posteriors (DiagonalGaussianDistribution or torch.distributions.Normal):
            logits (torch.FloatTensor): [B, 2*N], logits[:, 0:N] is the volume points; logits[:, N:2N] is the near points;
            labels (torch.FloatTensor): [B, 2*N], labels[:, 0:N] is the volume points; labels[:, N:2N] is the near points;
            split (str):
            **kwargs:

        Returns:
            loss (torch.Tensor): (,)
            log (dict):

        """

        if self.num_near_samples is None:
            num_vol = logits.shape[1] // 2
        else:
            num_vol = logits.shape[1] - self.num_near_samples

        vol_logits = logits[:, 0:num_vol]
        vol_labels = labels[:, 0:num_vol]

        near_logits = logits[:, num_vol:]
        near_labels = labels[:, num_vol:]

        # occupancy loss
        # vol_bce = self.geo_criterion(vol_logits, vol_labels)
        # near_bce = self.geo_criterion(near_logits, near_labels)
        vol_bce = self.geo_criterion(vol_logits.float(), vol_labels.float())
        near_bce = self.geo_criterion(near_logits.float(), near_labels.float())

        if posteriors is None:
            kl_loss = torch.tensor(0.0, dtype=vol_logits.dtype, device=vol_logits.device)
        else:
            kl_loss = posteriors.kl(dims=(1, 2))
            kl_loss = torch.mean(kl_loss)

        loss = vol_bce + near_bce * self.near_weight + kl_loss * self.kl_weight

        with torch.no_grad():
            preds = logits >= 0
            accuracy = (preds == labels).float()
            accuracy = accuracy.mean()
            pos_ratio = torch.mean(labels)

        log = {
            "{}/total_loss".format(split): loss.clone().detach(),
            "{}/near".format(split): near_bce.detach(),
            "{}/far".format(split): vol_bce.detach(),
            "{}/kl".format(split): kl_loss.detach(),
            "{}/accuracy".format(split): accuracy,
            "{}/pos_ratio".format(split): pos_ratio
        }

        if posteriors is not None:
            log[f"{split}/mean"] = posteriors.mean.mean().detach()
            log[f"{split}/std_mean"] = posteriors.std.mean().detach()
            log[f"{split}/std_max"] = posteriors.std.max().detach()

        return loss, log


class KLNearFarColor(nn.Module):
    def __init__(self,
                 near_weight: float = 0.1,
                 kl_weight: float = 1.0,
                 color_weight: float = 1.0,
                 color_criterion: str = "mse",
                 num_near_samples: Optional[int] = None):

        super().__init__()

        self.color_weight = color_weight
        self.near_weight = near_weight
        self.kl_weight = kl_weight
        self.num_near_samples = num_near_samples

        if color_criterion == "mse":
            self.color_criterion = nn.MSELoss()

        elif color_criterion == "l1":
            self.color_criterion = nn.L1Loss()

        else:
            raise ValueError(f"{color_criterion} must be [`mse`, `l1`].")

        self.geo_criterion = nn.BCEWithLogitsLoss()

    def forward(self,
                posteriors: Optional[DiagonalGaussianDistribution],
                logits: torch.FloatTensor,
                labels: torch.FloatTensor,
                pred_colors: torch.FloatTensor,
                gt_colors: torch.FloatTensor,
                split: Optional[str] = "train", **kwargs) -> Tuple[torch.FloatTensor, Dict[str, float]]:

        """

        Args:
            posteriors (DiagonalGaussianDistribution or torch.distributions.Normal):
            logits (torch.FloatTensor): [B, 2*N], logits[:, 0:N] is the volume points; logits[:, N:2N] is the near points;
            labels (torch.FloatTensor): [B, 2*N], labels[:, 0:N] is the volume points; labels[:, N:2N] is the near points;
            pred_colors (torch.FloatTensor): [B, M, 3]
            gt_colors (torch.FloatTensor): [B, M, 3]
            split (str):
            **kwargs:

        Returns:
            loss (torch.Tensor): (,)
            log (dict):

        """

        if self.num_near_samples is None:
            num_vol = logits.shape[1] // 2
        else:
            num_vol = logits.shape[1] - self.num_near_samples

        vol_logits = logits[:, 0:num_vol]
        vol_labels = labels[:, 0:num_vol]

        near_logits = logits[:, num_vol:]
        near_labels = labels[:, num_vol:]

        # occupancy loss
        # vol_bce = self.geo_criterion(vol_logits, vol_labels)
        # near_bce = self.geo_criterion(near_logits, near_labels)
        vol_bce = self.geo_criterion(vol_logits.float(), vol_labels.float())
        near_bce = self.geo_criterion(near_logits.float(), near_labels.float())

        # surface color loss
        color = self.color_criterion(pred_colors, gt_colors)

        if posteriors is None:
            kl_loss = torch.tensor(0.0, dtype=pred_colors.dtype, device=pred_colors.device)
        else:
            kl_loss = posteriors.kl(dims=(1, 2))
            kl_loss = torch.mean(kl_loss)

        loss = vol_bce + near_bce * self.near_weight + color * self.color_weight + kl_loss * self.kl_weight

        with torch.no_grad():
            preds = logits >= 0
            accuracy = (preds == labels).float()
            accuracy = accuracy.mean()
            psnr = compute_psnr(pred_colors, gt_colors)

        log = {
            "{}/total_loss".format(split): loss.clone().detach(),
            "{}/near".format(split): near_bce.detach(),
            "{}/far".format(split): vol_bce.detach(),
            "{}/color".format(split): color.detach(),
            "{}/kl".format(split): kl_loss.detach(),
            "{}/psnr".format(split): psnr.detach(),
            "{}/accuracy".format(split): accuracy
        }

        return loss, log


class ContrastKLNearFar(nn.Module):
    def __init__(self,
                 contrast_weight: float = 1.0,
                 near_weight: float = 0.1,
                 kl_weight: float = 1.0,
                 normal_weight: float = 0.0,
                 surface_weight: float = 0.0,
                 eikonal_weight: float = 0.0,
                 sdf_bce_weight: float = 0.0,
                 sdf_l1l2_weight: float = 1.0,
                 num_near_samples: Optional[int] = None,
                 sdf_trunc_val: float = 0.05,
                 gt_sdf_soft: bool = False,
                 normal_supervision_type: str = "cosine",
                 supervision_type: str = 'occupancy'):

        super().__init__()

        self.labels = None
        self.last_local_batch_size = None
        self.supervision_type = supervision_type

        assert normal_supervision_type in ["l1", "l2", "cosine", "l1_cosine", "l2_cosine", "von_mises"]
        self.normal_supervision_type = normal_supervision_type

        self.contrast_weight = contrast_weight
        self.near_weight = near_weight
        self.kl_weight = kl_weight
        self.normal_weight = normal_weight
        self.surface_weight = surface_weight
        self.eikonal_weight = eikonal_weight
        self.sdf_bce_weight = sdf_bce_weight # only used in sigmoid-sdf
        self.sdf_l1l2_weight = sdf_l1l2_weight # only used in sigmoid-sdf
        self.sdf_trunc_val = sdf_trunc_val
        self.gt_sdf_soft = gt_sdf_soft
        self.num_near_samples = num_near_samples
        self.geo_criterion = nn.BCEWithLogitsLoss()
        self.geo_criterion_sdf = nn.MSELoss()

    def sdf_loss(self, pred_sdf, gt_sdf):
        scaled_sdf = gt_sdf / self.sdf_trunc_val
        greater_mask = scaled_sdf > 1.
        smaller_mask = scaled_sdf < -1.
        inside_mask = 1. - greater_mask - smaller_mask
        greater_loss = F.smooth_l1_loss(F.relu(1. - pred_sdf), torch.zeros_like(pred_sdf), reduction="none") * greater_mask
        smaller_loss = F.smooth_l1_loss(F.relu(pred_sdf + 1.), torch.zeros_like(pred_sdf), reduction="none") * smaller_mask
        inside_loss = F.smooth_l1_loss(pred_sdf, gt_sdf, beta=1e-2, reduction="none") * inside_mask
        loss = (greater_loss + smaller_loss + inside_loss).mean()
        return loss

    def von_mises(self, x, y, k=1):
        cos = F.cosine_similarity(x, y, dim=-1)
        exp = torch.exp(k * (cos - 1))
        return 1 - exp

    def forward(self,
                shape_embed: torch.FloatTensor,
                text_embed: torch.FloatTensor,
                image_embed: torch.FloatTensor,
                logit_scale: torch.FloatTensor,
                posteriors: Optional[DiagonalGaussianDistribution],
                latents: torch.FloatTensor,
                shape_logits: torch.FloatTensor,
                shape_labels: torch.FloatTensor,
                surface_logits: Optional[torch.FloatTensor],
                surface_normals: Optional[torch.FloatTensor],
                gt_surface_normals: Optional[torch.FloatTensor],
                split: Optional[str] = "train", **kwargs):
        if self.supervision_type == 'occupancy':
            shape_logits = shape_logits.squeeze(-1)
            shape_labels[shape_labels>=0] = 1
            shape_labels[shape_labels<0] = 0

        elif self.supervision_type == 'occupancy-shapenet':
            shape_logits = shape_logits.squeeze(-1)

        elif self.supervision_type == 'occupancy-w-surface':
            shape_logits = shape_logits.squeeze(-1)
            shape_labels[shape_labels==10] = 0
            shape_labels[shape_labels>0] = 1
            shape_labels[shape_labels<0] = 0

        elif 'sdf' in self.supervision_type:
            shape_logits = shape_logits.squeeze(-1)
            if self.gt_sdf_soft:
                shape_labels_sdf = torch.tanh(shape_labels / self.sdf_trunc_val)# * self.sdf_trunc_val
            else:
                shape_labels_sdf = torch.clamp(shape_labels, min=-self.sdf_trunc_val, max=self.sdf_trunc_val) / self.sdf_trunc_val
        else:
            raise ValueError(f"Invalid supervision_type {self.supervision_type}")

        local_batch_size = shape_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * misc.get_rank() + torch.arange(
                local_batch_size, device=shape_embed.device
            ).long()
            self.last_local_batch_size = local_batch_size
            

        if text_embed is not None and image_embed is not None:
            # normalized features
            shape_embed = F.normalize(shape_embed, dim=-1, p=2)
            text_embed = F.normalize(text_embed, dim=-1, p=2)
            image_embed = F.normalize(image_embed, dim=-1, p=2)

            # gather features from all GPUs
            shape_embed_all, text_embed_all, image_embed_all = misc.all_gather_batch(
                [shape_embed, text_embed, image_embed]
            )

            # cosine similarity as logits
            logits_per_shape_text = logit_scale * shape_embed @ text_embed_all.t()
            logits_per_text_shape = logit_scale * text_embed @ shape_embed_all.t()
            logits_per_shape_image = logit_scale * shape_embed @ image_embed_all.t()
            logits_per_image_shape = logit_scale * image_embed @ shape_embed_all.t()
            contrast_loss = (F.cross_entropy(logits_per_shape_text, self.labels) +
                            F.cross_entropy(logits_per_text_shape, self.labels)) / 2 + \
                            (F.cross_entropy(logits_per_shape_image, self.labels) +
                            F.cross_entropy(logits_per_image_shape, self.labels)) / 2
        else:
            contrast_loss = torch.tensor(0.0, dtype=shape_logits.dtype, device=shape_logits.device)

        # shape reconstruction
        if self.num_near_samples is None:
            num_vol = shape_logits.shape[1] // 2
        else:
            num_vol = shape_logits.shape[1] - self.num_near_samples

        # occupancy/sdf loss
        if self.supervision_type == 'occupancy' or self.supervision_type == 'occupancy-shapenet':
            vol_logits = shape_logits[:, 0:num_vol]
            vol_labels = shape_labels[:, 0:num_vol]

            near_logits = shape_logits[:, num_vol:]
            near_labels = shape_labels[:, num_vol:]

            vol_loss = self.geo_criterion(vol_logits.float(), vol_labels.float())
            near_loss = self.geo_criterion(near_logits.float(), near_labels.float())

        elif 'sdf' in self.supervision_type:
            if self.supervision_type == "sigmoid-sdf":
                shape_sdfs = logits_to_sdf(shape_logits)
            else:
                shape_sdfs = shape_logits

            vol_logits = shape_logits[:, 0:num_vol]
            vol_sdfs = shape_sdfs[:, 0:num_vol]
            vol_labels_sdf = shape_labels_sdf[:, 0:num_vol]

            near_logits= shape_logits[:, num_vol:]
            near_sdfs = shape_sdfs[:, num_vol:]
            near_labels_sdf = shape_labels_sdf[:, num_vol:]

            # use both sdf loss and occupancy loss
            vol_loss = torch.mean(torch.abs(vol_sdfs - vol_labels_sdf)) + torch.mean((vol_sdfs - vol_labels_sdf) ** 2) #+ self.geo_criterion(vol_logits_sdf, vol_labels)
            near_loss = torch.mean(torch.abs(near_sdfs - near_labels_sdf)) + torch.mean((near_sdfs - near_labels_sdf) ** 2) #+ self.geo_criterion(near_logits_sdf, near_labels)

            if self.supervision_type == "sigmoid-sdf":
                vol_labels = (vol_labels_sdf + 1) / 2
                near_labels = (near_labels_sdf + 1) / 2
                vol_loss = self.sdf_l1l2_weight * vol_loss + self.sdf_bce_weight * self.geo_criterion(vol_logits, vol_labels)
                near_loss = self.sdf_l1l2_weight * near_loss + self.sdf_bce_weight * self.geo_criterion(near_logits, near_labels)
                # print(vol_loss, self.sdf_bce_weight * self.geo_criterion(vol_logits, vol_labels))

        # surface loss
        if "sdf" in self.supervision_type and surface_logits is not None:
            if self.supervision_type == "sigmoid-sdf":
                surface_sdfs = logits_to_sdf(surface_logits)
            else:
                surface_sdfs = surface_logits
            surface_loss = torch.mean(surface_sdfs ** 2)
        else:
            surface_loss = torch.tensor(0.0, dtype=shape_logits.dtype, device=shape_logits.device)

        if surface_normals is not None and gt_surface_normals is not None and "sdf" in self.supervision_type:    

            valid_mask = surface_sdfs.squeeze(-1) < (self.sdf_trunc_val * 0.8)

            if valid_mask is not None:
                surface_normals = surface_normals[valid_mask]
                gt_surface_normals = gt_surface_normals[valid_mask]

            # eikonal loss
            surface_normals_norm = torch.norm(surface_normals, dim=-1)
            eikonal_loss = F.mse_loss(surface_normals_norm * self.sdf_trunc_val, surface_normals_norm.new_ones(surface_normals_norm.shape), reduction="mean")

            # surface normal loss
            # surface_normals = F.normalize(surface_normals, dim=-1)
            surface_normals = surface_normals * self.sdf_trunc_val
            gt_surface_normals = F.normalize(gt_surface_normals, dim=-1)

            if self.normal_supervision_type == "cosine":
                # use cosine similarity loss 
                normal_loss = 1 - F.cosine_similarity(F.normalize(surface_normals, dim=-1), gt_surface_normals, dim=-1).mean()
            elif self.normal_supervision_type == "l1":
                # use l1 loss
                normal_loss = F.l1_loss(surface_normals, gt_surface_normals)
            elif self.normal_supervision_type == "l2":
                normal_loss = F.mse_loss(surface_normals, gt_surface_normals)
            elif self.normal_supervision_type == "von_mises":
                normal_loss = self.von_mises(surface_normals, gt_surface_normals).mean()
            elif self.normal_supervision_type == "l1_cosine":
                normal_loss_cos = 1 - F.cosine_similarity(F.normalize(surface_normals, dim=-1), gt_surface_normals, dim=-1).mean()
                normal_loss_l1 = F.l1_loss(surface_normals, gt_surface_normals)
                normal_loss = normal_loss_cos + normal_loss_l1
            elif self.normal_supervision_type == "l2_cosine":
                normal_loss_cos = 1 - F.cosine_similarity(F.normalize(surface_normals, dim=-1), gt_surface_normals, dim=-1).mean()
                normal_loss_l2 = F.mse_loss(surface_normals, gt_surface_normals)
                normal_loss = normal_loss_cos + normal_loss_l2
            else:
                raise NotImplementedError
        else:
            normal_loss = torch.tensor(0.0, dtype=shape_logits.dtype, device=shape_logits.device)
            eikonal_loss = torch.tensor(0.0, dtype=shape_logits.dtype, device=shape_logits.device)
            surface_normals_norm = torch.tensor(0.0, dtype=shape_logits.dtype, device=shape_logits.device)

        if posteriors is None:
            kl_loss = torch.tensor(0.0, dtype=shape_logits.dtype, device=shape_logits.device)
        else:
            kl_loss = posteriors.kl(dims=(1, 2))
            kl_loss = torch.mean(kl_loss)

        loss = vol_loss + near_loss * self.near_weight + kl_loss * self.kl_weight + contrast_loss * self.contrast_weight + normal_loss * self.normal_weight + self.eikonal_weight * eikonal_loss + self.surface_weight * surface_loss

        # compute accuracy
        with torch.no_grad():
            if "sdf" in self.supervision_type:
                preds = shape_sdfs >= 0
                sdf_labels = shape_labels_sdf >= 0
                accuracy = (preds == sdf_labels).float()
            else:
                preds = shape_logits >= 0
                accuracy = (preds == shape_labels).float()
            accuracy = accuracy.mean()

            log = {
                # "{}/contrast".format(split): contrast_loss.clone().detach(),
                "{}/near".format(split): near_loss.detach(),
                "{}/far".format(split): vol_loss.detach(),
                "{}/normal".format(split): normal_loss.detach(),
                "{}/surface".format(split): surface_loss.detach(),
                "{}/eikonal".format(split): eikonal_loss.detach(),
                "{}/kl".format(split): kl_loss.detach(),
                "{}/surface_grad_norm".format(split): surface_normals_norm.mean().detach(),
                # "{}/shape_text_acc".format(split): shape_text_acc,
                # "{}/shape_image_acc".format(split): shape_image_acc,
                "{}/total_loss".format(split): loss.clone().detach(),
                "{}/accuracy".format(split): accuracy,
            }

            if posteriors is not None:
                log[f"{split}/posteriors_mean"] = posteriors.mean.mean().detach()
                log[f"{split}/posteriors_std_mean"] = posteriors.std.mean().detach()
                log[f"{split}/posteriors_std_max"] = posteriors.std.max().detach()

        return loss, log, near_loss
