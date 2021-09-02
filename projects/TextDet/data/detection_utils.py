import logging

import numpy as np
import torch

from detectron2.data import transforms as T
from detectron2.data.detection_utils import \
    annotations_to_instances as d2_anno_to_inst
from detectron2.data.detection_utils import \
    transform_instance_annotations as d2_transform_inst_anno

from .augmentation import (
    RandomRotationWithProb
)
from detectron2.structures import (
    Boxes,
    BoxMode,
    Instances,
)


def bezier_to_polygon(bezier):
    u = np.linspace(0, 1, 20)
    bezier = bezier.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
    points = np.outer((1 - u) ** 3, bezier[:, 0]) \
        + np.outer(3 * u * ((1 - u) ** 2), bezier[:, 1]) \
        + np.outer(3 * (u ** 2) * (1 - u), bezier[:, 2]) \
        + np.outer(u ** 3, bezier[:, 3])

    # convert points to polygon
    points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)
    return torch.from_numpy(points)


def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):
    annotation = d2_transform_inst_anno(
        annotation,
        transforms,
        image_size,
        keypoint_hflip_indices=keypoint_hflip_indices,
    )
    if "beziers" in annotation:
        beziers = transform_beziers_annotations(
            annotation["beziers"], transforms)
        annotation["beziers"] = beziers
        if "points" in annotation:
            print("points================================")
            points = transform_points_annotations(
                annotation["points"], transforms)
            annotation["points"] = points
        old_bbox = annotation["bbox"]

        if (old_bbox[0] == 0. and old_bbox[2] == 0.) or (old_bbox[1] == 0. and old_bbox[3] == 0.):
            # print("annotation", annotation)
            return annotation
        else:
            # recompute box
            # print("recompute")
            curves = bezier_to_polygon(beziers)
            new_box = [torch.min(curves[:, 0]), torch.min(
                curves[:, 1]), torch.max(curves[:, 0]), torch.max(curves[:, 1])]
            annotation["bbox"] = new_box
            return annotation
            
    # just for only points
    if "points" in annotation:
        # print("pointspointspointspoints")
        points = transform_points_annotations(
            annotation["points"], transforms)
        annotation["points"] = points
        old_bbox = annotation["bbox"]
        if (old_bbox[0] == 0. and old_bbox[2] == 0.) or (old_bbox[1] == 0. and old_bbox[3] == 0.):
            return annotation
        else:
            points = torch.from_numpy(points.reshape(-1,2))
            new_box = [torch.min(points[:, 0]), torch.min(
                points[:, 1]), torch.max(points[:, 0]), torch.max(points[:, 1])]
            annotation["bbox"] = new_box
            return annotation
        return annotation
    return annotation


def transform_beziers_annotations(beziers, transforms):
    """
    Transform keypoint annotations of an image.

    Args:
        beziers (list[float]): Nx16 float in Detectron2 Dataset format.
        transforms (TransformList):
    """
    # (N*2,) -> (N, 2)
    beziers = np.asarray(beziers, dtype="float64").reshape(-1, 2)
    beziers = transforms.apply_coords(beziers).reshape(-1)

    # This assumes that HorizFlipTransform is the only one that does flip
    do_hflip = (
        sum(isinstance(t, T.HFlipTransform)
            for t in transforms.transforms) % 2 == 1
    )
    if do_hflip:
        raise ValueError(
            "Flipping text data is not supported (also disencouraged).")

    return beziers

def transform_points_annotations(points, transforms):
    """
    Transform keypoint annotations of an image.

    Args:
        points (list[float]): Nx40 float in Detectron2 Dataset format.
        transforms (TransformList):
    """
    # (N*2,) -> (N, 2)
    points = np.asarray(points, dtype="float64").reshape(-1, 2)
    points = transforms.apply_coords(points).reshape(-1)

    # This assumes that HorizFlipTransform is the only one that does flip
    do_hflip = (
        sum(isinstance(t, T.HFlipTransform)
            for t in transforms.transforms) % 2 == 1
    )
    if do_hflip:
        raise ValueError(
            "Flipping text data is not supported (also disencouraged).")

    return points



def annotations_to_instances(annos, image_size, mask_format="polygon"):
    instance = d2_anno_to_inst(annos, image_size, mask_format)

    if not annos:
        return instance

    # add attributes
    if "beziers" in annos[0]:
        beziers = [obj.get("beziers", []) for obj in annos]
        instance.beziers = torch.as_tensor(beziers, dtype=torch.float32)
    
    if "points" in annos[0]:
        points = [obj.get("points", []) for obj in annos]
        instance.points = torch.as_tensor(points, dtype=torch.float32)

    if "rec" in annos[0]:
        text = [obj.get("rec", []) for obj in annos]
        instance.text = torch.as_tensor(text, dtype=torch.int32)

    return instance


def build_augmentation(cfg, is_train):
    """
    With option to don't use hflip

    Returns:
        list[Augmentation]
    """
    augmentation = []
    is_rotate = cfg.INPUT.ROTATE.ENABLED
    rotate_probability = cfg.INPUT.ROTATE.ROTATE_TRAIN_PRO
    # add rotate aug by GYH
    if is_rotate and is_train:
        angle = cfg.INPUT.ROTATE.ANGLE_TRAIN
        sample_style = cfg.INPUT.ROTATE.ANGLE_TRAIN_SAMPLING
        if sample_style == "range":
            assert (
                len(angle) == 2
            ), "more than 2 ({}) angle(s) are provided for ranges".format(len(angle))
        augmentation.append(RandomRotationWithProb(
            prob=rotate_probability, angle=angle, sample_style=sample_style))

    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert (
            len(min_size) == 2
        ), "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)

    augmentation.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        if cfg.INPUT.HFLIP_TRAIN:
            augmentation.append(T.RandomFlip())
        logger.info("Augmentations used in training: " + str(augmentation))
    
    
    return augmentation


build_transform_gen = build_augmentation
"""
Alias for backward-compatibility.
"""
