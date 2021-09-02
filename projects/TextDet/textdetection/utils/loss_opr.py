import torch
# from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F
from detectron2.layers import nonzero_tuple

def softmax_loss(score, label, ignore_label=-1):
    # print(score)
    with torch.no_grad():
        max_score, _ = score.max(axis=1, keepdims=True)
    score -= max_score
    log_prob = score - torch.log(torch.exp(score).sum(axis=1, keepdims=True))
    mask = label != ignore_label
    vlabel = label * mask
    onehot = torch.zeros(vlabel.shape[0], 2 , device=score.device)
    onehot.scatter_(1, vlabel.reshape(-1, 1), 1)
    loss = -(log_prob * onehot).sum(axis=1)
    loss = loss * mask
    return loss
def softmax_loss_same(score, label, ignore_label=-1, index=0):
    # print(score)
    if index == 0:
        label = label.reshape(-1,2)
        label = label[:,0].unsqueeze(1)
        label = torch.cat([label,label],dim=1)
        label = label.flatten()
    else:
        label = label.reshape(-1,2)
        label = label[:,1].unsqueeze(1)
        label = torch.cat([label,label],dim=1)
        label = label.flatten()

    with torch.no_grad():
        max_score,_ = score.max(axis=1, keepdims=True)
    score -= max_score
    log_prob = score - torch.log(torch.exp(score).sum(axis=1, keepdims=True))
    mask = label != ignore_label
    vlabel = label * mask
    onehot = torch.zeros(vlabel.shape[0], 2 , device=score.device)
    onehot.scatter_(1, vlabel.reshape(-1, 1), 1)
    loss = -(log_prob * onehot).sum(axis=1)
    loss = loss * mask
    return loss

def smooth_l1_loss(pred, target, beta: float):
    if beta < 1e-5:
        loss = torch.abs(pred - target)
    else:
        abs_x = torch.abs(pred- target)
        in_mask = abs_x < beta
        loss = torch.where(in_mask, 0.5 * abs_x ** 2 / beta, abs_x - 0.5 * beta)
    return loss.sum(axis=1)

def _softmax_cross_entropy_loss(_no_instances, pred_class_logits, gt_classes):
    """
    Compute the softmax cross entropy loss for box classification.

    Returns:
        scalar Tensor
    """
    if _no_instances:
        return 0.0 * pred_class_logits.sum()
    else:
        # self._log_accuracy()
        cls_loss = F.cross_entropy(pred_class_logits, gt_classes, reduction="none")
        return cls_loss


def _box_reg_loss(_no_instances, pred_class_logits, pred_proposal_deltas, proposals, 
                    gt_boxes, gt_classes, box2box_transform, smooth_l1_beta, fg_inds):
    """
    Compute the smooth L1 loss for box regression.

    Returns:
        scalar Tensor
    """
    if _no_instances:
        return 0.0 * pred_proposal_deltas.sum()

    box_dim = gt_boxes.tensor.size(1)  # 4 or 5
    cls_agnostic_bbox_reg = pred_proposal_deltas.size(1) == box_dim
    device = pred_proposal_deltas.device

    # bg_class_ind = pred_class_logits.shape[1] - 1

    # fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < bg_class_ind))[0]
    if cls_agnostic_bbox_reg:
        # pred_proposal_deltas only corresponds to foreground class for agnostic
        gt_class_cols = torch.arange(box_dim, device=device)
    else:
        fg_gt_classes = gt_classes[fg_inds]
        # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
        # where b is the dimension of box representation (4 or 5)
        # Note that compared to Detectron1,
        # we do not perform bounding box regression for background classes.
        gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

    #"smooth_l1":
    target_proposals = proposals.tensor.repeat(1, 2).reshape(-1, proposals.tensor.shape[-1])
    # print(proposals.tensor.shape)
    # print(target_proposals.shape)
    # print(gt_boxes.tensor.shape)
    gt_proposal_deltas = box2box_transform.get_deltas(
        target_proposals, gt_boxes.tensor
    )
    loss_box_reg = smooth_l1_loss(
        pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
        gt_proposal_deltas[fg_inds],
        smooth_l1_beta,
    )

    # loss_box_reg = loss_box_reg * self.box_reg_loss_weight / self.gt_classes.numel()
    return loss_box_reg

def _box_reg_loss_same(_no_instances, pred_class_logits, pred_proposal_deltas, proposals, 
                    gt_boxes, gt_classes, box2box_transform, smooth_l1_beta, fg_inds, index):
    """
    Compute the smooth L1 loss for box regression.

    Returns:
        scalar Tensor
    """
    if _no_instances:
        return 0.0 * pred_proposal_deltas.sum()

    box_dim = gt_boxes.tensor.size(1)  # 4 or 5
    cls_agnostic_bbox_reg = pred_proposal_deltas.size(1) == box_dim
    device = pred_proposal_deltas.device
    
    if index == 0:
        gt_boxes = gt_boxes.tensor.flatten().reshape(-1,8)[:,:4]
        gt_boxes = torch.cat([gt_boxes,gt_boxes],dim=1).reshape(-1,4)
    else:
        gt_boxes = gt_boxes.tensor.flatten().reshape(-1,8)[:,4:]
        gt_boxes = torch.cat([gt_boxes,gt_boxes],dim=1).reshape(-1,4)

    # bg_class_ind = pred_class_logits.shape[1] - 1

    # fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < bg_class_ind))[0]
    if cls_agnostic_bbox_reg:
        # pred_proposal_deltas only corresponds to foreground class for agnostic
        gt_class_cols = torch.arange(box_dim, device=device)
    else:
        fg_gt_classes = gt_classes[fg_inds]
        # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
        # where b is the dimension of box representation (4 or 5)
        # Note that compared to Detectron1,
        # we do not perform bounding box regression for background classes.
        gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

    #"smooth_l1":
    target_proposals = proposals.tensor.repeat(1, 2).reshape(-1, proposals.tensor.shape[-1])
    # print(proposals.tensor.shape)
    # print(target_proposals.shape)
    # print(gt_boxes.tensor.shape)
    gt_proposal_deltas = box2box_transform.get_deltas(
        target_proposals, gt_boxes
    )
    loss_box_reg = smooth_l1_loss(
        pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
        gt_proposal_deltas[fg_inds],
        smooth_l1_beta,
    )

    # loss_box_reg = loss_box_reg * self.box_reg_loss_weight / self.gt_classes.numel()
    return loss_box_reg

def compute_beziers_targets(src_boxes, target_beziers):
    """
    Compute beziers targets for bezier control points.

    Args:
        src_boxes: dim=4
        target_beziers: dim=16

    Returns:
        scalar Tensor
    """
    assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
    assert isinstance(target_beziers, torch.Tensor), type(target_beziers)

    widths = src_boxes[:, 2] - src_boxes[:, 0]
    heights = src_boxes[:, 3] - src_boxes[:, 1]
    ctr_x = src_boxes[:, 0] + 0.5 * widths
    ctr_y = src_boxes[:, 1] + 0.5 * heights

    xy_deltas = []
    for index in range(0,16):
        if index % 2 == 0:
            xy_deltas.append((target_beziers[:, index] - ctr_x) / widths)
        else:
            xy_deltas.append((target_beziers[:, index] - ctr_y) / heights)

    deltas = torch.stack(tuple(xy_deltas), dim=1)
    return deltas

def _bezier_reg_loss(_no_instances, pred_class_logits, pred_bezier_deltas, proposals, 
                    gt_beziers, gt_classes, box2box_transform, smooth_l1_beta, fg_inds):
    """
    Compute the smooth L1 loss for bezier control points.

    Returns:
        scalar Tensor
    """
    if _no_instances:
        return 0.0 * pred_bezier_deltas.sum()

    box_dim = 16  # 16
    cls_agnostic_bbox_reg = True # GYH pred_bezier_deltas.size(1) == box_dim
    device = pred_bezier_deltas.device

    # bg_class_ind = pred_class_logits.shape[1] - 1

    # fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < bg_class_ind))[0]
    if cls_agnostic_bbox_reg:
        # pred_proposal_deltas only corresponds to foreground class for agnostic
        gt_class_cols = torch.arange(box_dim, device=device)
    else:
        fg_gt_classes = gt_classes[fg_inds]
        # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
        # where b is the dimension of box representation (4 or 5)
        # Note that compared to Detectron1,
        # we do not perform bounding box regression for background classes.
        gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

    target_proposals = proposals.tensor.repeat(1, 2).reshape(-1, proposals.tensor.shape[-1])

    gt_bezier_deltas = compute_beziers_targets(
        target_proposals, gt_beziers
    )
    loss_bezier_reg = smooth_l1_loss(
        pred_bezier_deltas[fg_inds[:, None], gt_class_cols],
        gt_bezier_deltas[fg_inds],
        smooth_l1_beta,
    )

    # loss_bezier_reg = loss_bezier_reg * self.box_reg_loss_weight / self.gt_classes.numel()
    return loss_bezier_reg
def _bezier_reg_loss_same(_no_instances, pred_class_logits, pred_bezier_deltas, proposals, 
                    gt_beziers, gt_classes, box2box_transform, smooth_l1_beta, fg_inds, index):
    """
    Compute the smooth L1 loss for bezier control points.

    Returns:
        scalar Tensor
    """
    if _no_instances:
        return 0.0 * pred_bezier_deltas.sum()

    box_dim = 16  # 16
    cls_agnostic_bbox_reg = True # GYH pred_bezier_deltas.size(1) == box_dim
    device = pred_bezier_deltas.device

    if index == 0:
        gt_beziers = gt_beziers.flatten().reshape(-1,32)[:,:16]
        gt_beziers = torch.cat([gt_beziers,gt_beziers],dim=1).reshape(-1,16)
    else:
        gt_beziers = gt_beziers.flatten().reshape(-1,32)[:,16:]
        gt_beziers = torch.cat([gt_beziers,gt_beziers],dim=1).reshape(-1,16)

    # bg_class_ind = pred_class_logits.shape[1] - 1

    # fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < bg_class_ind))[0]
    if cls_agnostic_bbox_reg:
        # pred_proposal_deltas only corresponds to foreground class for agnostic
        gt_class_cols = torch.arange(box_dim, device=device)
    else:
        fg_gt_classes = gt_classes[fg_inds]
        # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
        # where b is the dimension of box representation (4 or 5)
        # Note that compared to Detectron1,
        # we do not perform bounding box regression for background classes.
        gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

    target_proposals = proposals.tensor.repeat(1, 2).reshape(-1, proposals.tensor.shape[-1])

    gt_bezier_deltas = compute_beziers_targets(
        target_proposals, gt_beziers
    )
    loss_bezier_reg = smooth_l1_loss(
        pred_bezier_deltas[fg_inds[:, None], gt_class_cols],
        gt_bezier_deltas[fg_inds],
        smooth_l1_beta,
    )

    # loss_bezier_reg = loss_bezier_reg * self.box_reg_loss_weight / self.gt_classes.numel()
    return loss_bezier_reg

def crpn_compute_beziers_targets(src_boxes, src_beziers, target_beziers):
        """
        Compute beziers targets for bezier control points.

        Args:
            src_beziers: dim=16
            target_beziers: dim=16

        Returns:
            scalar Tensor
        """
        # print("src_boxes",src_boxes.shape)
        # print("src_beziers",src_beziers.shape)
        # print("target_beziers",target_beziers.shape)
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(src_beziers, torch.Tensor), type(src_beziers)
        assert isinstance(target_beziers, torch.Tensor), type(target_beziers)

        widths = src_boxes[:, 2] - src_boxes[:, 0]
        heights = src_boxes[:, 3] - src_boxes[:, 1]

        xy_deltas = []
        for index in range(0,16):
            if index % 2 == 0:
                xy_deltas.append((target_beziers[:, index] - src_beziers[:, index]) / widths)
            else:
                xy_deltas.append((target_beziers[:, index] - src_beziers[:, index]) / heights)

        deltas = torch.stack(tuple(xy_deltas), dim=1)
        return deltas

def _crpn_bezier_reg_loss(_no_instances, pred_class_logits, pred_bezier_deltas, proposals, 
                    proposals_curves, gt_beziers, gt_classes, box2box_transform, smooth_l1_beta, fg_inds):
    """
    Compute the smooth L1 loss for bezier control points.

    Returns:
        scalar Tensor
    """
    if _no_instances:
        return 0.0 * pred_bezier_deltas.sum()

    box_dim = 16  # 16
    cls_agnostic_bbox_reg = True  # GYH self.pred_bezier_deltas.size(1) == box_dim
    device = pred_bezier_deltas.device

    if cls_agnostic_bbox_reg:
        # pred_proposal_deltas only corresponds to foreground class for agnostic
        gt_class_cols = torch.arange(box_dim, device=device)
    else:
        fg_gt_classes = gt_classes[fg_inds]
        # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
        # where b is the dimension of box representation (4 or 5)
        # Note that compared to Detectron1,
        # we do not perform bounding box regression for background classes.
        gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

    proposals_boxes = proposals.tensor.repeat(1, 2).reshape(-1, proposals.tensor.shape[-1])
    proposals_curves_caluate = proposals_curves.repeat(1, 2).reshape(-1, proposals_curves.shape[-1])

    gt_bezier_deltas = crpn_compute_beziers_targets(
        proposals_boxes, proposals_curves_caluate, gt_beziers
    )
    loss_bezier_reg = smooth_l1_loss(
        pred_bezier_deltas[fg_inds[:, None], gt_class_cols],
        gt_bezier_deltas[fg_inds],
        smooth_l1_beta,
    )

    # loss_bezier_reg = loss_bezier_reg * self.box_reg_loss_weight / self.gt_classes.numel()
    return loss_bezier_reg

