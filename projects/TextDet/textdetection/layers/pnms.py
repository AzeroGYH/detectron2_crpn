# -*- coding: utf-8 -*-

import numpy as np
import torch
from shapely.geometry import *

def iou_filter(beziers, scores, filter_inds):
    if beziers.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=beziers.device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    else:
        # max_coordinate = beziers.max()
        # offsets = idxs.to(beziers) * (max_coordinate + torch.tensor(1).to(beziers))
        # beziers_for_nms = beziers + offsets[:, None]

        scores = scores.cpu().numpy()
        beziers = beziers.cpu().numpy()
        pts = []
        for bezier in beziers:
            pt = bezier_to_polygon(bezier)
            # sample 14 points
            pt_t = sampe_points(pt[0:20])
            pt_b = sampe_points(pt[20:])
            pt_t.extend(pt_b)
            pt = pt_t
            pts.append(pt)
        pts = np.array(pts)

        areas = np.zeros(scores.shape)
        # 得分降序
        order = scores.argsort()[::-1]
        inter_areas = np.zeros((scores.shape[0], scores.shape[0]))

        for il in range(len(pts)):
            # 当前点集组成多边形，并计算该多边形的面积
            poly = Polygon(pts[il]).buffer(0.000001)
            areas[il] = poly.area

            # 多剩余的进行遍历
            for jl in range(il, len(pts)):
                polyj = Polygon(pts[jl]).buffer(0.000001)

                # 计算两个多边形的交集，并计算对应的面积
                inS = poly.intersection(polyj)
                inter_areas[il][jl] = inS.area
                inter_areas[jl][il] = inS.area

        # 下面做法和nms一样
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            ovr = inter_areas[i][order[1:]] / (areas[i])
            inds = np.where(ovr <= 0.95)[0]
            order = order[inds + 1]
        # print(keep)
        return torch.tensor(keep)



def remove_unvalid_polygon(boxes, beziers, scores, filter_inds):
    device = boxes.device
    boxes = boxes.cpu().numpy()
    beziers = beziers.cpu().numpy()
    scores = scores.cpu().numpy()
    filter_inds = filter_inds.cpu().numpy()
    new_boxes = []
    new_beziers = []
    new_scores = []
    new_filter_inds = []

    for i in range(len(beziers)):
        poly = bezier_to_polygon(beziers[i])
        try:
            pdet = Polygon(poly)
        except:
            # print('not a valid polygon!!!!')
            continue
        # The polygon should be valid.
        if not pdet.is_valid: 
            # print('polygon has intersection sides!!!!')
            continue
        # pRing = LinearRing(poly)
        # if pRing.is_ccw:
        #     print('polygon not clockwise!!!!')
        #     continue
    
        new_boxes.append(boxes[i])
        new_beziers.append(beziers[i])
        new_scores.append(scores[i])
        new_filter_inds.append(filter_inds[i])
    if len(new_filter_inds) == 0:
        new_boxes = torch.empty(size= (0,4), device=device) 
        new_beziers = torch.empty(size= (0,16), device=device) 
        new_filter_inds= torch.empty(size= (0,2), device=device, dtype=torch.int64) 
    else:
        new_boxes = torch.tensor(new_boxes, device=device)
        new_beziers = torch.tensor(new_beziers, device=device)
        new_filter_inds = torch.tensor(new_filter_inds, device=device)
    return new_boxes, new_beziers, torch.tensor(new_scores, device=device), new_filter_inds

def remove_unvalid_polygon_crpn(beziers):
    device = beziers.device
    beziers = beziers.cpu().numpy()
    keep = []

    for i in range(len(beziers)):
        poly = bezier_to_polygon(beziers[i])
        try:
            pdet = Polygon(poly)
        except:
            continue
        # The polygon should be valid.
        if not pdet.is_valid: 
            continue
    
        keep.append(i)

    return torch.tensor(keep)



def bezier_to_polygon(bezier):
    u = np.linspace(0, 1, 20)
    bezier = bezier.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
    points = np.outer((1 - u) ** 3, bezier[:, 0]) \
        + np.outer(3 * u * ((1 - u) ** 2), bezier[:, 1]) \
        + np.outer(3 * (u ** 2) * (1 - u), bezier[:, 2]) \
        + np.outer(u ** 3, bezier[:, 3])

    # convert points to polygon
    points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)
    return points.tolist()

def sampe_points(pts):
    new_pts = []
    new_pts.append(pts[0])
    new_pts.append(pts[3])
    new_pts.append(pts[6])
    new_pts.append(pts[9])
    new_pts.append(pts[13])
    new_pts.append(pts[16])
    new_pts.append(pts[19])

    return new_pts

def pnms(beziers_for_nms, scores, iou_threshold):
    # 获取检测坐标点及对应的得分
    scores = scores.cpu().numpy()
    beziers = beziers_for_nms.cpu().numpy()
    pts = []
    for bezier in beziers:
        pt = bezier_to_polygon(bezier)
        # sample 14 points
        # pt_t = sampe_points(pt[0:20])
        # pt_b = sampe_points(pt[20:])
        # pt_t.extend(pt_b)
        # pt = pt_t
        pts.append(pt)
    pts = np.array(pts)

    delete_inds = []
    for i, poly in enumerate(pts):
        if not Polygon(poly).is_valid:
            delete_inds.append(i)
    pts = np.delete(pts, delete_inds, 0)
    scores = np.delete(scores, delete_inds, 0)

    areas = np.zeros(scores.shape)
    # 得分降序
    order = scores.argsort()[::-1]
    inter_areas = np.zeros((scores.shape[0], scores.shape[0]))

    for il in range(len(pts)):
        # 当前点集组成多边形，并计算该多边形的面积
        poly = Polygon(pts[il])
        areas[il] = poly.area

        # 多剩余的进行遍历
        for jl in range(il, len(pts)):
            polyj = Polygon(pts[jl])

            # 计算两个多边形的交集，并计算对应的面积
            inS = poly.intersection(polyj)
            inter_areas[il][jl] = inS.area
            inter_areas[jl][il] = inS.area

    # 下面做法和nms一样
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = inter_areas[i][order[1:]] / (areas[i] + areas[order[1:]] - inter_areas[i][order[1:]])
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    # print(keep)
    return torch.tensor(keep)


def batched_pnms(
    beziers: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float
):
    """
    Performs polygon non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.
    """
    assert beziers.shape[-1] == 16

    if beziers.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=beziers.device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    else:
        # max_coordinate = beziers.max()
        # offsets = idxs.to(beziers) * (max_coordinate + torch.tensor(1).to(beziers))
        # beziers_for_nms = beziers + offsets[:, None]
        keep = pnms(beziers, scores, iou_threshold)
        return keep


