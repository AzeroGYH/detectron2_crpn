# -*- coding: utf-8 -*-
#
# Modified by GYH
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN

def add_textdetection_config(cfg):
    """
    Add config for VideoText.
    """
    cfg.INPUT.HFLIP_TRAIN = False
    cfg.MODEL.CROP = CN()
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.CROP_INSTANCE = False
    cfg.INPUT.CROP.SIZE = [0.1, 0.1]

    cfg.INPUT.ROTATE = CN({"ENABLED": False})
    cfg.INPUT.ROTATE.ANGLE_TRAIN = (-15,15)
    cfg.INPUT.ROTATE.ROTATE_TRAIN_PRO = 1.0
    cfg.INPUT.ROTATE.ANGLE_TRAIN_SAMPLING = "range"
    
    # cfg.MODEL.SparseRCNN = CN()
    # cfg.MODEL.SparseRCNN.NUM_CLASSES = 80
    # cfg.MODEL.SparseRCNN.NUM_PROPOSALS = 300

    # # RCNN Head.
    # cfg.MODEL.SparseRCNN.NHEADS = 8
    # cfg.MODEL.SparseRCNN.DROPOUT = 0.0
    # cfg.MODEL.SparseRCNN.DIM_FEEDFORWARD = 2048
    # cfg.MODEL.SparseRCNN.ACTIVATION = 'relu'
    # cfg.MODEL.SparseRCNN.HIDDEN_DIM = 256
    # cfg.MODEL.SparseRCNN.NUM_CLS = 1
    # cfg.MODEL.SparseRCNN.NUM_REG = 3
    # cfg.MODEL.SparseRCNN.NUM_HEADS = 6

    # # Dynamic Conv.
    # cfg.MODEL.SparseRCNN.NUM_DYNAMIC = 2
    # cfg.MODEL.SparseRCNN.DIM_DYNAMIC = 64

    # # Loss.
    # cfg.MODEL.SparseRCNN.CLASS_WEIGHT = 2.0
    # cfg.MODEL.SparseRCNN.GIOU_WEIGHT = 2.0
    # cfg.MODEL.SparseRCNN.L1_WEIGHT = 5.0
    # cfg.MODEL.SparseRCNN.DEEP_SUPERVISION = True
    # cfg.MODEL.SparseRCNN.NO_OBJECT_WEIGHT = 0.1

    # # Focal Loss.
    # cfg.MODEL.SparseRCNN.USE_FOCAL = True
    # cfg.MODEL.SparseRCNN.ALPHA = 0.25
    # cfg.MODEL.SparseRCNN.GAMMA = 2.0
    # cfg.MODEL.SparseRCNN.PRIOR_PROB = 0.01

    # # Optimizer.
    # cfg.SOLVER.OPTIMIZER = "ADAMW"
    # cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
