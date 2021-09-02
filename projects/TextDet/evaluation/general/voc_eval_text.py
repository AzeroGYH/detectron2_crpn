import numpy as np
import logging
from shapely.geometry import Polygon,LinearRing
from multiprocessing import Pool
logger = logging.getLogger(__name__)

# polynms
def py_cpu_pnms(pts, scores, thresh=0.2):
    assert len(pts) == len(scores), 'py_cpu_pnms'
    areas = np.zeros(scores.shape)
    order = scores.argsort()[::-1]
    inter_areas = np.zeros((scores.shape[0], scores.shape[0]))
    for il in range(len(pts)):
        poly = Polygon(pts[il])
        areas[il] = poly.area
        for jl in range(il, len(pts)):
            polyj = Polygon(pts[jl])
            try:
                inS = poly.intersection(polyj)
            except:
                # print(poly, polyj)
                pass
            inter_areas[il][jl] = inS.area
            inter_areas[jl][il] = inS.area

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = inter_areas[i][order[1:]] / \
            (areas[i] + areas[order[1:]] - inter_areas[i][order[1:]])
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def format(res, pnms=True):
    # img_fn, (det_bboxes, polys) = res
    # scores = det_bboxes[:, -1]
    # print(res)
    polys, scores = res["instances"]["polys"], res["instances"]["scores"]
    polys = np.array(polys)
    scores = np.array(scores)
    # print("polys_", polys.shape)
    # print("scores_", scores)
    # polys = polys.reshape(-1, polys.shape[1] // 2, 2).round().astype(np.int)
    # print("polys_", polys.shape)
    # nps
    delete_inds = []
    for i, poly in enumerate(polys):
        try:
            pdet = Polygon(poly)
            if not pdet.is_valid:
                # print(poly)
                print("not valid")
                delete_inds.append(i)
            # pRing = LinearRing(poly)
            # if pRing.is_ccw:
            #     delete_inds.append(i)
        except:
            # print(poly)
            print("except")
            delete_inds.append(i)
       
        
    # # print("polys_",len(polys))
    # # print("scores_",len(scores))
    polys = np.delete(polys, delete_inds, 0)
    scores = np.delete(scores, delete_inds, 0)
    # print("polys",len(polys))
    # print("scores",len(scores))
    if len(polys) == 0 or len(scores) == 0:
        print("-------------")
    assert len(polys) == len(scores), 'format'
    if pnms:
        keep = py_cpu_pnms(polys, scores)
        polys, scores = polys[keep], scores[keep]
    return dict(polys=polys, scores=scores)


def to_eval_format(res, pnms=True, nproc=48):
    """convert detections to eval format."""
    pool = Pool(nproc)
    new_res = pool.starmap(
        format,
        zip(res, [pnms for _ in range(len(res))]))
    pool.close()
    return new_res


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_intersection(pD, pG):
    pInt = pD & pG
    if pInt.length == 0:
        return 0
    return pInt.area


def get_union(pD, pG):
    areaA = pD.area
    areaB = pG.area
    return areaA + areaB - get_intersection(pD, pG)


def get_intersection_over_union(pD, pG):
    try:
        return get_intersection(pD, pG) / get_union(pD, pG)
    except:
        return 0


def get_intersection_over_detection(pD, pG):
    try:
        return get_intersection(pD, pG) / pD.area
    except:
        return 0


def tpfp_voc(det_polys,
             det_scores,
             gt_polys,
             gt_polys_ignore,
             iou_thr=0.5,
             ignore_mode='iod'):
    assert len(det_polys) == len(det_scores)
    assert ignore_mode in ['iou', 'iod']
    # match ignore
    ignore_inds = []
    for i, det in enumerate(det_polys):
        plg_det = Polygon(det)
        for gt in gt_polys_ignore:
            plg_gt = Polygon(gt)
            if ignore_mode == 'iou':
                overlap = get_intersection_over_union(plg_det, plg_gt)
            else:
                overlap = get_intersection_over_detection(plg_det, plg_gt)
            if overlap > iou_thr:
                ignore_inds.append(i)
                break
    # tpfp match
    ndet = len(det_polys)
    ngt = len(gt_polys)
    match = np.zeros(ngt, np.bool)
    tp = np.zeros(ndet)
    fp = np.zeros(ndet)
    for i, det in enumerate(det_polys):
        if i in ignore_inds:
            continue
        plg_det = Polygon(det)
        overlaps = np.zeros(ngt)
        for j, gt in enumerate(gt_polys):
            plg_gt = Polygon(gt)
            overlaps[j] = get_intersection_over_union(plg_det, plg_gt)
        if ngt > 0:
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
            if ovmax > iou_thr:
                if not match[jmax]:
                    tp[i] = 1
                    match[jmax] = 1
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1
    return tp, fp


def get_results(det_results, annotations):
    det_polys = []
    det_scores = []
    gt_polys = []
    gt_polys_ignore = []
    for det in det_results:
        det_polys.append(det['polys'])
        det_scores.append(det['scores'])
    for ann in annotations:
        gt_polys.append([poly.reshape(-1, 2) for poly in ann['polys']])
        gt_polys_ignore.append([poly.reshape(-1, 2)
                                for poly in ann['polys_ignore']])
    return det_polys, det_scores, gt_polys, gt_polys_ignore


def eval_fmeasure(det_results,
                  annotations,
                  iou_thr=0.5,
                  ignore_mode='iod',
                  logger=None,
                  nproc=48,
                  use_07_metric=False):
    assert len(det_results) == len(annotations)
    num_imgs = len(det_results)
    det_polys, det_scores, gt_polys, gt_polys_ignore = \
        get_results(det_results, annotations)
    num_det = np.array([len(score) for score in det_scores]).sum()
    num_gt = np.array([len(poly) for poly in gt_polys]).sum()

    pool = Pool(nproc)
    tpfp = pool.starmap(
        tpfp_voc,
        zip(det_polys, det_scores, gt_polys, gt_polys_ignore,
            [iou_thr for _ in range(num_imgs)],
            [ignore_mode for _ in range(num_imgs)]))
    tp, fp = tuple(zip(*tpfp))
    pool.close()

    # sort all det bboxes by score, also sort tp and fp
    det_scores = np.hstack(det_scores)
    sort_inds = np.argsort(-det_scores)
    det_scores = det_scores[sort_inds]
    tp = np.hstack(tp)[sort_inds]
    fp = np.hstack(fp)[sort_inds]

    # calculate recall and precision with tp and fp
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    eps = np.finfo(np.float32).eps
    recalls = tp / np.maximum(num_gt, eps)
    precisions = tp / np.maximum((tp + fp), eps)
    ap = voc_ap(recalls, precisions, use_07_metric)
    fmeasures = 2 * recalls * precisions / \
        np.maximum((recalls + precisions), eps)
    max_fmeasure = np.max(fmeasures) if len(fmeasures) > 0 else 0.
    jmax = np.argmax(fmeasures) if len(fmeasures) > 0 else 0.
    maxf_thr = det_scores[jmax] if len(fmeasures) > 0 else 0.
    maxf_prec = precisions[jmax] if len(fmeasures) > 0 else 0.
    maxf_rec = recalls[jmax] if len(fmeasures) > 0 else 0.

    log = f'------------------------------' \
        f'\nEvaluation results:' \
        f'\nrecall: {maxf_rec:.4f}' \
        f'\nprecision: {maxf_prec:.4f}' \
        f'\nhmean: {max_fmeasure:.4f}' \
        f'\nmaxf_thr: {maxf_thr:.4f}' \
        f'\n------------------------------'
    print(log)
    det_only_methodMetrics = r"DETECTION_ONLY_RESULTS: precision: {}, recall: {}, hmean: {}, ap: {}, maxf_thr: {}".format(str(maxf_prec), str(maxf_rec), str(max_fmeasure), str(ap), str(maxf_thr))
    resDict = {'calculated':True,'Message':'','det_only_method': det_only_methodMetrics}
    return resDict
    # eval_results = {'num_gt': num_gt,
    #                 'num_det': num_det,
    #                 'ap': ap,
    #                 'recall': maxf_rec,
    #                 'precision': maxf_prec,
    #                 'hmean': max_fmeasure,
    #                 'maxf_thr': maxf_thr}
    # return eval_results

# def get_txts(formatted_results):
#     path = "/data2/gyh/Codes/AdelaiDet/ctw_txts_test/"
#     i = 1001
#     for res in formatted_results:
#         txt_name = path +"000" + str(i) + '.txt'
#         i = i + 1
#         polys = res['polys']
#         lines = ""
#         for poly in polys:
#             poly = np.array(poly)
#             poly = poly.reshape(1,-1)[0]
#             poly = poly.tolist()
#             poly = list(map(str,poly))
#             # poly = list(map(str,list(map(int,poly))))
#             line = ",".join(poly) + '\n'
#             lines += line
#             # print(lines)
#         with open(txt_name,'w') as w_f:
#             w_f.write(lines)



# additional
def evaluate_best(results,
                  annotations,
                  metric='fmeasure',
                  logger=None,
                  iou_thr=0.5,
                  ignore_mode='iod'):
    """Evaluate text detection in VOC protocol."""
    assert ignore_mode in ['iou', 'iod']
    # annotations = [self.get_ann_info(i) for i in range(len(self))]
    formatted_results = to_eval_format(results)
    print("formatted_results",len(formatted_results))
    print("annotations",len(annotations))
    # print("formatted_results",formatted_results)

    # get_txts(formatted_results)

    eval_results = eval_fmeasure(
        formatted_results,
        annotations,
        iou_thr=iou_thr,
        ignore_mode=ignore_mode,
        logger=logger)
    return eval_results
