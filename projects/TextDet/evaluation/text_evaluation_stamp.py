import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import re
import torch
from collections import OrderedDict
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

from detectron2.utils import comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator

import cv2
import glob
import shutil
from shapely.geometry import Polygon, LinearRing
from .rotate_icdar13 import script
import zipfile

from scipy.spatial import distance as dist

import random

def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")


class StampEvaluator(DatasetEvaluator):
    """
    Evaluate text proposals and recognition.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self._tasks = ("polygon", "recognition")
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            raise AttributeError(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'."
            )

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # use dataset_name to decide eval_gt_path
        if "totaltext" in dataset_name:
            self._text_eval_gt_path = "datasets/evaluation/gt_totaltext.zip"
            self._word_spotting = True
        elif "ctw1500" in dataset_name:
            self._text_eval_gt_path = "datasets/evaluation/gt_ctw1500.zip"
            self._word_spotting = False
        elif "icdar13" in dataset_name:
            self._text_eval_gt_path = "datasets/evaluation/rotate_icdar13/gt_0.zip"
            self._word_spotting = False
        elif "icdar15_test_rotate30" in dataset_name:
            self._text_eval_gt_path = "/data1/gyh/datasets/rotate_ic15/gt_ic15_rotate30.zip"
            self._word_spotting = False
        elif "icdar15_test_rotate45" in dataset_name:
            self._text_eval_gt_path = "/data1/gyh/datasets/rotate_ic15/gt_ic15_rotate45.zip"
            self._word_spotting = False
        elif "icdar15_test_rotate60" in dataset_name:
            self._text_eval_gt_path = "/data1/gyh/datasets/rotate_ic15/gt_ic15_rotate60.zip"
            self._word_spotting = False
        elif "icdar15" in dataset_name:
            self._text_eval_gt_path = "/data1/gyh/Codes/AdelaiDet/datasets/evaluation/icdar15/icdar15_gt.zip"
            self._word_spotting = False
        elif "td500" in dataset_name:
            self._text_eval_gt_path = "/data1/gyh/datasets/MSRA-TD500/gt_td500.zip"
            self._word_spotting = False
        elif "mlt" in dataset_name:
            self._text_eval_gt_path = "/data4/gyh/17MLT/mlt17_val_gt.zip"
            print(self._text_eval_gt_path)
            self._word_spotting = False
        elif "stamp" in dataset_name:
            self._text_eval_gt_path = "/data3/gyh/codes/stamp/detectron2/datasets/Stamp/stamp_gt.zip"
            print(self._text_eval_gt_path)
            self._word_spotting = False
        self._text_eval_confidence = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST

        self.temp_dataset_save_path = cfg.DATASETS.TEST[0]

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            # print(output)
            instances = output["instances"].to(self._cpu_device)
            prediction["instances"] = instances_to_coco_json_without_recs(instances, input["image_id"])
            self._predictions.append(prediction)

    def to_eval_format(self, file_path, temp_dir="temp_det_results", cf_th=0.5):
        with open(file_path, 'r') as f:
            data = json.load(f)
            with open('temp_all_det_cors.txt', 'w') as f2:
                for ix in range(len(data)):
                    if data[ix]['score'] > 0.1:
                        outstr = '{}: '.format(data[ix]['image_id'])
                        xmin = 1000000
                        ymin = 1000000
                        xmax = 0 
                        ymax = 0

                        pts = []
                        for i in range(len(data[ix]['polys'])):
                            pts.append([int(data[ix]['polys'][i][0]),int(data[ix]['polys'][i][1])])
                        pts = np.array(pts)
                        # box = pts
                        rect = cv2.minAreaRect(pts)
                        box = cv2.boxPoints(rect)

                        box = order_points(box)
                        box = np.int0(box)

                        for i in range(len(box)):
                            outstr = outstr + str(int(box[i][0])) +','+str(int(box[i][1])) +','
                        outstr = outstr + str(round(data[ix]['score'], 3)) +',####'+'null'+'\n'	
                        f2.writelines(outstr)
                f2.close()
        dirn = temp_dir
        lsc = [cf_th] 
        fres = open('temp_all_det_cors.txt', 'r').readlines()
        for isc in lsc:	
            if not os.path.isdir(dirn):
                os.mkdir(dirn)

            for line in fres:
                line = line.strip()
                s = line.split(': ')
                # filename = 'res_img_{:05d}.txt'.format(int(s[0]))  # FOR MLT17
                filename = 'res_img_{}.txt'.format(int(s[0]))
                outName = os.path.join(dirn, filename)
                with open(outName, 'a') as fout:
                    ptr = s[1].strip().split(',####')
                    score = ptr[0].split(',')[-1]
                    if float(score) < isc:
                        continue
                    cors = ','.join(e for e in ptr[0].split(',')[:-1])
                    fout.writelines(cors+',####'+ptr[1]+'\n')
        os.remove("temp_all_det_cors.txt")

    def sort_detection(self, temp_dir):
        origin_file = temp_dir
        random_d = str(random.randint(1,100))
        output_file = "final_ic15_"+ random_d+"_"+temp_dir

        if not os.path.isdir(output_file):
            os.mkdir(output_file)

        files = glob.glob(origin_file+'*.txt')
        files.sort()

        for i in files:
            out = i.replace(origin_file, output_file)
            fin = open(i, 'r').readlines()
            fout = open(out, 'w')
            for iline, line in enumerate(fin):
                line = line.replace('\x00', '')
                ptr = line.strip().split(',####')
                rec  = ptr[1]
                cors = ptr[0].split(',')
                # print(ptr)
                assert(len(cors) %2 == 0), 'cors invalid.'
                pts = [(int(cors[j]), int(cors[j+1])) for j in range(0,len(cors),2)]
                # print(pts)
                try:
                    pgt = Polygon(pts)
                except Exception as e:
                    print(e)
                    print('An invalid detection in {} line {} is removed ... '.format(i, iline))
                    continue
                
                if not pgt.is_valid:
                    print('An invalid detection in {} line {} is removed ... '.format(i, iline))
                    continue
                    
                # pRing = LinearRing(pts)
                # if pRing.is_ccw:
                #     pts.reverse()
                outstr = ''
                for ipt in pts[:-1]:
                    outstr += (str(int(ipt[0]))+','+ str(int(ipt[1]))+',')
                outstr += (str(int(pts[-1][0]))+','+ str(int(pts[-1][1])))
                outstr = outstr #  # FOR MLT17 + ',' + '1.0' #+',' + rec
                fout.writelines(outstr+'\n')
            fout.close()
        os.chdir(output_file)

        def zipdir(path, ziph):
            # ziph is zipfile handle
            for root, dirs, files in os.walk(path):
                for file in files:
                    ziph.write(os.path.join(root, file))

        zipf = zipfile.ZipFile('../'+self.temp_dataset_save_path + random_d +'_det.zip', 'w', zipfile.ZIP_DEFLATED)
        zipdir('./', zipf)
        zipf.close()
        os.chdir("../")
        # clean temp files
        shutil.rmtree(origin_file)
        shutil.rmtree(output_file)
        return self.temp_dataset_save_path + random_d +'_det.zip'
    
    def evaluate_with_official_code(self, result_path, gt_path):
        return script.text_eval_main(det_file=result_path, gt_file=gt_path, is_word_spotting=self._word_spotting)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        PathManager.mkdirs(self._output_dir)

        file_path = os.path.join(self._output_dir, "text_results.json")
        self._logger.info("Saving results to {}".format(file_path))
        with PathManager.open(file_path, "w") as f:
            f.write(json.dumps(coco_results))
            f.flush()

        self._results = OrderedDict()
        
        # eval text
        temp_dir = "temp_det_results/"
        self.to_eval_format(file_path, temp_dir, self._text_eval_confidence)
        result_path = self.sort_detection(temp_dir)
        text_result = self.evaluate_with_official_code(result_path, self._text_eval_gt_path)
        # os.remove(result_path)

        # parse
        for task in ("method",):
            result = text_result[task]

            self._results[task] = result

        return copy.deepcopy(self._results)


def instances_to_coco_json(instances, img_id):
    num_instances = len(instances)
    if num_instances == 0:
        return []

    scores = instances.scores.tolist()
    beziers = instances.beziers.numpy()
    recs = instances.recs.numpy()

    results = []
    for bezier, rec, score in zip(beziers, recs, scores):
        # convert beziers to polygons
        poly = bezier_to_polygon(bezier)
        s = decode(rec)
        result = {
            "image_id": img_id,
            "category_id": 1,
            "polys": poly,
            "rec": s,
            "score": score,
        }
        results.append(result)
    return results

# GYH
def instances_to_coco_json_without_recs(instances, img_id):
    num_instances = len(instances)
    if num_instances == 0:
        return []
    scores = instances.scores.tolist()
    boxes = instances.pred_boxes.tensor.numpy() 

    results = []
    for score, box in zip(scores, boxes):
        temp_box = box.tolist()
        temp_poly = [temp_box[0],temp_box[1],temp_box[2],temp_box[1],temp_box[2],temp_box[3],temp_box[0],temp_box[3]]
        poly = np.array(temp_poly).reshape(-1,2).tolist()
        result = {
            "image_id": img_id,
            "category_id": 1,
            "polys": poly,
            "score": score,
        }
        results.append(result)
    return results


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

def box_to_polygon(box):
    box = box.tolist()
    x1, y1, x4, y4 = box[0], box[1], box[2], box[3]
    x3, y3 = x4, y1
    x6, y6 = x1, y4
    x2, y2 = x1 + (x4 - x1)/2, y1
    x5, y5 = x1 + (x4 - x1)/2, y4

    return [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6]]



CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']


def ctc_decode(rec):
    # ctc decoding
    last_char = False
    s = ''
    for c in rec:
        c = int(c)
        if c < 95:
            if last_char != c:
                s += CTLABELS[c]
                last_char = c
        elif c == 95:
            s += u'口'
        else:
            last_char = False
    return s


def decode(rec):
    s = ''
    for c in rec:
        c = int(c)
        if c < 95:
            s += CTLABELS[c]
        elif c == 95:
            s += u'口'

    return s
            