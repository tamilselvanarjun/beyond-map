'''
Functions to make pycocotools functions fast
'''
from _mask import iou
import numpy as np
cimport cython

def computeIoU(dict _gts, dict _dts, int[:] imgIds, int[:] catIds, int useCats, \
        int maxDets, str iouType):
    '''
        imgIds: list of all images
        catIds: list of all categories (if useCats is False, then we do not need these)
    '''
    cdef dict all_ious = {}  # return value
    cdef (int, int) key
    cdef int[:] catIdsToUse = catIds if useCats else np.array([-1], dtype=np.int32)
    cdef list gt, dt
    cdef list iscrowd

    for imgId in imgIds:
        for catId in catIdsToUse:
            if useCats:
                key = (imgId, catId)
                gt = _gts.get(key, [])
                dt = _dts.get(key, [])
            else:
                # No catIDs to use here
                key = (imgId, -1)
                gt = [_ for cId in catIds for _ in _gts.get((imgId, cId), [])]
                dt = [_ for cId in catIds for _ in _dts.get((imgId, cId), [])]

            if len(gt) == 0 or len(dt) == 0:
                all_ious[key] = []
                continue

            inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
            dt = [dt[i] for i in inds]
            if len(dt) > maxDets:
                dt=dt[0:maxDets]

            if iouType == 'segm':
                g = [g['segmentation'] for g in gt]
                d = [d['segmentation'] for d in dt]
            elif iouType == 'bbox':
                g = [g['bbox'] for g in gt]
                d = [d['bbox'] for d in dt]
            else:
                raise Exception('unknown iouType for iou computation')
            # compute iou between each dt and gt region
            iscrowd = [int(o['iscrowd']) for o in gt]
            ious = iou(d, g, iscrowd)
            all_ious[key] = ious    

    return all_ious 
