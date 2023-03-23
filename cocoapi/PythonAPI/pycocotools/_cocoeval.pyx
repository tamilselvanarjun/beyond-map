'''
Functions to make pycocotools functions fast
'''
from _mask import iou
cimport numpy as np
import numpy as np
cimport cython

def computeIoU(dict _gts, dict _dts, int[:] imgIds, int[:] catIds, int useCats, \
        int maxDets, str iouType):
    '''
        :param imgIds: list of all images
        :param catIds: list of all categories (if useCats is False, then we do not need these)
    '''
    cdef dict all_ious = {}  # return value
    cdef (int, int) key
    cdef int[:] catIdsToUse = catIds if useCats else np.array([-1], dtype=np.int32)
    cdef list gt, dt
    cdef list g, d
    cdef dict _g, _d, o
    cdef list iscrowd
    cdef int[:] inds

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

            if len(gt) == 0 and len(dt) == 0:
                all_ious[key] = []
                continue

            inds = np.argsort([-_d['score'] for _d in dt], kind='mergesort').astype(np.int32)
            dt = [dt[i] for i in inds]
            if len(dt) > maxDets:
                dt=dt[0:maxDets]

            if iouType == 'segm':
                g = [_g['segmentation'] for _g in gt]
                d = [_d['segmentation'] for _d in dt]
            elif iouType == 'bbox':
                g = [_g['bbox'] for _g in gt]
                d = [_d['bbox'] for _d in dt]
            else:
                raise Exception('unknown iouType for iou computation')
            # compute iou between each dt and gt region
            iscrowd = [int(o['iscrowd']) for o in gt]
            ious = iou(d, g, iscrowd)
            all_ious[key] = ious    
    return all_ious 

def evaluateImg(int[:] catIds, int[:] imgIds, float[:, :] areaRng, int useCats, dict _gts, dict _dts, int maxDet, float[:] iouThrs, dict all_ious):
    '''
    perform evaluation for single category and image
    :return: dict (single image results)
    '''
    cdef int[:] catIdsToUse = catIds if useCats else np.array([-1], dtype=np.int32)
    cdef int catId, imgId
    cdef float[:] aRng
    cdef list gt, dt
    cdef list ret_dict = []
    cdef int[:] gtind, dtind
    cdef list iscrowd
    cdef double[:, :] ious

    cdef int T, G, D
    cdef long[:, :] gtm, dtm
    cdef int[:] gtIg
    cdef int[:, :] dtIg
    cdef np.ndarray[np.uint8_t, ndim=2, cast=True] dtIgtmp
    cdef int tind, dind
    cdef int m
    cdef dict d, g
    cdef float t
    # cdef np[:, :] a
    cdef np.ndarray[np.uint8_t, ndim=2, cast=True] a
    cdef int valid_ioumat
    cdef int cId

    T = len(iouThrs)

    for catId in catIdsToUse:
        for aRng in areaRng:
            for imgId in imgIds:
                # begin evaluation
                if useCats:
                    gt = _gts.get((imgId, catId), [])
                    dt = _dts.get((imgId, catId), [])
                else:
                    gt = [_ for cId in catIds for _ in _gts.get((imgId, cId), [])]
                    dt = [_ for cId in catIds for _ in _dts.get((imgId, cId), [])]
                    
                if len(gt) == 0 and len(dt) == 0:
                    ret_dict.append(None)
                    continue

                for g in gt:
                    if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                        g['_ignore'] = int(1)
                    else:
                        g['_ignore'] = int(0)
                
                gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort').astype(np.int32)
                gt = [gt[i] for i in gtind]
                dtind = np.argsort([-d['score'] for d in dt], kind='mergesort').astype(np.int32)
                dt = [dt[i] for i in dtind[0:maxDet]]
                iscrowd = [int(o['iscrowd']) for o in gt]
                # get ious
                ious = all_ious[imgId, catId][:, gtind] if len(all_ious[imgId, catId]) > 0 else np.array([[]], dtype=np.float64)
                valid_ioumat = int(len(all_ious[imgId, catId]) > 0)
                G = len(gt)
                D = len(dt)
                gtm  = np.zeros((T,G), dtype=np.int64)
                dtm  = np.zeros((T,D), dtype=np.int64)
                gtIg = np.array([g['_ignore'] for g in gt], dtype=np.int32)
                dtIg = np.zeros((T,D), dtype=np.int32)
                if not (not valid_ioumat or ious.shape[0] == 0):
                    # for tind, t in enumerate(p.iouThrs):
                    for tind in range(T):
                        t = iouThrs[tind]
                        for dind in range(D):
                            d = dt[dind]
                            # information about best match so far (m=-1 -> unmatched)
                            iou = float(min([t, 1-1e-10]))
                            m   = -1
                            for gind in range(G):
                                g = gt[gind]
                                # if this gt already matched, and not a crowd, continue
                                if gtm[tind,gind]>0 and not iscrowd[gind]:
                                    continue
                                # if dt matched to reg gt, and on ignore gt, stop
                                if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                                    break
                                # continue to next gt unless better match made
                                if ious[dind,gind] < iou:
                                    continue
                                # if match successful and best so far, store appropriately
                                iou=ious[dind,gind]
                                m=gind
                            # if match made store id of match for both dt and gt
                            if m == -1:
                                continue
                            dtIg[tind,dind] = gtIg[m]
                            dtm[tind,dind]  = gt[m]['id']
                            gtm[tind,m]     = d['id']
                # set unmatched detections outside of area range to ignore
                a = np.array([[d['area']<aRng[0] or d['area']>aRng[1] for d in dt]], dtype=bool)
                dtIgbool = np.logical_or(np.asarray(dtIg)>0, np.logical_and(np.asarray(dtm)==0, np.repeat(a,T,0)))

                # store results for given image and category
                ret_dict.append({
                        'image_id':     imgId,
                        'category_id':  catId,
                        'aRng':         list(aRng),
                        'maxDet':       maxDet,
                        'dtIds':        [d['id'] for d in dt],
                        'gtIds':        [g['id'] for g in gt],
                        'dtMatches':    np.array(dtm),
                        'gtMatches':    np.array(gtm),
                        'dtScores':     [d['score'] for d in dt],
                        'gtIgnore':     np.array(gtIg),
                        'dtIgnore':     dtIgbool, 
                        'ious'    :     np.array(ious) if valid_ioumat else [],
                    })
    return ret_dict
