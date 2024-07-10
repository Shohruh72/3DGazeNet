import os
import cv2
import torch
import numpy as np
from skimage import transform as trans


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        self.num = self.num + n
        self.sum = self.sum + v * n
        self.avg = self.sum / self.num


def strip_optimizer(filename):
    x = torch.load(filename, map_location=torch.device('cpu'))
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, filename)


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    # translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2,
                                                output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data,
                             M, (output_size, output_size),
                             borderValue=0.0)
    return cropped, M


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts


def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)


def angles_from_vec(vec):
    x, y, z = -vec[2], vec[1], -vec[0]
    theta = np.arctan2(y, x)
    phi = np.arctan2(np.sqrt(x ** 2 + y ** 2), z) - np.pi / 2
    theta_x, theta_y = phi, theta
    return theta_x, theta_y


def vec_from_eye(eye, iris_lms_idx):
    p_iris = eye[iris_lms_idx] - eye[:32].mean(axis=0)
    vec = p_iris.mean(axis=0)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def angles_and_vec_from_eye(eye, iris_lms_idx):
    vec = vec_from_eye(eye, iris_lms_idx)
    theta_x, theta_y = angles_from_vec(vec)
    return theta_x, theta_y, vec


def draw_item(iris_idx_481, tri481, eimg, item):
    num_eye = 481
    eye_kps = item
    eye_l = eye_kps[:num_eye, :]
    eye_r = eye_kps[num_eye:, :]
    for _eye in [eye_l, eye_r]:
        tmp = _eye[:, 0].copy()
        _eye[:, 0] = _eye[:, 1].copy()
        _eye[:, 1] = tmp

    for _eye in [eye_l, eye_r]:
        _kps = _eye[iris_idx_481, :].astype(int)
        for l in range(_kps.shape[0]):
            color = (0, 255, 0)
            cv2.circle(eimg, (_kps[l][1], _kps[l][0]), 4, color, 4)
        for _tri in tri481:
            color = (0, 0, 255)
            for k in range(3):
                ix = _tri[k]
                iy = _tri[(k + 1) % 3]
                x = _eye[ix, :2].astype(int)[::-1]
                y = _eye[iy, :2].astype(int)[::-1]
                cv2.line(eimg, x, y, color, 1)

    theta_x_l, theta_y_l, vec_l = angles_and_vec_from_eye(eye_l, iris_idx_481)
    theta_x_r, theta_y_r, vec_r = angles_and_vec_from_eye(eye_r, iris_idx_481)
    gaze_pred = np.array([(theta_x_l + theta_x_r) / 2, (theta_y_l + theta_y_r) / 2])

    diag = np.sqrt(float(eimg.shape[0] * eimg.shape[1]))

    eye_pos_left = eye_l[iris_idx_481].mean(axis=0)[[0, 1]]
    eye_pos_right = eye_r[iris_idx_481].mean(axis=0)[[0, 1]]

    gaze_pred = np.array([theta_x_l, theta_y_l])
    dx = 0.4 * diag * np.sin(gaze_pred[1])
    dy = 0.4 * diag * np.sin(gaze_pred[0])
    x = np.array([eye_pos_left[1], eye_pos_left[0]])
    y = x.copy()
    y[0] += dx
    y[1] += dy
    x, y = x.astype(int), y.astype(int)
    cv2.line(eimg, x, y, (0, 255, 255), 2)

    gaze_pred = np.array([theta_x_r, theta_y_r])
    dx = 0.4 * diag * np.sin(gaze_pred[1])
    dy = 0.4 * diag * np.sin(gaze_pred[0])
    x = np.array([eye_pos_right[1], eye_pos_right[0]])
    y = x.copy()
    y[0] += dx
    y[1] += dy
    x, y = x.astype(int), y.astype(int)
    cv2.line(eimg, x, y, (0, 255, 255), 2)
    return eimg


def draw_on(iris_idx_481, tri481, eimg, results):
    face_sizes = [(x[0][2] - x[0][0]) for x in results]
    max_index = np.argmax(face_sizes)
    max_face_size = face_sizes[max_index]
    rescale = 300.0 / max_face_size
    oimg = eimg.copy()
    eimg = cv2.resize(eimg, None, fx=rescale, fy=rescale)
    for pred in results:
        _, _, eye_kps = pred
        eye_kps = eye_kps.copy()
        eye_kps *= rescale
        eimg = draw_item(iris_idx_481, tri481, eimg, eye_kps)
    eimg = cv2.resize(eimg, (oimg.shape[1], oimg.shape[0]))
    return eimg

    pred_max = results[max_index]
    bbox, kps, eye_kps = pred_max
    width = bbox[2] - bbox[0]
    center = (kps[0] + kps[1]) / 2.0
    _size = max(width / 1.5, np.abs(kps[1][0] - kps[0][0])) * 1.5
    rotate = 0
    _scale = args.input_size / _size
    aimg, M = face_align.transform(oimg, center, args.input_size, _scale, rotate)
    eye_kps = face_align.trans_points(eye_kps, M)
    center_eye_rescale = 4.0
    aimg = cv2.resize(aimg, None, fx=center_eye_rescale, fy=center_eye_rescale)
    eye_kps *= center_eye_rescale
    aimg = draw_item(aimg, eye_kps)

    rimg = np.zeros((max(eimg.shape[0], aimg.shape[0]), eimg.shape[1] + aimg.shape[1], 3), dtype=np.uint8)
    rimg[:eimg.shape[0], :eimg.shape[1], :] = eimg
    rimg[:aimg.shape[0], eimg.shape[1]:eimg.shape[1] + aimg.shape[1], :] = aimg
    return rimg


class FaceDetector:
    def __init__(self, onnx_file=None, session=None):
        from onnxruntime import InferenceSession
        self.session = session

        self.batched = False
        if self.session is None:
            assert onnx_file is not None
            assert os.path.exists(onnx_file)
            self.session = InferenceSession(onnx_file, providers=['CUDAExecutionProvider'])

        self.nms_thresh = 0.4
        self.center_cache = {}
        self._init_vars()

    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = input_name
        self.output_names = output_names
        self.use_kps = False
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def forward(self, image, thresh):
        scores_list = []
        bboxes_list = []
        kps_list = []
        input_size = tuple(image.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(image, 1.0 / 128, input_size,
                                     (127.5, 127.5, 127.5), swapRB=True)

        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        fmc = self.fmc
        input_width = blob.shape[3]
        input_height = blob.shape[2]

        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                scores = net_outs[idx][0]
                boxes = net_outs[idx + fmc][0]
                boxes = boxes * stride
                if self.use_kps:
                    points = net_outs[idx + fmc * 2][0] * stride
            # If model doesn't support batching take output as is
            else:
                scores = net_outs[idx]
                boxes = net_outs[idx + fmc]
                boxes = boxes * stride
                if self.use_kps:
                    points = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_index = np.where(scores >= thresh)[0]
            bboxes = distance2bbox(anchor_centers, boxes)
            pos_scores = scores[pos_index]
            pos_bboxes = bboxes[pos_index]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, points)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                kps_list.append(kpss[pos_index])
        return scores_list, bboxes_list, kps_list

    def detect(self, image, thresh=0.5, input_size=None, max_num=0, metric='default'):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size

        im_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / image.shape[0]
        resized_img = cv2.resize(image, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kps_list = self.forward(det_img, thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kps_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None

        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = image.shape[0] // 2, image.shape[1] // 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])

            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            index = np.where(ovr <= thresh)[0]
            order = order[index + 1]

        return keep
