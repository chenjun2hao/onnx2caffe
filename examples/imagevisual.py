import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def concat_imgs(img_list, ori=None, pad=0, pad_value=0):
    img_list = [img.image if hasattr(img, 'image') else img for img in img_list]
    if len(img_list) == 1:
        return img_list[0]
    if ori == 'hor':
        height = max([img.shape[0] for img in img_list])
        width = sum([img.shape[1] for img in img_list]) + pad * (len(img_list) - 1)
        new_img = np.full((height, width, img_list[0].shape[2]), pad_value, dtype=img_list[0].dtype)
        begin = 0
        for img in img_list:
            end = begin + img.shape[1]
            new_img[:img.shape[0], begin:end] = img
            begin = end + pad
    elif ori == 'ver':
        height = sum([img.shape[0] for img in img_list]) + pad * (len(img_list) - 1)
        width = max([img.shape[1] for img in img_list])
        new_img = np.full((height, width, img_list[0].shape[2]), pad_value, dtype=img_list[0].dtype)
        begin = 0
        for img in img_list:
            end = begin + img.shape[0]
            new_img[begin:end, :img.shape[1]] = img
            begin = end + pad
    else:
        assert False
    return new_img


class ImageVisualizer(object):
    """Visualizer that draws data on image."""

    def __init__(self, image, save_image_dir='', colormap=None):
        self.image = image  # BGR
        self.save_image_dir = save_image_dir
        self.colormap = colormap

    @property
    def shape(self):
        return self.image.shape

    def copy(self):
        return type(self)(self.image.copy(), save_image_dir=self.save_image_dir, colormap=self.colormap)

    def save(self, save_name):
        cv2.imwrite(os.path.join(self.save_image_dir, save_name), self.image)

    def resize(self, val, mode, interp=cv2.INTER_LINEAR):
        if mode == 'long':
            scale = val / np.max(self.shape[:2])
        elif mode == 'short':
            scale = val / np.min(self.shape[:2])
        elif mode == 'height':
            scale = val / self.shape[0]
        elif mode == 'width':
            scale = val / self.shape[1]
        elif mode == 'fixed_scale':
            scale = val
        elif mode == 'fixed_size':
            scale = (float(val[0]) / self.shape[0], float(val[1]) / self.shape[1])
        else:
            assert False
        if not isinstance(scale, (tuple, list)):
            scale = (scale, scale)
        self.image = cv2.resize(self.image, None, None, fx=scale[1], fy=scale[0], interpolation=interp)
        return self

    def new_visulizer(self, image, **kwargs):
        if isinstance(image, list):
            image = concat_imgs(image, **kwargs)
        return type(self)(image, save_image_dir=self.save_image_dir, colormap=self.colormap)

    def set_colormap(self, method, num_colors):
        if method == 1:
            cmap = plt.cm.get_cmap('hsv', num_colors)
            colormap = []
            for i in range(num_colors):
                color = tuple(cmap(i))
                colormap.append(tuple([color[j] * 255 for j in range(3)]))
        elif method == 2:
            color_per_class = 255 * 255 * 255 // (num_colors + 1)
            colormap = []
            for i in range(1, num_colors + 1):
                color_i = color_per_class * i
                b, g, r = color_i // 255 // 255, (color_i // 255) % 255, color_i % 255
                colormap.append((int(b), int(g), int(r)))
        else:
            assert False
        assert len(colormap) == num_colors
        self.colormap = colormap

    def _get_color(self, i, color=None):
        if color is None:
            color = self.colormap
        if isinstance(color, list):
            color = color[int(i) % len(color)]
        assert isinstance(color, tuple) and len(color) == 3
        return color

    def draw_points(self, points, radius=1, color=None, thickness=1, draw_num=False):
        # points: (num_points, 2 or 3)
        if len(points) == 0:
            return
        points = points.astype(np.int32)
        for i, point in enumerate(points):
            if len(point) == 3:
                x, y, z = point
                color_i = self._get_color(z, color)
            elif len(point) == 2:
                x, y = point
                color_i = self._get_color(i, color)
            else:
                assert False
            cv2.circle(self.image, center=(x, y), radius=radius, color=color_i, thickness=thickness)
            if draw_num:
                cv2.putText(self.image, '%d' % i, (x, y - 3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, color_i, 1)

    def draw_boxes(self, boxes, classes=None, color=None, thickness=2):
        # boxes: (num_boxes, 4)
        if len(boxes) == 0:
            return
        boxes = boxes.astype(np.int32)
        for i, box in enumerate(boxes):
            i = classes[i] - 1 if classes is not None else i
            color_i = self._get_color(i, color)
            cv2.rectangle(self.image, (box[0], box[1]), (box[2], box[3]), color=color_i, thickness=thickness)

    def draw_poly_boxes(self, boxes, classes=None, color=None, thickness=2):
        # boxes: (num_boxes, n * 2)
        if len(boxes) == 0:
            return
        boxes = boxes.astype(np.int32)
        for i, box in enumerate(boxes):
            i = classes[i] - 1 if classes is not None else i
            color_i = self._get_color(i, color)
            poly_box = box.reshape((-1, 2))
            cv2.polylines(self.image, [poly_box], isClosed=True, color=color_i, thickness=thickness)

    def draw_corners(self, corners, classes=None, color=None, thickness=2, draw_num=False):
        # corners: (num_boxes, 8, 2)
        if len(corners) == 0:
            return
        assert corners.shape[1] == 8 and corners.shape[2] == 2
        corners = corners.astype(np.int32)
        for i, corner in enumerate(corners):
            i = classes[i] - 1 if classes is not None else i
            color_i = self._get_color(i, color)
            for k in range(4):
                i, j = k, (k + 1) % 4
                cv2.line(self.image, (corner[i, 0], corner[i, 1]), (corner[j, 0], corner[j, 1]), color_i, thickness)
                i, j = k + 4, (k + 1) % 4 + 4
                cv2.line(self.image, (corner[i, 0], corner[i, 1]), (corner[j, 0], corner[j, 1]), color_i, thickness)
                i, j = k, k + 4
                cv2.line(self.image, (corner[i, 0], corner[i, 1]), (corner[j, 0], corner[j, 1]), color_i, thickness)
            if draw_num:
                for k in range(8):
                    cv2.putText(self.image, '%d' % k, (corner[k, 0], corner[k, 1] - 3),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, color_i, 1)

    def _attach_color_to_seg(self, seg, color=None):
        if seg.ndim == 3:
            assert seg.shape[2] == 3
            return seg
        seg_img = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        min_ind = int(seg.min())
        max_ind = int(seg.max())
        for i in range(min_ind, max_ind + 1):
            if i <= 0:
                continue
            color_i = self._get_color(i - 1, color)
            seg_img[seg == i, 0] = color_i[0]
            seg_img[seg == i, 1] = color_i[1]
            seg_img[seg == i, 2] = color_i[2]
        return seg_img

    def draw_seg(self, seg, color=None, scale=0.7):
        seg_img = self._attach_color_to_seg(seg, color)
        mask = seg > 0
        mask = mask[:, :, np.newaxis]
        self.image = self.image - scale * self.image * mask + scale * seg_img * mask
        self.image = np.clip(self.image, 0, 255).astype(np.uint8)

    def draw_keypoints(self, keypoints, skeleton=None, point_color=(0, 255, 255),
                       skeleton_color=None, kps_thresh=0, kps_show_num=False):
        # keypoints: (num_boxes, num_kps * 3)
        if len(keypoints) == 0:
            return
        keypoints = keypoints.reshape((keypoints.shape[0], -1, 3))
        for i in range(keypoints.shape[0]):
            for j in range(keypoints.shape[1]):
                x = int(keypoints[i, j, 0] + 0.5)
                y = int(keypoints[i, j, 1] + 0.5)
                v = keypoints[i, j, 2]
                if kps_thresh < v < 3:
                    color_j = self._get_color(j, point_color)
                    cv2.circle(self.image, (x, y), radius=2, color=color_j, thickness=2)
                    if kps_show_num:
                        # cv2.putText(self.image, '%.2f' % v, (x+3, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, color, 1)
                        cv2.putText(self.image, '%d' % j, (x + 3, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, color_j, 1)
            if skeleton is not None:
                for j in range(skeleton.shape[0]):
                    p1 = skeleton[j, 0]
                    p2 = skeleton[j, 1]
                    x1 = int(keypoints[i, p1, 0] + 0.5)
                    y1 = int(keypoints[i, p1, 1] + 0.5)
                    v1 = keypoints[i, p1, 2]
                    x2 = int(keypoints[i, p2, 0] + 0.5)
                    y2 = int(keypoints[i, p2, 1] + 0.5)
                    v2 = keypoints[i, p2, 2]
                    if kps_thresh < v1 < 3 and kps_thresh < v2 < 3:
                        color_j = self._get_color(j, skeleton_color)
                        cv2.line(self.image, (x1, y1), (x2, y2), color=color_j, thickness=2)

    def draw_coco_keypoints(self, keypoints, skeleton=None, skeleton_color=None, **kwargs):
        # keypoints: (num_boxes, num_kps * 3)
        if skeleton is None:
            # skeleton = np.array([[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            #                      [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            #                      [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]) - 1
            kps_info = keypoints_utils.get_coco_kps_info()
            names, lines = kps_info['names'], kps_info['lines']
            num_boxes = len(keypoints)
            num_kps = len(names)
            keypoints = keypoints.reshape((num_boxes, num_kps, 3))
            new_keypoints = np.zeros((num_boxes, num_kps + 2, 3), dtype=keypoints.dtype)
            new_keypoints[:, :num_kps, :] = keypoints
            new_keypoints[:, num_kps, :2] = (keypoints[:, names.index('left_shoulder'), :2] +
                                             keypoints[:, names.index('right_shoulder'), :2]) / 2.0
            new_keypoints[:, num_kps, 2] = np.minimum(keypoints[:, names.index('left_shoulder'), 2],
                                                      keypoints[:, names.index('right_shoulder'), 2])
            new_keypoints[:, num_kps + 1, :2] = (keypoints[:, names.index('left_hip'), :2] +
                                                 keypoints[:, names.index('right_hip'), :2]) / 2.0
            new_keypoints[:, num_kps + 1, 2] = np.minimum(keypoints[:, names.index('left_hip'), 2],
                                                          keypoints[:, names.index('right_hip'), 2])
            names.extend(['mid_shoulder', 'mid_hip'])
            lines.extend([[names.index('mid_shoulder'), names.index('nose')],
                          [names.index('mid_shoulder'), names.index('mid_hip')]])
            skeleton = np.array(lines)
            keypoints = new_keypoints

        if skeleton_color is None:
            skeleton_color = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
                              (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
                              (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
                              (255, 0, 255), (255, 0, 170), (255, 0, 85), (0, 0, 255)]

        self.draw_keypoints(keypoints, skeleton=skeleton, skeleton_color=skeleton_color, **kwargs)

    def draw_masks(self, masks, boxes=None, draw_edge=False, mask_color=(60, 20, 220),
                   edge_mask_color=(200, 200, 200), scale=0.5, binary_thresh=0.5):
        # mask: (num_boxes, mask_height, mask_width)
        if len(masks) == 0:
            return
        if boxes is not None:
            assert len(masks) == len(boxes)
            boxes = boxes.astype(np.int32)

        for i in range(len(masks)):
            mask_color_i = self._get_color(i, mask_color)
            mask_color_i = np.array(mask_color_i).reshape((1, 1, 3))
            if draw_edge:
                edge_mask_color_i = self._get_color(i, edge_mask_color)
                edge_mask_color_i = np.array(edge_mask_color_i).reshape((1, 1, 3))
            mask_i = masks[i]
            if mask_i.shape[:2] != self.image.shape[:2]:
                box_i = boxes[i]
                img_roi = self.image[box_i[1]:box_i[3], box_i[0]: box_i[2], :]
                mask_i = cv2.resize(mask_i, (box_i[2] - box_i[0], box_i[3] - box_i[1]),
                                    interpolation=cv2.INTER_LINEAR)
                mask_i_thr = (mask_i >= binary_thresh).astype(np.uint8)
                mask_i = mask_i_thr[:, :, np.newaxis]
                img_roi[:] = img_roi - scale * img_roi * mask_i + scale * mask_color_i * mask_i
                if draw_edge:
                    edge_mask_i = get_edge_mask(mask_i_thr)
                    edge_mask_i = edge_mask_i[:, :, np.newaxis]
                    img_roi[:] = img_roi - img_roi * edge_mask_i + edge_mask_color_i * edge_mask_i
            else:
                mask_i = (mask_i >= binary_thresh).astype(np.uint8)
                mask_i = mask_i[:, :, np.newaxis]
                self.image = self.image - scale * self.image * mask_i + scale * mask_color_i * mask_i

    def draw_polys_or_rles(self, polys_or_rles, boxes=None, mask_height=28, mask_width=28, **kwargs):
        if boxes is not None:
            from bit_cv.common.data import mask_utils
            masks = mask_utils.polys_or_rles_to_masks(polys_or_rles, boxes, mask_height, mask_width)
            self.draw_masks(masks, boxes, **kwargs)
        else:
            from pycocotools import mask as mask_utils
            img_mask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype='uint8')
            for ann in polys_or_rles:
                if isinstance(ann, list):
                    # Polygon format
                    if len(ann) > 0:
                        if not isinstance(ann[0], list):
                            ann = [ann]
                        polys = [np.array(ann[i]).astype(np.int32).reshape((-1, 2)) for i in range(len(ann))]
                        # method 1:
                        cv2.fillPoly(img_mask, polys, 1)
                        # method 2:
                        # cv2.drawContours(img_mask, polys, -1, 1, -1)
                else:
                    # RLE format
                    if isinstance(ann['counts'], list):
                        ann = mask_utils.frPyObjects(ann, ann['size'][0], ann['size'][1])
                    rle_mask = mask_utils.decode(ann)
                    img_mask += rle_mask
            self.draw_masks(img_mask[None, :], **kwargs)
