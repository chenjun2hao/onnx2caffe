import caffe
from PIL import Image
import numpy as np
import cv2
caffe.set_mode_cpu()


def load_norm(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    img = np.float32(img)

    mean = np.array(mean)
    std = np.array(std)
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace

    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img.transpose((2, 0, 1))[None]


def topk_by_sort(input, k, axis=None, ascending=True):
    if not ascending:
        input *= -1
    ind = np.argsort(input, axis=axis)[::-1]
    ind = np.take(ind, np.arange(k), axis=axis)
    if not ascending:
        input *= -1
    val = np.take_along_axis(input, ind, axis=axis) 
    return val, ind


def get_local_maximum(heat, kernel=3):
    """Extract local maximum pixel with given kernel.

    Args:
        heat (Tensor): Target heatmap.
        kernel (int): Kernel size of max pooling. Default: 3.

    Returns:
        heat (Tensor): A heatmap where local maximum pixels maintain its
            own value and other positions are 0.
    """
    import torch
    import torch.nn.functional as F
    pad = (kernel - 1) // 2
    heat = torch.from_numpy(heat)
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    heat = heat * keep
    return heat.numpy()


def get_topk_from_heatmap(scores, k=20):
    batch, _, height, width = scores.shape
    topk_scores, topk_inds = topk_by_sort(scores.reshape(batch, -1), k)
    topk_clses = topk_inds // (height * width)
    topk_inds = topk_inds % (height * width)
    topk_ys = topk_inds // width
    topk_xs = (topk_inds % width)
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def transpose_and_gather_feat(feat, ind):
    b, c, h, w = feat.shape
    feat = feat.reshape(c, -1)
    out = np.zeros((1, len(ind), 2))
    for i, idx in enumerate(ind):
        out[0, i, :] = feat[:, idx]
    return out


def postprocess(inp_h, inp_w, center_heatmap_pred, wh_pred, offset_pred, topk=100):
    center_heatmap_pred = get_local_maximum(center_heatmap_pred)

    height, width = center_heatmap_pred.shape[2:]
    *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
        center_heatmap_pred, k=topk)
    batch_scores, batch_index, batch_topk_labels = batch_dets

    wh = transpose_and_gather_feat(wh_pred, batch_index)
    offset = transpose_and_gather_feat(offset_pred, batch_index)
    topk_xs = topk_xs + offset[..., 0]
    topk_ys = topk_ys + offset[..., 1]
    tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
    tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
    br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
    br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

    bboxes = np.stack([tl_x, tl_y, br_x, br_y], axis=2)[0]
    bboxes = np.concatenate([bboxes, batch_scores[..., None]], axis=-1)
    return bboxes, batch_topk_labels
    

def test_centernet():
    img_path = 'examples/person.png'
    prototxt = 'output/centerface_tiny.prototxt'
    caffemodel = 'output/centerface_tiny.caffemodel'
    net = caffe.Net(prototxt, caffemodel, caffe.TRAIN)

    ## load input
    img = cv2.imread(img_path)
    h=416
    w=680
    if h and w:
        img = cv2.resize(img, (w, h))
        img = img[:h, :416, :]
    im = load_norm(img, mean=[0, 0, 0], std=[255.0, 255.0, 255.0])
    net.blobs['image'].data[...] = im
    
    ## infer
    out = net.forward()
    
    ih, iw = im.shape[-2:]
    bboxes, classes = postprocess(ih, iw, out['deploy0'], out['deploy1'], out['deploy2'])       #

    ## draw vis
    from imagevisual import ImageVisualizer
    vis = ImageVisualizer(img, save_image_dir='output')
    vis.set_colormap(method=2, num_colors=80)

    keep = np.where(bboxes[..., 4] > 0.3)[0]
    boxes = bboxes[keep,:4]
    classes = classes[keep]
    vis.draw_boxes(boxes, classes)
    vis.save('after_nms.jpg')


if __name__ == '__main__':
    test_centernet()