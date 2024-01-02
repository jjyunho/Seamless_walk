import torch
import copy
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import cv2
from skimage import measure
from sklearn.linear_model import LinearRegression
import math

def upsample_idxs(flat_image, x_idxs, y_idxs):
    maxv, minv = 10, 1
    image = flat_image
    image = (image - image.min()) / (image.max() - image.min() + 1e-6)
    image = image * (maxv - minv) + minv
    image = image.astype(np.int32)
    total_idxs = []
    level_idxs = list(zip(image, x_idxs, y_idxs))
    for level, x, y in level_idxs:
        total_idxs += [(x, y) for _ in range(level)]
    total_idxs = np.array(total_idxs)
    x_idxs, y_idxs = total_idxs[:, 0], total_idxs[:, 1]
    return x_idxs, y_idxs

def vector_angle(vector1, vector2):
    x1, y1 = vector1
    x2, y2 = vector2
    inner_product = x1*x2 + y1*y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    inner_product /= (len1*len2)
    inner_product *= 0.98
    angle = math.acos(inner_product)
    return angle

def get_slope_bias(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection_line(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def get_foot_direction(line1, line2):
    try:
        foot1 = line1
        foot2 = line2

        # get all possible angles
        foot1_v11 = (foot1[0] - foot1[1], foot1[0], foot1[1])
        foot1_v12 = (foot1[1] - foot1[0], foot1[1], foot1[0])
        foot2_v21 = (foot2[0] - foot2[1], foot2[0], foot2[1])
        foot2_v22 = (foot2[1] - foot2[0], foot2[1], foot2[0])

        vectors = [
            [foot1_v11, foot2_v21],
            [foot1_v11, foot2_v22],
            [foot1_v12, foot2_v21],
            [foot1_v12, foot2_v22]
        ]

        angles = []
        for vector1, vector2 in vectors:
            v1, _, _ = vector1
            v2, _, _ = vector2

            angle = vector_angle(v1, v2)
            angles.append([angle, vector1, vector2])
        angles.sort(key=lambda x: x[0])

        # check orthodromic or opposite
        corrct_angles = []
        for angle, vector1, vector2 in angles:
            v1, end1, start1 = vector1
            v2, end2, start2 = vector2

            point = intersection_line(get_slope_bias(start1, end1), get_slope_bias(start2, end2))

            alpha1 = ((point - start1)[v1 != 0]/v1[v1 != 0]).sum()
            alpha2 = ((point - start2)[v2 != 0]/v2[v2 != 0]).sum()

            v1_reachable = alpha1 > 0
            v2_reachable = alpha2 > 0

            if v1_reachable == v2_reachable:
                corrct_angles.append([angle, vector1, vector2])
        # check front or back
        foot_vectors = []
        for _, vector1, vector2 in corrct_angles:
            v1, end1, start1 = vector1
            v2, end2, start2 = vector2

            start_dist = ((start1-start2)**2).sum()
            end_dist = ((end1 - end2) ** 2).sum()
            foot_vectors.append([vector1, vector2, end_dist - start_dist])

        foot_vectors.sort(key = lambda x:x[2], reverse=True)
        vector1, vector2, _ = foot_vectors[0]
        v1, end1, start1 = vector1
        v2, end2, start2 = vector2
        direction = (v1 + v2)/2
    except:
        direction = [0, 0]
    return direction

def get_linear_reg(x, y):
    model = LinearRegression()
    model.fit(x, y)
    pred = model.predict(x)
    error = ((pred-y)**2).mean()
    return model, error

def get_line(x, y):
    x_idxs = np.expand_dims(x, axis=-1)
    y_idxs = np.expand_dims(y, axis=-1)

    model1, error1 = get_linear_reg(x_idxs, y_idxs)
    model2, error2 = get_linear_reg(y_idxs, x_idxs)
    if error1 > error2:
        Y_minmax = np.array([[y_idxs.min()], [y_idxs.max()]])
        pred = model2.predict(Y_minmax)
        y_re = Y_minmax[:, 0].astype(np.int32).tolist()
        x_re = pred[:, 0].astype(np.int32).tolist()
        return np.array(list(zip(x_re, y_re))), error2
    else:
        X_minmax = np.array([[x_idxs.min()], [x_idxs.max()]])
        pred = model1.predict(X_minmax)
        x_re = X_minmax[:, 0].astype(np.int32).tolist()
        y_re = pred[:, 0].astype(np.int32).tolist()
        return np.array(list(zip(x_re, y_re))), error1

def get_chunks(image_):
    image = copy.deepcopy(image_)

    image[image != 0] = 1
    label_image = measure.label(image, connectivity=2)

    # get chunks
    chunks = []
    for i in range(1, label_image.max() + 1):
        idxs = label_image == i
        chunks.append((i, idxs, image_[idxs].sum(), image_[idxs].mean()))
    chunks.sort(key=lambda x: x[2], reverse=True)
    return chunks