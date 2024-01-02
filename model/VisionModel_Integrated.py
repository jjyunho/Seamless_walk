import torch
import copy
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import cv2
from skimage import measure
from sklearn.linear_model import LinearRegression
import math
from utils import *


class FootDetector:
    def __init__(self, visualize, smooth_speed=True):
        self.visualize = visualize
        self.speed_buffer = []
        self.smooth_speed = smooth_speed

    def __call__(self, images, **kwargs):
        '''
        INPUT
        images : (time, 64, 64)
        OUTPUT
        available: True if result angle and speed are available
        angle: angle of human
        speed: speed of human
        '''
        hmd_yaw = kwargs['hmd_yaw']

        #images = images ** 3
        foot_signal_lower_bound = images.mean() + 2.0 * images.std()
        foot_x = np.where(images >= foot_signal_lower_bound)[1].astype(int)
        foot_y = np.where(images >= foot_signal_lower_bound)[2].astype(int)

        if foot_x.size > 3:
            foot_cordination = list(zip(foot_x, foot_y))
            foot_infos = KMeans(n_clusters=2, random_state=0).fit(foot_cordination)
        else:
            foot_infos = None

        available = True
        if foot_infos != None:
            centers_of_foot = foot_infos.cluster_centers_

            ''' linear reg '''
            images_temp = copy.deepcopy(images)
            images_temp[images_temp < foot_signal_lower_bound] = 0
            images_temp[images_temp != 0] = 1
            images_temp = images_temp.mean(axis=0)

            chunks = get_chunks(images_temp)

            linear_reg_avail = True
            if len(chunks) > 1:
                mean_images = images.mean(axis=0)
                two_chunks = chunks[:2]

                lines = []
                errors = []
                for _, idxs, _, _ in two_chunks:
                    x_idxs = np.where(idxs)[0].astype(int)
                    y_idxs = np.where(idxs)[1].astype(int)
                    x_idxs, y_idxs = upsample_idxs(mean_images[idxs], x_idxs, y_idxs)
                    line, error = get_line(x_idxs, y_idxs)
                    lines.append(line)
                    errors.append(error)
                direction_vector = get_foot_direction(*lines)
                angle_candidate1 = math.degrees(math.atan2(direction_vector[1], direction_vector[0]))
                angle_candidate1 += 90 # it depend monitor position
            else:
                linear_reg_avail = False

            if linear_reg_avail:
                left_len = ((lines[0][0] - lines[0][1]) ** 2).sum() ** 0.5
                right_len = ((lines[1][0] - lines[1][1]) ** 2).sum() ** 0.5
                threshold = 8 # it depend sensor resolution
                linear_reg_avail = left_len > threshold and right_len > threshold

            ''' if linear_reg is not available, use simple clustering '''
            if not linear_reg_avail:
                slope = ((64 - centers_of_foot[0][0]) - (64 - centers_of_foot[1][0])) / \
                        (centers_of_foot[0][1] - centers_of_foot[1][1])
                orthogonal_slope = -1 / slope
                angle_candidate1 = np.degrees(np.arctan(orthogonal_slope))

            ''' use hmd yaw to match gaze direction and walking direction'''
            if angle_candidate1 > 0:
                angle_candidate1 %= 180
            else:
                angle_candidate1 = -(abs(angle_candidate1) % 180)

            angle_candidate1 = angle_candidate1 - 90
            angle_candidate1 = angle_candidate1 * (-1)
            angle_candidate2 = angle_candidate1 - 180

            if hmd_yaw >= 0:
                if abs(angle_candidate1 - hmd_yaw) <= 90:
                    angle = angle_candidate1
                else:
                    angle = angle_candidate2
            else:
                if abs(angle_candidate2 - hmd_yaw) <= 90:
                    angle = angle_candidate2
                else:
                    angle = angle_candidate1

            ''' produce speed '''
            cluster_idxs = []
            two_foot_diff = []
            for i in range(len(images)):
                image_temp = copy.deepcopy(images[i])
                image_temp[image_temp < image_temp.mean() + 2.0 * image_temp.std()] = 0
                image_temp[image_temp != 0] += 1
                chunks = get_chunks(image_temp)
                if len(chunks) > 0:
                    # get max pressure idx
                    chunks.sort(key=lambda x: x[2], reverse=True)
                    idxs = chunks[0][1]
                    x_idxs = np.where(idxs)[0].mean().astype(int)
                    y_idxs = np.where(idxs)[1].mean().astype(int)
                    max_pressure_index = np.array((x_idxs, y_idxs))

                    # get stand
                    if len(chunks) >= 2:
                        chunks.sort(key=lambda x: x[3], reverse=True)
                        sum1 = chunks[0][3]
                        sum2 = chunks[1][3]
                        diff = abs(sum1 - sum2)
                        two_foot_diff.append(diff)
                else:
                    max_pressure_index = np.unravel_index(images[i].argmax(), images[i].shape)

                foot_cluster_idx = foot_infos.predict([list(max_pressure_index)])
                cluster_idxs.append(foot_cluster_idx)


            ''' Check stand '''
            two_foot_diff.append(0)
            diff_max = max(two_foot_diff)

            if diff_max < 0.13:
                stand = True
            else:
                stand = False

            '''if not stand, get speed from foot interval'''
            if not stand:
                temp = 0
                interval = []
                for i in range(1, len(cluster_idxs)):
                    if cluster_idxs[i] != cluster_idxs[i-1]:
                        interval.append(temp)
                        temp = 0
                    else:
                        temp += 1

                if len(interval) > 1:
                    del interval[0]
                    interval = max(interval)
                    speed = -interval + 10
                    speed /= 5
                else:
                    speed = 0
                speed = min(max(speed, 0), 1)
            else:
                speed = 0

            '''visualize'''
            if self.visualize:

                draw_lines = [
                    [centers_of_foot.astype(np.int32), (0, 0, 255)]
                ]

                # draw direction
                center_of_body = centers_of_foot.mean(axis=0)
                visual_angle = angle-90
                direction_vector = np.array([
                    math.sin(math.radians(visual_angle)),
                    math.cos(math.radians(visual_angle))
                ])
                end = center_of_body + direction_vector * 10
                direction_line = [center_of_body.astype(np.int32), end.astype(np.int32)]
                draw_lines.append([direction_line, (0, 255, 0)])

                if linear_reg_avail:
                    # draw foot
                    draw_lines += [
                        [lines[0], (255, 0, 0)],
                        [lines[1], (255, 0, 0)]
                    ]

                    # draw direction
                    start = center_of_body = centers_of_foot.mean(axis=0).astype(np.int32)
                    end = center_of_body + np.array(direction_vector)*2
                    direction_line = [start.astype(np.int32), end.astype(np.int32)]
                    draw_lines.append([direction_line, (0, 255, 0)])

                viz = images[-1]
                viz = (viz - viz.mean())/viz.std()
                viz = (viz+15)/30

                viz = (viz*255).astype(np.uint8)
                viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)

                if not stand:
                    for line, color in draw_lines:
                        cv2.line(viz, line[0][::-1], line[1][::-1], color, 1)
                    cv2.circle(viz, np.array(max_pressure_index)[::-1].astype(np.int32), 3, (0, 0, 255), 1)

                self.visualized_image = viz

            ''' speed smoothing '''
            if self.smooth_speed:
                self.speed_buffer.append(speed)
                smooth_len = 5
                if len(self.speed_buffer) > smooth_len:
                    buffer = self.speed_buffer[-smooth_len:]
                    speed = sum(buffer)/len(buffer)

        else:
            available, angle, speed = False, None, None
        return available, angle, speed


