########################################################################

'''
Real-time detection using Intel Realsense d435i.

out = np.array([[-0.5211449 , -0.5608407 ,  0.09475598],  # 小指顶点
                [-0.3877076 , -0.26955765,  0.10899305],  # 小指手掌
                [-0.13829438, -0.584938  ,  0.00213338],  # 无名顶点
                [-0.17500372, -0.41086006,  0.12675118],  # 无名手掌
                [-0.08450034, -0.7325556 ,  0.04596908],  # 中指顶点
                [-0.02003123, -0.4200949 ,  0.14079444],  # 中指手掌
                [ 0.25274026, -0.67753404,  0.01210476],  # 食指顶点
                [ 0.25423834, -0.40538555,  0.08445187],  # 食指手掌
                [ 0.7556279 , -0.22524479, -0.09432331],  # 拇指顶点
                [ 0.6267641 , -0.08225113, -0.04437109],  # 拇指手掌
                [ 0.448883  ,  0.33178666,  0.09997568],  # 拇指肌肉
                [-0.31449842,  0.60406464,  0.2599266 ],  # 把脉外侧
                [ 0.1996114 ,  0.7707358 ,  0.2559907 ],  # 把脉点
                [ 0.04733039,  0.26531252,  0.24434374]])  # 手掌中心
'''

########################################################################

import os
import cv2
import os.path as osp
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation

from realsense import DepthCamera
from mediapipe import solutions as mp

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from model.resnet_deconv import get_deconv_net
from model.hourglass import PoseNet
from util.feature_tool import FeatureModule
from config import opt

########################################################################

pt = (0, 0)

def show_dist(event, x, y, args, params):
    global pt
    pt = (x, y)

# Create Mouse event
cv2.namedWindow('Color')
cv2.setMouseCallback('Color', show_dist)


def cv2lines(img, pts, col=(0, 255, 0), thickness=4):
    for i, pt in enumerate(pts):
        if i == 0:
            continue
        cv2.line(img, pt, pts[i - 1], col, thickness=thickness)
    return img


def out2img(ctr_uvds, img_sizes, ctr_bbox):
    # # Flip the "width" coordinate in order to match image axis.
    # ctr_uvds[:, :, 1] *= -1
    # # Move the (0, 0) coordinate to Upper Left corner.
    # ctr_uvds[:, :, :2] += 0.5

    for i, size in enumerate(img_sizes):
        # Scale the relative coords back to abs coords to the image.
        ctr_uvds[i, :, :2] *= (size / 2)

    for i, xy in enumerate(ctr_bbox):
        # Shift the key points to the axis of a full image.
        ctr_uvds[i, :, 0] += xy[0]
        ctr_uvds[i, :, 1] += xy[1]

    return ctr_uvds


def crop3d(depth_frame, xyxyd, cube, thres_d=True, bg=0):
    H, W = depth_frame.shape
    dstart, dend = (xyxyd[-1] - cube[-1] / 2, xyxyd[-1] + cube[-1] / 2)
    bbox = [max(xyxyd[1], 0), min(xyxyd[3], H), max(xyxyd[0], 0), min(xyxyd[2], W)]
    img = depth_frame[bbox[0]:bbox[1], bbox[2]:bbox[3]]

    # Pad extra pixels for those bbox out of image window in order to keep aspect ratio.
    img = np.pad(img, ((abs(xyxyd[1] - bbox[0]), abs(xyxyd[3] - bbox[1])),
                       (abs(xyxyd[0] - bbox[2]), abs(xyxyd[2] - bbox[3]))),
                 mode='constant', constant_values=bg)

    if thres_d:
        # Constrain the depth distance into a fixed-size cube.
        img[np.logical_and(img < dstart, img != 0)] = dstart
        img[np.logical_and(img > dend, img != 0)] = bg

    return img


def normalize(depth_frame, ctr_d, cube, bg=0):
    depth_min = ctr_d - (cube[2] / 2.)
    depth_max = ctr_d + (cube[2] / 2.)

    # Invalid points are assigned as bg.
    depth_frame[depth_frame == bg] = depth_max
    depth_frame[depth_frame == np.max(depth_frame)] = depth_max

    depth_frame = np.clip(depth_frame, depth_min, depth_max)
    depth_frame -= ctr_d
    depth_frame /= (cube[2] / 2.)
    return depth_frame


class Detector(object):

    def __init__(self, config, height=480, width=640):
        self.H, self.W = height, width
        self.device = torch.device('cuda:{}'.format(config.gpu_id) if torch.cuda.is_available() else 'cpu')
        # torch.cuda.set_device(config.gpu_id)
        cudnn.benchmark = True

        self.config = config
        self.cube = config.cube
        self.dsize = (config.img_size, config.img_size)
        self.data_dir = osp.join(self.config.data_dir, self.config.dataset)

        # output dirs for model, log and result figure saving
        self.model_save = osp.join(self.config.output_dir, self.config.dataset, 'checkpoint')
        self.result_dir = osp.join(self.config.output_dir, self.config.dataset, 'results' )
        if not osp.exists(self.model_save):
            os.makedirs(self.model_save)
        if not osp.exists(self.result_dir):
            os.makedirs(self.result_dir)

        if 'resnet' in self.config.net:
            net_layer = int(self.config.net.split('_')[1])
            self.net = get_deconv_net(net_layer, self.config.jt_num, self.config.downsample)
        elif 'hourglass' in self.config.net:
            self.stacks = int(self.config.net.split('_')[1])
            self.net = PoseNet(self.config.net, self.config.jt_num)

        self.net = self.net.to(self.device)  # .cuda()

        if self.config.load_model :
            print('loading model from %s' % self.config.load_model)
            pth = torch.load(self.config.load_model, map_location=self.device)
            self.net.load_state_dict(pth['model'])
            print(pth['best_records'])

        self.net = self.net.to(self.device)  # .cuda()
        self.net.eval()
        self.FM = FeatureModule()

        self.cam = DepthCamera(height=height, width=width)
        self.hands = mp.hands.Hands(min_detection_confidence=0.6,
                                    min_tracking_confidence=0.4)

    def detect(self):
        ret, depth_frame, color_frame = self.cam.get_frame()
        depth_map = cv2.applyColorMap(cv2.convertScaleAbs(
            depth_frame, alpha=0.03), cv2.COLORMAP_JET)

        jt_uvd_pred = []
        results = self.hands.process(cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))
        if results.hand_rects:
            cropped_frames, cropped_sizes, centers = [], [], []
            for num, box in enumerate(results.hand_rects):
                if num > 0:
                    break

                x1 = int((box.x_center - box.width / 2) * self.W)
                y1 = int((box.y_center - box.height / 2) * self.H)
                x2 = int((box.x_center + box.width / 2) * self.W)
                y2 = int((box.y_center + box.height / 2) * self.H)
                HEIGHT, WIDTH = y2 - y1, x2 - x1
                cropped_sizes.append(HEIGHT if HEIGHT > WIDTH else WIDTH)
                centers.append([box.x_center * self.W, box.y_center * self.H])
                # ==========================================================
                # This block is only for bbox visualization to both color and depth frames.
                cv2.rectangle(depth_map, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.rectangle(color_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                # ==========================================================

                d = depth_frame[int(box.y_center * self.H), int(box.x_center * self.W)]
                cropped_frame = crop3d(depth_frame, (x1, y1, x2, y2, d),
                                       self.cube, thres_d=True, bg=0)
                cropped_frame = cv2.resize(cropped_frame, self.dsize,
                                           interpolation=cv2.INTER_NEAREST)
                cropped_frame = normalize(cropped_frame, d, self.cube)
                # print('cropped-frame: ', cropped_frame.shape)

                # To make extra dim for batch size & channel.
                cropped_frame = np.expand_dims(cropped_frame, axis=0).astype(np.float32)
                cropped_frames.append(cropped_frame)

            cropped_frames = np.stack(cropped_frames, axis=0)
            # print('cropped-frame: ', cropped_frames.shape)
            cropped_frames = torch.tensor(cropped_frames, dtype=torch.float32).to(self.device)

            if 'hourglass' in self.config.net:
                for stage_idx in range(self.stacks):
                    offset_pred = self.net(cropped_frames)[stage_idx]
                    jt_uvd_pred = self.FM.offset2joint_softmax(offset_pred, cropped_frames, self.config.kernel_size)
            else:
                offset_pred = self.net(cropped_frames)
                jt_uvd_pred = self.FM.offset2joint_softmax(offset_pred, cropped_frames, self.config.kernel_size)

            jt_uvd_pred = jt_uvd_pred.detach().cpu().numpy()
            jt_uvd_pred = out2img(jt_uvd_pred, cropped_sizes, centers)

        return depth_frame, color_frame, depth_map, jt_uvd_pred

    def stop(self):
        self.hands.close()
        self.cam.release()


if __name__=='__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    detector = Detector(opt)

    plt.ion()
    fig = plt.figure(figsize=(12, 9))
    ax = plt.axes(projection='3d')
    try:
        while True:
            depth_frame, color_frame, depth_map, uvd = detector.detect()

            # Visualize the distance for a specific point
            cv2.circle(color_frame, pt, 4, (0, 0, 255), 2)
            dist = depth_frame[pt[1], pt[0]]

            cv2.putText(color_frame, '{} mm'.format(dist),
                        (pt[0] + 5, pt[1] - 10),
                        cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 255, 0), 2)

            plt.cla()
            if len(uvd) > 0:
                print(uvd)

                # ----------------------------------------------------------
                ax.plot3D(uvd[0][:, 0][[0, 1, -1]], uvd[0][:, 1][[0, 1, -1]], uvd[0][:, 2][[0, 1, -1]], 'red')
                ax.plot3D(uvd[0][:, 0][[2, 3, -1]], uvd[0][:, 1][[2, 3, -1]], uvd[0][:, 2][[2, 3, -1]], 'red')
                ax.plot3D(uvd[0][:, 0][[4, 5, -1]], uvd[0][:, 1][[4, 5, -1]], uvd[0][:, 2][[4, 5, -1]], 'red')
                ax.plot3D(uvd[0][:, 0][[6, 7, -1]], uvd[0][:, 1][[6, 7, -1]], uvd[0][:, 2][[6, 7, -1]], 'red')
                ax.plot3D(uvd[0][:, 0][[8, 9, 10, -1]], uvd[0][:, 1][[8, 9, 10, -1]], uvd[0][:, 2][[8, 9, 10, -1]], 'blue')
                ax.plot3D(uvd[0][:, 0][[-3, -1, -2]], uvd[0][:, 1][[-3, -1, -2]], uvd[0][:, 2][[-3, -1, -2]], 'green')
                ax.scatter3D(uvd[0][:, 0], uvd[0][:, 1], uvd[0][:, 2], s=40)
                plt.pause(0.05)
                # ----------------------------------------------------------
                for kpt in uvd[0]:
                    cv2.circle(color_frame, kpt[:-1].astype(np.int), 4, (0, 0, 255), 2)

                color_frame = cv2lines(color_frame, uvd[0][[0, 1, -1], :2].astype(int),
                                       col=(0, 255, 0), thickness=2)
                color_frame = cv2lines(color_frame, uvd[0][[2, 3, -1], :2].astype(int),
                                       col=(0, 255, 0), thickness=2)
                color_frame = cv2lines(color_frame, uvd[0][[4, 5, -1], :2].astype(int),
                                       col=(0, 255, 0), thickness=2)
                color_frame = cv2lines(color_frame, uvd[0][[6, 7, -1], :2].astype(int),
                                       col=(0, 255, 0), thickness=2)
                color_frame = cv2lines(color_frame, uvd[0][[8, 9, 10, -1], :2].astype(int),
                                       col=(255, 0, 255), thickness=2)
                color_frame = cv2lines(color_frame, uvd[0][[-3, -1, -2], :2].astype(int),
                                       col=(255, 0, 0), thickness=2)
                # ----------------------------------------------------------
                for kpt in uvd[0]:
                    cv2.circle(depth_map, kpt[:-1].astype(np.int), 4, (0, 0, 255), 2)

                depth_map = cv2lines(depth_map, uvd[0][[0, 1, -1], :2].astype(int),
                                     col=(0, 255, 0), thickness=2)
                depth_map = cv2lines(depth_map, uvd[0][[2, 3, -1], :2].astype(int),
                                     col=(0, 255, 0), thickness=2)
                depth_map = cv2lines(depth_map, uvd[0][[4, 5, -1], :2].astype(int),
                                     col=(0, 255, 0), thickness=2)
                depth_map = cv2lines(depth_map, uvd[0][[6, 7, -1], :2].astype(int),
                                     col=(0, 255, 0), thickness=2)
                depth_map = cv2lines(depth_map, uvd[0][[8, 9, 10, -1], :2].astype(int),
                                     col=(255, 0, 255), thickness=2)
                depth_map = cv2lines(depth_map, uvd[0][[-3, -1, -2], :2].astype(int),
                                     col=(255, 0, 0), thickness=2)
                # ----------------------------------------------------------

            cv2.imshow('Depth', depth_map)
            cv2.imshow('Color', color_frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

        plt.ioff()
        plt.show()

    finally:
        detector.stop()
