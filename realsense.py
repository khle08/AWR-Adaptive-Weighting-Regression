#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pyrealsense2 as rs

class DepthCamera(object):
    def __init__(self, height=480, width=640, align=True):
        self.align = align
        self.pipeline = rs.pipeline()
        cfg = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = cfg.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        self.device_product_line = str(device.get_info(rs.camera_info.product_line))

        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

        profile = self.pipeline.start(cfg)

        if self.align:
            # Getting the depth sensor's depth scale.
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            # The background of objects more than clipping_distance will be removed.
            clipping_distance_in_meters = 1  # 1 meter
            self.clipping_distance = clipping_distance_in_meters / depth_scale
            # Create an align object to align depth frame to others frames.
            self.align_to = rs.align(rs.stream.color)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()

        if self.align:
            frames = self.align_to.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return False, None, None

        depth_image = np.array(depth_frame.get_data())
        color_image = np.array(color_frame.get_data())
        return True, depth_image, color_image

    def release(self):
        self.pipeline.stop()
