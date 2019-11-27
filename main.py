import cv2
import numpy as np
import videostabilization as vid

VIDEO_NAME = 'input1'
VIDEO_EXTENSION = '.mp4'
CROP_PERCENTAGE = 0

video = cv2.VideoCapture(VIDEO_NAME + VIDEO_EXTENSION)

transforms = vid.getTransforms(video)
trajectories = np.cumsum(transforms, axis=1)

smoothTrajectories = vid.getSmoothTrajectories(trajectories)
vid.graphs(trajectories, smoothTrajectories)

differences = smoothTrajectories - trajectories
smoothTransforms = transforms + differences

vid.writeVideo(smoothTransforms, video, VIDEO_NAME + '_smoothed' + VIDEO_EXTENSION, CROP_PERCENTAGE)

video.release()
