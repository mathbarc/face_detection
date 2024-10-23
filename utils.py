import cv2
import numpy


def show_all_keypoints(image, predicted_key_pts, gt_pts=[]):
    """Show image with predicted keypoints"""
    
    for point in predicted_key_pts:
        cv2.circle(image, point.astype(numpy.int32), 5, (0,255,0), -1)

    for point in gt_pts:
        cv2.circle(image, point.astype(numpy.int32), 5, (255,0,255), -1)
