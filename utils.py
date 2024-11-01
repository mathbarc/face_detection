import cv2
from matplotlib.pyplot import draw
import numpy

def separate_regions(predicted_key_pts):
    
    if isinstance(predicted_key_pts, numpy.ndarray):

        jaw = predicted_key_pts[0:17]
    
        right_eyebrow = predicted_key_pts[17:22]
        right_eye = predicted_key_pts[36:42]
    
        left_eyebrow = predicted_key_pts[22:27]
        left_eye = predicted_key_pts[42:48]

        nose = predicted_key_pts[27:36]

        upper_lip = predicted_key_pts[49:55]
        lower_lip = predicted_key_pts[55:60]

        upper_lip_lower = predicted_key_pts[60:64]
        lower_lip_upper = predicted_key_pts[64:68]
    else:
        jaw = predicted_key_pts[:,0:17]
    
        right_eyebrow = predicted_key_pts[:,17:22]
        right_eye = predicted_key_pts[:,36:42]
    
        left_eyebrow = predicted_key_pts[:,22:27]
        left_eye = predicted_key_pts[:,42:48]

        nose = predicted_key_pts[:,27:36]

        upper_lip = predicted_key_pts[:,49:55]
        lower_lip = predicted_key_pts[:,55:60]

        upper_lip_lower = predicted_key_pts[:,60:64]
        lower_lip_upper = predicted_key_pts[:,64:68]


    return [jaw, right_eyebrow, right_eye, left_eyebrow, left_eye, nose, upper_lip, upper_lip_lower, lower_lip_upper, lower_lip]


def draw_region(image, region, color):
    print(region.shape)

    for point in region:
        cv2.circle(image, point.astype(numpy.int32), 5, color, -1)


def show_all_keypoints(image:numpy.ndarray, predicted_key_pts, gt_pts=[]):
    """Show image with predicted keypoints"""
    

    jaw, right_eyebrow, right_eye, left_eyebrow, left_eye, nose, upper_lip, upper_lip_lower, lower_lip_upper, lower_lip = separate_regions(predicted_key_pts)

    draw_region(image, jaw, (0,150,150))
    draw_region(image, right_eyebrow, (0,230,25))
    draw_region(image, right_eye, (0,230,25))
    draw_region(image, left_eyebrow, (200,0,55))
    draw_region(image, left_eye, (200,0,55))
    draw_region(image, nose, (100,155,0))
    draw_region(image, upper_lip, (100,0,155))
    draw_region(image, upper_lip_lower, (100,0,155))
    draw_region(image, lower_lip_upper, (155,0,100))
    draw_region(image, lower_lip, (155,0,100))
