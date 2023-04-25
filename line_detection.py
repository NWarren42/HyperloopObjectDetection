"""
Notes: this only works if there isn't significant curvature in the track -
this is expected as with a high speed hyperloop tube, in order to have acceptable g-forces on the passenger,
the radius of curvature must be quite high
as well, this will be less accurate with camera angles that are higher.
This is also within parameters as a higher viewing angle reduces the depth of the field of view
This works best if the camera is centred in the lane due to the averaging function.
This should be where the camera is placed anyway in order to meet the requirements for side view
As well, the image should be high-resolution for accuracy
Lastly, this will be less accurate with track intersections -
as there are only a few models for Hyperloop track intersections,
and only one of which involves any intersections visible to the camera, this should be okay.
As far as image quality, this becomes inaccurate if the image is less than 500x500 pixels
but since the camera will be high quality it should be fine.
"""
import cv2
import numpy as np
import time
# frame_work = cv2.imread("TEST_IMAGE_2.jpg")
# frame = frame_work
# apply grayscale to simplify
def grey(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Mask image to area right in front of camera
# define height and width of image
def mask(image):
    height, width = image.shape
    # define area of interest
    area = np.array([
        (0, height), (0, round(height*(3/4))),
        (round((width*(1/3))), round(height*(1/4))), (round(width*(2/3)), round(height*(1/4))),
        (width, round(height*(3/4))), (width, height)
    ], np.int32)
    masked_image = np.zeros_like(image)
    # fill with white, apply to image
    masked_image = cv2.fillPoly(masked_image, [area], 255)
    return cv2.bitwise_and(image, masked_image)
# simplify further by applying a median blur to reduce noise.
# Out of the types of noise reduction in OpenCV, this is the fastest method that also preserves edges
def median(image):
    return cv2.medianBlur(image, 5)
# Canny edge detection
def canny(image):
    edges = cv2.Canny(image, 70, 130)
    return edges
# threshold to pick out lines from background
def threshold(image):
    placeholder, thresh = cv2.threshold(image, 100, 145, cv2.THRESH_BINARY)
    return thresh
# apply hough line transformation
def hough(image):
    return cv2.HoughLinesP(image, 2, np.pi/180, threshold=100, minLineLength=50, maxLineGap=20)
# average out left edge of track and right edge of track
def smooth_lines(image, lines):
    left = []
    right = []
    right_avg = 0
    left_avg = 0
    # take out lines with shallow slope (i.e. close to horizontal, to avoid horizon lines)
    try:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            y_int = parameters[1]
            if slope < -0.5:
                left.append((slope, y_int))
            elif slope > 0.5:
                right.append((slope, y_int))
        if right:
            right_avg = np.average(right, axis=0)
        if left:
            left_avg = np.average(left, axis=0)
        if right_avg is not np.nan:
            right_line = make_points(image, right_avg)
        if left_avg is not np.nan:
            left_line = make_points(image, left_avg)
        return np.array([left_line, right_line])
    except TypeError:
        return np.array([make_points(image, 0), make_points(image, 0)])
def make_points(image, average):
    try:
        slope, y_int = average
    except TypeError:
        slope, y_int = 0.001, 0
    # slope, y_int = average
    y1 = image.shape[0]
    # so y1 is always the bottom point
    y2 = int(y1 * (1/4))
    # slope formula
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])
# attempt to try and deal with intersections
def line_intersection(image, lines):
    # need to convert to floats for accurate line intersection, convert back afterward
    lines = lines.astype('float64')
    if lines is not None:
        lines[0] = line_slope(lines[0])
        lines[1] = line_slope(lines[1])
        lx1, ly1, lm, lb = lines[0]
        rx1, ry1, rm, rb = lines[1]
        # calculate intersections
        if lm != rm:
            x_intersection = (rb-lb)//(lm-rm)
            y_intersection = rm*x_intersection + rb
            height, width = image.shape
            if 0 <= y_intersection <= height and 0 <= x_intersection <= width:
                lines[0] = lx1, ly1, x_intersection, y_intersection
                lines[1] = rx1, ry1, x_intersection, y_intersection
    lines = lines.astype('int32')
    return lines
def line_slope(line):
    x1, y1, x2, y2 = line
    parameters = np.polyfit((x1, x2), (y1, y2), 1)
    m = parameters[0]
    b = parameters[1]
    line = x1, y1, m, b
    return line
# draw lines and lane on the frame
def track_lines(image, lines):
    area = []
    image_additions = image.copy()
    image_final = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(image_final, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(image_additions, (x1, y1), (x2, y2), (0, 255, 0), 2)
            area.append((x1, y1))
            area.append((x2, y2))
    area = sorted(area)
    area_fill = np.array(area, np.int32)
    cv2.fillPoly(image_additions, [area_fill], (144, 238, 144))
    return cv2.addWeighted(image_additions, 0.5, image_final, 0.5, 0)

def process_frame (frame):
    working_frame = threshold(mask(canny(median(grey(frame)))))
    tracks = hough(working_frame)
    tracks = smooth_lines(working_frame, tracks)
    tracks = line_intersection(working_frame, tracks)
    processed_frame = track_lines(frame, tracks)
    return processed_frame
