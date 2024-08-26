import cv2
import numpy as np
# import matplotlib.pyplot as plt


def canny(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5,5), 0)
    canny_image = cv2.Canny(blur_image, 50, 150)
    return canny_image


def region_of_interest(image):
    ht = image.shape[0]
    triangle = np.array([
            [(200, ht), (1100, ht), (550, 250)]
        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters

    # let us start the y1 from the bottom-most part of the view port
    y1 = image.shape[0]

    # our view port of interest starts from the bottom to 3/5th of y-axis
    y2 = int(y1 * (3 / 5))

    # compute Xs using the formula you learned in your high school
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)

    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # fit the line to a polynomial (y = mx + c) and print m and c
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        p_m = parameters[0] # slope
        p_c = parameters[1] # y-intercept
        if p_m < 0:
            left_fit.append((p_m, p_c))
        else:
            right_fit.append((p_m, p_c))

    # average the fits along the vertical axis (axis = 0)
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)

    # make line?
    left_line = make_coordinates(image, left_fit_avg)
    right_line = make_coordinates(image, right_fit_avg)

    return np.array([left_line, right_line])


image = cv2.imread("Image/test_image.jpg")

# copy so that the original data is not modified
lane_image = np.copy(image)

canny_image = canny(lane_image)
cropped_img = region_of_interest(canny_image)

# find out lines in the region using Hough transform
lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

# let us remove double lines detected from the lanes markers on road
avgd_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, lines)
line_image = display_lines(lane_image, avgd_lines)
comp_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

# cv2.imshow("result", canny_image)
# cv2.imshow("result", cropped_img)
# cv2.imshow("result", line_image)
cv2.imshow("result", comp_image)
cv2.waitKey(0)

# plt.imshow(canny_image)
# plt.show()