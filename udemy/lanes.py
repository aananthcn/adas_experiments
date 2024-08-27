import cv2
import numpy as np


def canny(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
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
        for line in lines:
            if line is not None:  # Ensure line is valid
                x1, y1, x2, y2 = line
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def make_coordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        return None  # Invalid line parameters

    # Start y1 from the bottom-most part of the view port
    y1 = image.shape[0]

    # View port of interest starts from the bottom to 3/5th of y-axis
    y2 = int(y1 * (3 / 5))

    # Compute Xs using the formula y = mx + c
    try:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    except ZeroDivisionError:
        return None  # Slope was 0, leading to division error

    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    if lines is None:
        return None  # No lines detected

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # Fit the line to a polynomial (y = mx + c)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        p_m = parameters[0]  # Slope
        p_c = parameters[1]  # Intercept

        if p_m < 0:
            left_fit.append((p_m, p_c))
        else:
            right_fit.append((p_m, p_c))

    # Calculate the averages if there are valid fits
    left_line = None
    right_line = None

    if left_fit:
        left_fit_avg = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_avg)

    if right_fit:
        right_fit_avg = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_avg)

    # Only return lines that are not None
    return [line for line in [left_line, right_line] if line is not None]


# image = cv2.imread("Image/test_image.jpg")

# # copy so that the original data is not modified
# lane_image = np.copy(image)

repeat_cnt = 5
while repeat_cnt > 0:
    cap = cv2.VideoCapture("C:\\Users\\emb-aanahcn\\Downloads\\test2.mp4")

    while cap.isOpened():
        _, vframe = cap.read()
        if vframe is None:
            break  # End of the video

        # canny_image = canny(lane_image)
        canny_image = canny(vframe)
        cropped_img = region_of_interest(canny_image)

        # Find lines using Hough transform
        lines = cv2.HoughLinesP(cropped_img, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)

        # Remove double lines detected from the lanes markers on the road
        avgd_lines = average_slope_intercept(vframe, lines)
        if avgd_lines:
            line_image = display_lines(vframe, avgd_lines)
            comp_image = cv2.addWeighted(vframe, 0.8, line_image, 1, 1)
            cv2.imshow("result", comp_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            repeat_cnt = 0
            break  # Exit on 'q' key press

    repeat_cnt -= 1
    cap.release()

cv2.destroyAllWindows()
