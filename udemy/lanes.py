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

image = cv2.imread("Image/test_image.jpg")

# copy so that the original data is not modified
lane_image = np.copy(image)

canny_image = canny(lane_image)
cropped_img = region_of_interest(canny_image)

# cv2.imshow("result", canny_image)
cv2.imshow("result", cropped_img)
cv2.waitKey(0)

# plt.imshow(canny_image)
# plt.show()