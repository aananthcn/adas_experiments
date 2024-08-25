import cv2


image = cv2.imread("Image/test_image.jpg")
print("image read complete")
cv2.imshow("result", image)
print("showing image")
cv2.waitKey(0)