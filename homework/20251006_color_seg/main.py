import cv2
import numpy as np
import os

img_path = os.path.expanduser("./figura.jpg")
img = cv2.imread(img_path)

lower_red = np.array([170, 100, 100])
upper_red = np.array([180, 255, 255])
lower_green = np.array([40, 100, 100])
upper_green = np.array([80, 255, 255])
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([140, 255, 255])
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])


# def nearestNeighbor(image):


def readPixels(image):
    [height, width] = image.shape[:2]
    for px_h in range(height):
        for px_w in range(width):
            px = image[px_h, px_w]
            print(px)
            pass
        pass


# def countShapes(mask):
# shape_count = len(contours)
# return shape_count


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

red = cv2.bitwise_and(img, img, mask=mask_red)
green = cv2.bitwise_and(img, img, mask=mask_green)
blue = cv2.bitwise_and(img, img, mask=mask_blue)
yellow = cv2.bitwise_and(img, img, mask=mask_yellow)

# red_count = countShapes(mask_red)
# green_count = countShapes(mask_green)
# blue_count = countShapes(mask_blue)
# yellow_count = countShapes(mask_yellow)

# print(
#     f"Red shapes: {red_count}, Green shapes: {green_count}, Blue shapes: {blue_count}, Yellow shapes: {yellow_count}"
# )


while True:
    cv2.imshow("Original", img)
    cv2.imshow("Red", mask_red)
    cv2.imshow("Green", mask_green)
    cv2.imshow("Blue", mask_blue)
    cv2.imshow("Yellow", mask_yellow)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
