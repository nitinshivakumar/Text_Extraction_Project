import cv2
import numpy as np

# Function to convert the image to grayscale
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function to resize the image
def resize_image(image):
    height, width = image.shape
    resized_img = cv2.resize(image, (width * 2, height * 2))
    return resized_img

# Function to remove noise from the image using median blur
def remove_noise(image):
    return cv2.medianBlur(image, 3)

# Function for thresholding to create a binary image
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Function for dilation to enhance features in the image
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

# Function for erosion to reduce the size of objects in the image
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

# Function for opening (erosion followed by dilation)
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Function for Canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

# Function for deskewing (correcting skewness) in the image
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# Function to display an image using OpenCV
def call_func(image):
    cv2.imshow('sample image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()