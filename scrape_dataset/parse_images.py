from collections import Counter
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def extract_corners_from_image(image_path):
    NUM_COLORS = 7
    DEBUG = False

    # read image, use PIL to avoid libpng warnings
    img = cv2.cvtColor(np.array(Image.open(image_path)), cv2.COLOR_BGR2HSV)

    # get count of all colors present in image
    color_counter = Counter([tuple(col) for row in img for col in row])

    # get top NUM_COLORS most common non-gray colors
    colors = [np.array(color) for color, _ in color_counter.most_common(NUM_COLORS) if color[1] != 0]

    # get corners for tangram shape corresponding to each color
    corners = {}
    for color in colors:
        # generate image mask
        mask = np.all(img == color, axis=-1).astype(np.uint8) * 255

        # get contours around masked objects
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        # choose largest contour
        contour = contours[0]

        # approximate polygon based on contour
        # epsilon is max distance between contour and approximate polygon, larger epsilon results in more simplified polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
        corners[color] = [corner[0] for corner in approx_polygon]

        if DEBUG:
            # display mask
            mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.imshow('Mask', mask_display)
            cv2.waitKey(0)

            # display masked image
            masked_image = cv2.bitwise_and(img, img, mask=mask)
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_HSV2BGR)
            cv2.imshow('Masked Image', masked_image)
            cv2.waitKey(0)

            # display image with marked corners
            corner_image = img.copy()

            for corner in corners[color]:
                cv2.circle(corner_image, tuple(corner), 4, (0, 255, 255), -1)

            cv2.imshow('Corner Image', cv2.cvtColor(corner_image, cv2.COLOR_HSV2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return corners

if __name__ == '__main__':
    for image_path in tqdm((Path(__file__).parent / 'tangrams').iterdir()):
        extract_corners_from_image(image_path)
