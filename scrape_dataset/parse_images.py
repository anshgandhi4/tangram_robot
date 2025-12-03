from collections import Counter
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from tangram import Piece, Tangram

def extract_corners_from_image(image_path):
    NUM_COLORS = 7
    DEBUG = False

    # read image, use PIL to avoid libpng warnings
    img = cv2.cvtColor(np.array(Image.open(image_path)), cv2.COLOR_BGR2HSV)

    # get count of all non-gray colors present in image
    color_counter = Counter([tuple(pixel) for row in img for pixel in row if pixel[1] != 0])

    # get top NUM_COLORS most common colors
    colors = [np.array(color) for color, _ in color_counter.most_common(NUM_COLORS)]

    # get corners for tangram shape corresponding to each color
    tangram = Tangram()
    for color in colors:
        # generate image mask
        mask = np.all(img == color, axis=-1).astype(np.uint8) * 255

        # extract contours from mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # get largest contour
        contour = None
        for c in contours:
            # process contour to remove duplicate points
            c = cv2.convexHull(c)

            # approximate polygon based on contour
            # epsilon is max distance between contour and approximate polygon, larger epsilon results in more simplified polygon
            epsilon = 0.02 * cv2.arcLength(c, True)
            corners = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)

            if len(corners) in [3, 4]:
                contour = corners
                break

        # add piece to tangram
        tangram.add_piece(Piece(contour, color))

    # process tangram
    tangram.process(img.shape[1])

    if DEBUG:
        for piece in tangram.pieces:
            center = (int(piece.pose[0]), int(piece.pose[1]))
            cv2.circle(img, center, 4, (0, 255, 255), -1)

            box = cv2.boxPoints((center, (32, 4), np.degrees(piece.pose[2]))).astype(np.int32)
            cv2.fillPoly(img, [box], (0, 255, 255))

        cv2.imshow('Output Image', cv2.cvtColor(img, cv2.COLOR_HSV2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return tangram

if __name__ == '__main__':
    for image_path in tqdm(sorted(list((Path(__file__).parent / 'tangrams').iterdir()))):
        extract_corners_from_image(image_path)
