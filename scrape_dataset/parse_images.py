from collections import Counter
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from tangram import Piece, Tangram

def rectify(image, rect_pts, rect_size=(100, 100), center_pos=(100, 100), output_size=(500, 500)):
    """
    triangle_pts: list or array of 3 points (x,y) in any order
    output_size: (W, H) of output rectangle
    """
    pts = np.array(rect_pts, dtype=np.float32)

    w, h = rect_size
    centx, centy = center_pos
    dst = np.array([
        [centx, centy],
        [centx, centy+h-1],
        [centx+w-1, centy+h-1],
        [centx+w-1, centy]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(pts, dst, cv2.RANSAC, 3.0)
    rectified = cv2.warpPerspective(image, H, output_size)

    return rectified, H

def extract_corners_from_image(image_path):
    NUM_COLORS = 7
    REAL = True
    DEBUG = True

    # read image, use PIL to avoid libpng warnings
    img = cv2.cvtColor(np.array(Image.open(image_path)), cv2.COLOR_RGB2HSV)
    if REAL:
        img = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3), interpolation=cv2.INTER_AREA)
        # img = cv2.GaussianBlur(img, (15,15), 0)
        aruco_img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        # Create ArUco dictionary and detector parameters (4x4 tags)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()

        # Generate 3D positions for all of the tags
        tag_size = 0.06
        tag_position_mapping = []
        for t in range(6):
            horz = (t % 2) * 0.09
            vert = (t // 2) * 0.07567
            tag_pos = [(horz,vert,0), (horz + tag_size, vert,0), (horz + tag_size, vert + tag_size,0), (horz, vert+tag_size,0)]

            tag_position_mapping.append(tag_pos)

        # Detect ArUco markers in an image
        # Returns: corners (list of numpy arrays), ids (numpy array)
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, _, _ = detector.detectMarkers(aruco_img)

        found_four = False
        for cont in corners:
            if len(cont[0]) == 4:
                corners = [corner for corner in cont[0]]
                found_four = True
                break

        if found_four:
            img, _ = rectify(aruco_img, corners, rect_size=(50,50), center_pos=(aruco_img.shape[1]//2, aruco_img.shape[0]//2), output_size=(aruco_img.shape[1], aruco_img.shape[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            for corner in corners:
                cv2.circle(aruco_img, (int(corner[0]), int(corner[1])), 4, (255, 0, 0), -1)

            cv2.imshow("Detected Corners", aruco_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imshow("Rectified", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # # Interactive HSV threshold sliders
    # def update_threshold(*args):
    #     min_h = cv2.getTrackbarPos('Min H', 'HSV Threshold')
    #     min_s = cv2.getTrackbarPos('Min S', 'HSV Threshold')
    #     min_v = cv2.getTrackbarPos('Min V', 'HSV Threshold')
    #     max_h = cv2.getTrackbarPos('Max H', 'HSV Threshold')
    #     max_s = cv2.getTrackbarPos('Max S', 'HSV Threshold')
    #     max_v = cv2.getTrackbarPos('Max V', 'HSV Threshold')

    #     lower_bound = np.array([min_h, min_s, min_v])
    #     upper_bound = np.array([max_h, max_s, max_v])
    #     print(min_h, min_s, min_v, max_h, max_s, max_v)
    #     print(f'np.array([{min_h}, {min_s}, {min_v}]), np.array([{max_h}, {max_s}, {max_v}])')

    #     mask = cv2.inRange(img, lower_bound, upper_bound)
    #     result = cv2.bitwise_and(img, img, mask=mask)
    #     result_bgr = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

    #     # Display original, mask, and result side by side
    #     mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    #     img_bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    #     # Create a combined display
    #     h, w = img.shape[:2]
    #     combined = np.hstack([img_bgr, mask_bgr, result_bgr])
    #     cv2.imshow('HSV Threshold', combined)

    # # Create window and trackbars
    # cv2.namedWindow('HSV Threshold', cv2.WINDOW_NORMAL)
    # cv2.createTrackbar('Min H', 'HSV Threshold', 0, 179, update_threshold)
    # cv2.createTrackbar('Min S', 'HSV Threshold', 0, 255, update_threshold)
    # cv2.createTrackbar('Min V', 'HSV Threshold', 0, 255, update_threshold)
    # cv2.createTrackbar('Max H', 'HSV Threshold', 179, 179, update_threshold)
    # cv2.createTrackbar('Max S', 'HSV Threshold', 255, 255, update_threshold)
    # cv2.createTrackbar('Max V', 'HSV Threshold', 255, 255, update_threshold)

    # # Initial display
    # update_threshold()

    # # Run until user quits (press 'q' or close window)
    # print("Adjust sliders to control HSV thresholds. Press 'q' to quit.")
    # while True:
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord('q') or cv2.getWindowProperty('HSV Threshold', cv2.WND_PROP_VISIBLE) < 1:
    #         break

    # cv2.destroyAllWindows()

    if REAL:
        colors = [(np.array([79, 145, 109]), np.array([108, 255, 237]), 'blue'),
                  (np.array([36, 168, 82]),  np.array([68, 255, 255]),  'green'),
                  (np.array([25, 208, 154]), np.array([36, 255, 255]),  'yellow'),
                  (np.array([131, 91, 90]),  np.array([179, 255, 255]), 'purple'),
                  (np.array([0, 171, 154]),  np.array([11, 196, 176]),  'terracotta'),
                  (np.array([0, 143, 183]),  np.array([11, 173, 209]),  'pink'),
                  (np.array([0, 199, 165]),  np.array([11, 242, 216]),  'red')]
    else:
        # get count of all non-gray colors present in image
        color_counter = Counter([tuple(pixel) for row in img for pixel in row if pixel[1] != 0])

        # get top NUM_COLORS most common colors
        colors = [(np.array(color), np.array(color), '') for color, _ in color_counter.most_common(NUM_COLORS)]

    tangram = Tangram()
    if not REAL:
        tangram.prompt = str(image_path).split('tangram-')[1].split('-solution')[0].replace('-', ' ')

    # get corners for tangram shape corresponding to each color
    for lower, upper, color_name in colors:
        # generate image mask
        if REAL:
            mask = cv2.inRange(img, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (25, 25))
        else:
            mask = np.all(img == lower, axis=-1).astype(np.uint8) * 255

        # extract contours from mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # get largest contour
        contour = None
        max_area = 0
        for c in contours:
            area = cv2.contourArea(c)

            # process contour to remove redundant points
            c = cv2.convexHull(c)

            # approximate polygon based on contour
            # epsilon is max distance between contour and approximate polygon, larger epsilon results in more simplified polygon
            epsilon = 0.02 * cv2.arcLength(c, True)
            corners = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)

            if REAL and DEBUG:
                contour_img = img.copy()
                cv2.drawContours(contour_img, c, -1, (50, 255, 255), 3)
                cv2.imshow(f'{color_name} Contour Image', cv2.cvtColor(contour_img, cv2.COLOR_HSV2BGR))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if len(corners) in [3, 4]:
                if area > max_area:
                    contour = corners
                    max_area = area
                    if not REAL:
                        break

        # add piece to tangram
        if contour is not None:
            tangram.add_piece(Piece(contour, color_name))

        if DEBUG:
            # display mask
            mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.imshow(f'{color_name} Mask', mask_display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # display masked image
            masked_image = cv2.bitwise_and(img, img, mask=mask)
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_HSV2BGR)
            cv2.imshow(f'{color_name} Masked Image', masked_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # display image with marked corners
            if contour is not None:
                corner_image = img.copy()
                for corner in contour:
                    cv2.circle(corner_image, tuple(corner), 4, (0, 255, 255), -1)

                cv2.imshow(f'{color_name} Corner Image', cv2.cvtColor(corner_image, cv2.COLOR_HSV2BGR))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # process tangram
    flip = tangram.process(img.shape[1])
    if DEBUG and flip:
        img = np.ascontiguousarray(np.flip(img, axis=1))

    if DEBUG:
        for piece in tangram.pieces:
            center = (int(piece.pose[0]), int(piece.pose[1]))
            cv2.circle(img, center, 4, (12, 255, 255), -1)

            box = cv2.boxPoints((center, (32, 2), np.degrees(piece.pose[2]))).astype(np.int32)
            cv2.fillPoly(img, [box], (12, 255, 255))

        cv2.imshow('Output Image', cv2.cvtColor(img, cv2.COLOR_HSV2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return tangram

if __name__ == '__main__':
    for image_path in tqdm(sorted(list((Path(__file__).parent / 'asdf').iterdir()))):
        extract_corners_from_image(image_path)
