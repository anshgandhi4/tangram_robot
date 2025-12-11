from collections import Counter
import cv2
import numpy as np
from pathlib import Path

from tangram import Piece, Tangram

global rect_final
global output_final
global counter
counter = 0
rect_final = output_final = None

def rectify(image, rect_pts, tag_size=100, center_pos=(100, 100), output_size=(500, 500)):
    pts = np.array(rect_pts[::-1], dtype=np.float32)

    d = tag_size - 1
    x, y = center_pos
    dst = np.array([[x,     y + d],
                    [x + d, y + d],
                    [x + d, y],
                    [x,     y]], dtype=np.float32)

    H, _ = cv2.findHomography(pts, dst, cv2.RANSAC, 3.0)
    rectified = cv2.warpPerspective(image, H, output_size)

    return rectified, H

def extract_corners_from_image(rgb_image, node=None):
    NUM_COLORS = 7
    REAL = True
    DEBUG = False
    ROS_PUB = True
    global rect_final
    global output_final
    global counter

    # read rgb image
    img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    if REAL:
        # img = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3), interpolation=cv2.INTER_AREA)
        # img = cv2.GaussianBlur(img, (15,15), 0)
        aruco_img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        # Create ArUco dictionary and detector parameters (4x4 tags)
        if int(cv2.__version__.split('.')[1]) >= 7: # 4._.0
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters()
        else:
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
            aruco_params = cv2.aruco.DetectorParameters_create()

        # Generate 3D positions for all of the tags
        tag_size = 0.06
        tag_position_mapping = []
        for t in range(6):
            horz = (t % 2) * 0.09
            vert = (t // 2) * 0.07567
            tag_pos = [(horz,vert,0), (horz + tag_size, vert,0), (horz + tag_size, vert + tag_size,0), (horz, vert+tag_size,0)]

            tag_position_mapping.append(tag_pos)

        # Detect ArUco markers in an image
        # node.get_logger().info('detecting aruco')
        if int(cv2.__version__.split('.')[1]) >= 7: # 4._.0
            corners, _, _ = cv2.aruco.ArucoDetector(aruco_dict, aruco_params).detectMarkers(aruco_img)
        else:
            corners, _, _ = cv2.aruco.detectMarkers(aruco_img, aruco_dict, parameters=aruco_params)
        # node.get_logger().info('detected aruco')

        found_four = False
        for cont in corners:
            if len(cont[0]) == 4:
                corners = [corner for corner in cont[0]]
                found_four = True
                break

        if found_four:
            img, _ = rectify(aruco_img, corners, tag_size=50, center_pos=(aruco_img.shape[1]//2, aruco_img.shape[0]//2), output_size=(aruco_img.shape[1], aruco_img.shape[0]))

            if ROS_PUB and node is not None:
                if rect_final is None or counter < 100:
                    rect_final = img
                    counter += 1

                node.rect_pub.publish(node.bridge.cv2_to_imgmsg(rect_final, encoding='bgr8'))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            for corner in corners:
                cv2.circle(aruco_img, (int(corner[0]), int(corner[1])), 4, (255, 0, 0), -1)

            if DEBUG:
                cv2.imshow("Detected Corners", aruco_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                cv2.imshow("Rectified", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # node.get_logger().info('rectification done')

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
        colors = [(np.array([105, 183, 126]), np.array([115, 255, 255]), 'blue'),
                  (np.array([98, 213, 163]),  np.array([104, 255, 211]), 'light blue'),
                  (np.array([71, 143, 0]),    np.array([99, 255, 171]),  'green'),
                  (np.array([21, 71, 72]),    np.array([32, 255, 255]),  'yellow'),
                  (np.array([89, 97, 163]),   np.array([119, 152, 224]), 'purple'),
                  (np.array([154, 148, 164]), np.array([173, 197, 255]), 'hot pink'),
                  ((np.array([0, 134, 151]), np.array([172, 129, 132])), (np.array([3, 201, 173]), np.array([179, 255, 198])), 'red')]
    else:
        # get count of all non-gray colors present in image
        color_counter = Counter([tuple(pixel) for row in img for pixel in row if pixel[1] != 0])

        # get top NUM_COLORS most common colors
        colors = [(np.array(color), np.array(color), '') for color, _ in color_counter.most_common(NUM_COLORS)]

    tangram = Tangram()
    if not REAL:
        tangram.prompt = str(image_path).split('tangram-')[1].split('-solution')[0].replace('-', ' ')

    # get corners for tangram shape corresponding to each color
    masks = []
    for lower, upper, color_name in colors:
        # node.get_logger().info(f'processing {color_name}')
        # generate image mask
        if REAL:
            if color_name == 'red':
                mask = cv2.inRange(img, lower[0], upper[0])
                mask2 = cv2.inRange(img, lower[1], upper[1])
                mask = cv2.bitwise_or(mask, mask2)
            else:
                mask = cv2.inRange(img, lower, upper)

            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (25, 25))
        else:
            mask = np.all(img == lower, axis=-1).astype(np.uint8) * 255
        masks.append(mask)

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

    master_mask = masks[0]
    for mask in masks[1:]:
        master_mask = cv2.bitwise_or(master_mask, mask)

    # process tangram
    flip = tangram.process(img.shape[1])
    if DEBUG and flip:
        img = np.ascontiguousarray(np.flip(img, axis=1))

    if DEBUG or (ROS_PUB and node is not None):
        for piece in tangram.pieces:
            center = (int(piece.pose[0]), int(piece.pose[1]))
            cv2.circle(img, center, 4, (12, 255, 255), -1)

            box = cv2.boxPoints((center, (32, 2), np.degrees(piece.pose[2]))).astype(np.int32)
            cv2.fillPoly(img, [box], (12, 255, 255))

        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    if DEBUG:
        cv2.imshow('Output Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if ROS_PUB and node is not None:
        if output_final is None or counter < 100:
            output_final = img
        node.masked_pub.publish(node.bridge.cv2_to_imgmsg(cv2.cvtColor(master_mask, cv2.COLOR_GRAY2BGR), encoding='bgr8'))
        node.output_pub.publish(node.bridge.cv2_to_imgmsg(output_final, encoding='bgr8'))

    return tangram

if __name__ == '__main__':
    from PIL import Image
    from tqdm import tqdm
    for image_path in tqdm(sorted(list((Path(__file__).parent / 'asdf').iterdir()))):
        extract_corners_from_image(np.array(Image.open(image_path)))
