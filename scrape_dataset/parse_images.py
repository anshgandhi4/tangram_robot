from collections import Counter
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import skimage as sk
from pprint import pprint

from tangram import Piece, Tangram

# A function to visualize points on an image in question with red squares
def viz_points_on_image(im, pts, pt_width=8):
    new_im = im.copy()

    for point in pts:
        x = int(point[0])
        y = int(point[1])

        new_im[y - pt_width//2:y+pt_width//2, x-pt_width//2:x+pt_width//2, :] = np.zeros((pt_width, pt_width, 3))
        new_im[y - pt_width//2:y+pt_width//2, x-pt_width//2:x+pt_width//2, 0] = 255*np.ones((pt_width, pt_width))

    return new_im

def color_mask(image, color, epsilon=0.18):

    channel_image = np.array([image[:,:,0], image[:,:,1], image[:,:,2]])
    channel_masks = []
    for channel_num in range(len(channel_image)):
        channel = channel_image[channel_num]
        channel_masks.append(np.abs(channel-color[channel_num]) <= epsilon)

    overall_mask = np.ones((image.shape[0], image.shape[1]))
    for c in channel_masks:
        overall_mask = overall_mask * c
    
    grouped_mask = np.stack([overall_mask for _ in range(len(channel_image))], axis=2)
   

    return overall_mask, image * grouped_mask

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
    DEBUG = True

    # read image, use PIL to avoid libpng warnings
    img = cv2.cvtColor(np.array(Image.open(image_path)), cv2.COLOR_BGR2HSV)
    img1 = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    # img1 = cv2.GaussianBlur(img1, (15,15), 0)

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
    corners, ids, _ = detector.detectMarkers(img1)

    found_four = False
    for cont in corners:
        if len(cont[0]) == 4:
            corners = [corn for corn in cont[0]]
            found_four = True
            break
    print(corners)

    if found_four:
        cv2.imshow("Detected Corners", cv2.cvtColor(viz_points_on_image(img1, corners), cv2.COLOR_BGR2RGB)) # Matplotlib automatically handles RGB/RGBA arrays
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        img, H = rectify(img1, corners, rect_size=(50,50), center_pos=(img1.shape[1]//2, img1.shape[0]//2), output_size=(img1.shape[1], img1.shape[0]))

        cv2.imshow("Rectified", cv2.cvtColor(img, cv2.COLOR_HSV2RGB)) # Matplotlib automatically handles RGB/RGBA arrays
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # get count of all non-gray colors present in image
    color_counter = Counter([tuple(pixel) for row in img for pixel in row if pixel[1] != 0])

    # get top NUM_COLORS most common colors
    colors = [np.array(color) for color, _ in color_counter.most_common(NUM_COLORS)]
    colors = [(np.array([206, 96, 56]), np.array([206, 96, 56])),
              (np.array([304, 41, 51]), np.array([304, 41, 51])),
              (np.array([58, 99, 84]), np.array([58, 99, 84])),
              (np.array([1, 53, 94]), np.array([1, 53, 94])),
              (np.array([2, 78, 98]), np.array([2, 78, 98])),
              (np.array([10, 65, 79]), np.array([10, 65, 79]))]
    colors = [(np.array([250, 63, 56]), np.array([250, 63, 56])),
              (np.array([242, 117, 115]), np.array([242, 117, 115])),
              (np.array([216, 208, 3]), np.array([216, 208, 3])),
              (np.array([0, 99, 144]), np.array([0, 99, 144])),
              (np.array([130, 77, 126]), np.array([130, 77, 126])),
              (np.array([193, 93, 78]), np.array([193, 93, 78]))]
    # e = 20
    # for color, color2 in colors:
    #     color[0] = max(0, color[0] - e)
    #     color[1] = max(0, color[1] - e)
    #     color[2] = max(0, color[2] - e)
    #     # color[2] = int(color[0] * 255 / 360) - e
    #     # color[1] = int(color[1] * 255 / 100) - e
    #     # color[0] = int(color[2] * 255 / 100) - e

    #     # color2[2] = int(color2[0] * 255 / 360) + e
    #     # color2[1] = int(color2[1] * 255 / 100) + e
    #     # color2[0] = int(color2[2] * 255 / 100) + e
    #     color2[0] = min(255, color2[0] + e)
    #     color2[1] = min(255, color2[1] + e)
    #     color2[2] = min(255, color2[2] + e)

    # get corners for tangram shape corresponding to each color
    tangram = Tangram()#prompt=str(image_path).split('tangram-')[1].split('-solution')[0].replace('-', ' '))
    for color in colors:
        # generate image mask
        mask, mask_img = color_mask(
            img,
            color[0],
            epsilon=30
        )
        mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (15, 15))
        # cv2.inRange(img, color[0], color[1])
        # pprint([[[int(z) for z in y] for y in x] for x in img])
        # mask = np.all(np.abs(img - color) < 100, axis=2img.astype(np.uint8) * 255
        # print(img.shape, np.max(img), np.min(img), img.dtype, color)
        # print(np.min(np.abs(img - color)))

        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.imshow('maskim', mask_img.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # cv2.imshow('diff', np.abs(img - color[0]).astype(np.uint8))
        # cv2.waitKey(0)

        color_img = np.zeros_like(img, dtype=np.uint8)
        color_img[:,:,0] = color[0][0]
        color_img[:,:,1] = color[0][1]
        color_img[:,:,2] = color[0][2]
        # cv2.imshow('color', cv2.cvtColor(color_img, cv2.COLOR_HSV2BGR))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # extract contours from mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        if contour is not None:
            tangram.add_piece(Piece(contour, color[0]))

        # if DEBUG:
        #     # display mask
        #     mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        #     cv2.imshow('Mask', mask_display)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        #     # display masked image
        #     masked_image = cv2.bitwise_and(img, img, mask=mask)
        #     masked_image = cv2.cvtColor(masked_image, cv2.COLOR_HSV2BGR)
        #     cv2.imshow('Masked Image', masked_image)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

    # process tangram
    flip = tangram.process(img.shape[1])
    if DEBUG and flip:
        img = np.ascontiguousarray(np.flip(img, axis=1))

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
    for image_path in tqdm(sorted(list((Path(__file__).parent / 'asdf').iterdir()))):
        extract_corners_from_image(image_path)
