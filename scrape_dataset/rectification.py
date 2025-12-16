import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
import skimage as sk

# A function to visualize points on an image in question with red squares
def viz_points_on_image(im, pts, pt_width=8):
    new_im = np.zeros(im.shape)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            for k in range(im.shape[2]):
                new_im[i][j][k] = im[i][j][k]

    for point in pts:
        x = int(point[0])
        y = int(point[1])

        new_im[y - pt_width//2:y+pt_width//2, x-pt_width//2:x+pt_width//2, :] = np.zeros((pt_width, pt_width, 3))
        new_im[y - pt_width//2:y+pt_width//2, x-pt_width//2:x+pt_width//2, 0] = 255*np.ones((pt_width, pt_width))

    return new_im

def get_most_common_colors(image, num_colors=7):
    colors = {}
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if (image[y][x][0], image[y][x][1], image[y][x][2]) not in colors:
                colors[(image[y][x][0], image[y][x][1], image[y][x][2])] = 1
            else :
                colors[(image[y][x][0], image[y][x][1], image[y][x][2])] += 1

    ordered_color_keys = list(colors.keys())
    ordered_color_keys.sort(key=lambda col: colors[col], reverse=True)

    return ordered_color_keys[:num_colors], [colors[col_key] for col_key in ordered_color_keys]

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

# Usage
if __name__ == "__main__":
    for i in range(9, 10):
        n = '2025-12-04-181003.jpg'
        imname1 = f'/home/cc/ee106a/fa25/class/ee106a-aek/Pictures/Webcam/{n}'
        
        # Read in the image
        im1 = cv2.imread(imname1)
        im1 = im1[:,:,:3]
        # im1 = im1.transpose(1,0,2)  # Convert from BGR to RGB


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
        corners, ids, _ = detector.detectMarkers(im1)

        
        print(corners[0].shape)

        found_four = False
        for cont in corners:
            if len(cont[0]) == 4:
                corners = [corn for corn in cont[0]]
                found_four = True
                break
        print(corners)

        if found_four:
            # Convert to double
            im1 = sk.img_as_float(im1)

            plt.imshow(viz_points_on_image(im1, corners)) # Matplotlib automatically handles RGB/RGBA arrays
            plt.title("Detected Corners")
            plt.show()

            # square = corners  # example detected points
            rectified, H = rectify(im1, corners, rect_size=(50,50), center_pos=(im1.shape[1]//2, im1.shape[0]//2), output_size=(im1.shape[1], im1.shape[0]))

            plt.imshow(rectified) # Matplotlib automatically handles RGB/RGBA arrays
            plt.title("Rectified")
            plt.show()
