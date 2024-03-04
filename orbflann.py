import cv2
import numpy as np
import os

# Initialize ORB detector
orb = cv2.ORB_create()

# FLANN parameters
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Path to directories
left_images_dir = 'DrivingStereo_demo_images/image_L'

right_images_dir = 'DrivingStereo_demo_images/image_R'

# Function to get full image paths
def get_image_paths(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')]

# Sorting to ensure matching images are aligned
left_images = sorted(get_image_paths(left_images_dir))
right_images = sorted(get_image_paths(right_images_dir))

for img_L_path, img_R_path in zip(left_images, right_images):
    img_L = cv2.imread(img_L_path)
    img_R = cv2.imread(img_R_path)

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img_L, None)
    kp2, des2 = orb.detectAndCompute(img_R, None)

    # Matching descriptor vectors using FLANN matcher
    matches = flann.match(des1, des2)

    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches
    img_matches = cv2.drawMatches(img_L, kp1, img_R, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the matched image
    cv2.imshow('Feature Matching', img_matches)
    cv2.waitKey(1)  # Use cv2.waitKey(0) if you want to wait for a key press between each pair

cv2.destroyAllWindows()
