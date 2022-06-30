from help_scripts.python_scripts.COLMAP_functions import *
from help_scripts.python_scripts.estimate_plane import *
from help_scripts.python_scripts.color_virtual_image import *
from help_scripts.python_scripts.undistortion import compute_all_maps
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
#%%
image_dir = r'D:/research/image_stitch_with_colmap/data_327/'
real_image_dir = image_dir + 'images/'
cameras, points3D, images = get_data_from_binary(image_dir)

coordinates = []
for key in points3D:
    coordinates.append(points3D[key].xyz)
coordinates = np.asarray(coordinates)

plane, _ = ransac_find_plane(coordinates, threshold=0.01)

camera_intrinsics = {}
all_camera_matrices = {}
imgs = {}

for key in images.keys():
    print('imageid, cameraid', key, images[key].camera_id)
    imgs[key] = np.asarray(plt.imread(real_image_dir + images[key].name))
    all_camera_matrices[key] = camera_quat_to_P(images[key].qvec, images[key].tvec)
    camera_intrinsics[images[key].camera_id] = cameras[images[key].camera_id]

Pv = create_virtual_camera_for_each_images_based_on_center(all_camera_matrices,plane)
w = 4000   #5000
h = 4000   #5000
f = 600    #1000
K_virt = np.asarray([[f, 0, w/2],[0, f, h/2],[0, 0, 1]])

H = {}
for key in all_camera_matrices:
    K_temp, = build_intrinsic_matrix(camera_intrinsics[1])
    H[key], = compute_homography(Pv[key-1], all_camera_matrices[key]['P'], K_virt, K_temp, plane)
#%%  
top_view_images = {}
for i in range(len(imgs)):
    M = np.matmul(K_virt,np.linalg.inv(H[i+1]))
    M = np.matmul(M,np.linalg.inv(K_temp))
    M = M  / M[-1][-1]
    test = cv.warpPerspective(imgs[i+1], M, (w, h))
    output = 'result-%d' % (i+1) + '.jpg'
    cv.imwrite(output, test[...,::-1])
    top_view_images[i] = test
#%%
stitched_image = np.zeros((h, w, 3))
for j in range(w):
    print('Loop is on: %.2f' % (j/w*100) + '%')
    for k in range(h):
        for i in range(len(top_view_images)):
            if stitched_image[j][k][0] == 0:
                stitched_image[j][k] = top_view_images[i][j][k]
            if stitched_image[j][k][0] != 0:
                break
output = 'stitched_image.jpg'
cv.imwrite(output, stitched_image[...,::-1])             
            
    
    
    
    
    