from help_scripts.python_scripts.COLMAP_functions import *
from help_scripts.python_scripts.estimate_plane import *
from help_scripts.python_scripts.color_virtual_image import *
from help_scripts.python_scripts.undistortion import compute_all_maps
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
#%%
# Perform the reconstruction to get data
#automatic_reconstructor()
# image_undistorter()
# stereo_fusion()

# image_dir = r'/Users/ludvig/Documents/SSY226 Design project in MPSYS/Image-stitching-with-COLMAP/COLMAP_w_CUDA/'
image_dir = r'D:/research/image_stitch_with_colmap/data/'
cameras, points3D, images = get_data_from_binary(image_dir)
# print(cameras)
# print(images)
coordinates = []
for key in points3D:
    coordinates.append(points3D[key].xyz)
coordinates = np.asarray(coordinates)

#Estimate a floor plane
plane, _ = ransac_find_plane(coordinates, threshold=0.01)

# Get all camera matrices and images
camera_intrinsics = {}
all_camera_matrices = {}
imgs = {}
# image_dir = '../COLMAP_w_CUDA/images/'
# image_dir = '../COLMAP_w_CUDA/dense/0/images/'

# maps = compute_all_maps(image_dir, full_size_img=False)

real_image_dir = image_dir + 'images/'

# Rearrange COLMAP data
for key in images.keys():
    # print('cameraid, name', images[key].camera_id, cameras[key].id)
    print('imageid, cameraid', key, images[key].camera_id)
    # imgs[images[key].camera_id] = np.asarray(plt.imread(image_dir+"images/" + images[key].name))
    imgs[key] = np.asarray(plt.imread(real_image_dir + images[key].name))

    # map_x, map_y = maps[key]
    # imgs[images[key].camera_id] = cv.remap(imgs[images[key].camera_id], map_x, map_y, cv.INTER_LANCZOS4)
    # imgs[key] = cv.remap(imgs[key], map_x, map_y, cv.INTER_LANCZOS4)

    # all_camera_matrices[images[key].camera_id] = camera_quat_to_P(images[key].qvec, images[key].tvec)
    all_camera_matrices[key] = camera_quat_to_P(images[key].qvec, images[key].tvec)
    # camera_intrinsics[cameras[key].id] = cameras[key]
    camera_intrinsics[images[key].camera_id] = cameras[images[key].camera_id]
#%%
# VIRTUAL CAMERA WITH MEAN CENTER OF OTHER CAMERAS AND PRINCIPAL AXIS AS PLANE NORMAL
# Pv = create_virtual_camera(all_camera_matrices,plane)
# Pv = create_virtual_camera_for_each_images(all_camera_matrices,plane)
Pv = create_virtual_camera_for_each_images_based_on_center(all_camera_matrices,plane)
# w = 500
# h = 500
# f = 250
w = 3000   #5000
h = 3000   #5000
f = 600    #1000
K_virt = np.asarray([[f, 0, w/2],[0, f, h/2],[0, 0, 1]])
# f, cx, cy, _ = cameras[1].params
# K_virt = np.asarray([[f, 0, cx],[0, f, cy],[0, 0, 1]])

# TEST WITH EXISTING CAMERA
# K_temp, dist_temp = build_intrinsic_matrix(camera_intrinsics[1])
# Pv = all_camera_matrices[1]['P']
# K_virt = K_temp
# w = int(K_virt[0, 2]*2)
# h = int(K_virt[1, 2]*2)

# TEST HOMOGRAPHY 2.0
H = {}
P_real_new = {}

for key in all_camera_matrices:
    # print('Key vs cam id', key, camera_intrinsics[key].id)
    K_temp, dist_temp = build_intrinsic_matrix(camera_intrinsics[1])
    # H[key],plane_new,P_real_new[key],P_virt_trans = compute_homography(Pv, all_camera_matrices[key]['P'], K_virt, K_temp, plane)
    H[key],plane_new,P_real_new[key],P_virt_trans = compute_homography(Pv[key-1], all_camera_matrices[key]['P'], K_virt, K_temp, plane)
#%%
#print('HOMO',H)# color image
#color_images, stitched_image = color_virtual_image(plane, Pv, w, h, imgs, all_camera_matrices, camera_intrinsics, K_virt,'homography',H)

# top_view_images = get_top_view(plane, Pv, w, h, imgs, all_camera_matrices, camera_intrinsics, K_virt, H)
top_view_images = get_top_view_every_ten(plane, Pv, w, h, imgs, all_camera_matrices, camera_intrinsics, K_virt, H)
#stitched_image = stitched_image/255

#for i in range(len(top_view_images)):
    #output = 'result-%d' % i + '.jpg'
    #cv.imwrite(output, top_view_images[i])
    
#imgplot = plt.imshow(stitched_image)
#plt.imsave('stitched_image1.jpg', stitched_image)
#plt3d = plot_3D(points3D,plane,all_camera_matrices,Pv)
#%%
stitched_image = np.zeros((h, w, 3))
for j in range(h):
    print('Loop is on: %.2f' % (j/h*100) + '%')
    for k in range(w):
        not_nan_count = len(top_view_images)
        for i in range(len(top_view_images)):
            if (np.isnan(top_view_images[10*i][j][k][0]) == True):
                not_nan_count = not_nan_count - 1
            else:
                stitched_image[j][k] += top_view_images[10*i][j][k]
        if not_nan_count != 0:
            stitched_image[j][k] = stitched_image[j][k]/not_nan_count

output = 'stitched_image.jpg'
cv.imwrite(output, stitched_image[...,::-1])
#%%
#PLOT TRANSFORMED PLANE WITH TRANSFORMED CAMERAS (HOMOGRAPHY)
a, b, c, d = plane_new
x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)
X, Y = np.meshgrid(x, y)
Z = (d + a * X + b * Y) / -c
plt4d = plt.figure().gca(projection='3d', autoscale_on=False)
plt4d.plot_surface(X, Y, Z, alpha=0.5)
cam_center, principal_axis = get_camera_center_and_axis(P_virt_trans)
plt4d.quiver(cam_center[0],cam_center[1],cam_center[2], principal_axis[0,0], principal_axis[0,1], principal_axis[0,2], length=d, color='r')
colors = {1: 'r', 2: 'b', 3: 'g', 4: 'c'}
for key in P_real_new:
    cam_center, principal_axis = get_camera_center_and_axis(P_real_new[key])
    plt4d.quiver(cam_center[0],cam_center[1],cam_center[2], principal_axis[0,0], principal_axis[0,1], principal_axis[0,2], length=1, color=colors[key])

# Test for visualizing the projection of a virtual pixel to the plane
# pixelpoint = [0,0]
# line_dir,line_point = line_from_pixel(pixelpoint,Pv,K_virt)
# intersection_point = intersection_line_plane(line_dir,line_point,plane)
# plt3d.scatter3D(intersection_point[0], intersection_point[1], intersection_point[2],  cmap='Blues')
# plt3d.quiver(line_point[0],line_point[1],line_point[2], line_dir[0], line_dir[1], line_dir[2], length=10, color='y')
# cam_center, principal_axis = get_camera_center_and_axis(Pv)
# plt3d.quiver(cam_center[0],cam_center[1],cam_center[2], principal_axis[0,0], principal_axis[0,1], principal_axis[0,2], length=distance, color='Red')
plt.show()
plt.savefig('result1.jpg')