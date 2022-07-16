from help_scripts.python_scripts.COLMAP_functions import *
from help_scripts.python_scripts.estimate_plane import *
from help_scripts.python_scripts.color_virtual_image import *
from help_scripts.python_scripts.undistortion import compute_all_maps
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

image_dir = r'D:/research/image_stitch_with_colmap/data_300_v/'
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
    
#%%
Pv = create_virtual_camera_for_each_images_based_on_center(all_camera_matrices,plane)
w = 4000   #5000
h = 4000   #5000
f = 800    #1000
K_virt = np.asarray([[f, 0, w/2],[0, f, h/2],[0, 0, 1]])

#%%
H = {}
P_real_new = {}
#K_temp = {}
for key in all_camera_matrices:
    K_temp, dist_temp = build_intrinsic_matrix(camera_intrinsics[images[key].camera_id])
    H[key],plane_new,P_real_new[key],P_virt_trans = compute_homography(Pv[key], all_camera_matrices[key]['P'], K_virt, K_temp, plane)
    
#%%  
cut = False
if cut == True:
    cut_imgs = {}
    crop_size = 2888
    for key in imgs.keys():
        print(key)
        img = np.concatenate((np.zeros((imgs[key].shape[0] - crop_size, imgs[key].shape[1], 3), dtype = np.uint8), imgs[key][-crop_size:]))
        cut_imgs[key] = img
        
#%%
top_view_images = []
for key in imgs.keys():
    if cut == False:
        result = cv.warpPerspective(imgs[key], H[key], (w, h))
    else:   
        result = cv.warpPerspective(cut_imgs[key], H[key], (w, h))
    output = 'result-%d' % (key) + '.jpg'
    cv.imwrite(output, result[...,::-1])
    top_view_images.append(result)
    
#%%
#only pick one image for each pixel
stitched_image = np.zeros((h, w, 3))
#top_view_images.reverse()
for image in top_view_images:
    stitched_image = np.where(stitched_image == 0, image, stitched_image)
    
output = 'stitched_image.reverse.jpg'
cv.imwrite(output, stitched_image[...,::-1])  
 
#%%
#use average value for each pixel
#stitched_image = np.zeros((h, w, 3))
#for j in range(w):
    #print('Loop is on: %.2f' % (j/w*100) + '%')
    #for k in range(h):
        #count = 0
        #for i in range(len(top_view_images)):
            #if top_view_images[i][j][k][0] != 0:
                #stitched_image[j][k] += top_view_images[i][j][k]
                #count += 1
        #stitched_image[j][k] = stitched_image[j][k] / count
#output = 'stitched_image.jpg'
#cv.imwrite(output, stitched_image[...,::-1]) 


      
