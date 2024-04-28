import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import math
import time

def imshow(image):
    if len(image.shape) == 3:
      pass
    else:
      image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

p = int(input('Enter p:'))
macroblock_size = int(input('Enter macroblock size:'))
search_method = int(input('Select search method (Enter 1 for full search, 2 for 2D search):'))
method_name = 'full' if search_method == 1 else '2d'


# (1)
ref_img = cv2.imread('./img/40.jpg')
ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

target_img = cv2.imread('./img/42.jpg')
target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)


# divide input images into macroblocks
x_lim = ref_img.shape[0]
y_lim = ref_img.shape[1]

num_macroblocks_x = x_lim // macroblock_size
num_macroblocks_y = y_lim // macroblock_size
target_macroblocks = np.zeros((num_macroblocks_x, num_macroblocks_y, macroblock_size, macroblock_size, 3), dtype=float)
ref_macroblocks = np.zeros((num_macroblocks_x, num_macroblocks_y, macroblock_size, macroblock_size, 3), dtype=float)

for y in range(num_macroblocks_y):
   for x in range(num_macroblocks_x):
      target_macroblocks[x, y] = target_img[x*macroblock_size:(x+1)*macroblock_size, y*macroblock_size:(y+1)*macroblock_size]

for y in range(num_macroblocks_y):
   for x in range(num_macroblocks_x):
      ref_macroblocks[x, y] = ref_img[x*macroblock_size:(x+1)*macroblock_size, y*macroblock_size:(y+1)*macroblock_size]


# full search motion estimation
def motion_estimation_full_search(x, y, reference_img, search_range):
    best_match = None
    best_sad = np.inf
    # get the center pixel of the current macroblock
    cy = x*macroblock_size + macroblock_size // 2
    cx = y*macroblock_size + macroblock_size // 2
    # iterate over the search range to find the best match
    for dy in range(-search_range, search_range+1):
        for dx in range(-search_range, search_range+1):
            # compute the coordinates of the corresponding macroblock in the reference frame
            ry = cy + dy
            rx = cx + dx
            # check if the macroblock is within the reference image
            if ry - macroblock_size//2 >= 0 and ry + macroblock_size//2 <= reference_img.shape[0] and rx - macroblock_size//2 >= 0 and rx + macroblock_size//2 <= reference_img.shape[1]:
                # extract the corresponding macroblock from the reference frame
                reference_mb = reference_img[(ry-macroblock_size//2):(ry+macroblock_size//2), (rx-macroblock_size//2):(rx+macroblock_size//2)]
                # compute the sum of absolute differences (SAD) between the macroblocks
                sad = np.abs(target_macroblocks[x, y] - reference_mb)
                sad = np.sum(sad)
                # update the best match if the current SAD is smaller
                if sad < best_sad:
                    best_match = (dx, dy)
                    best_sad = sad
    return best_match, best_sad

def motion_estimation_2d_search(x, y, reference_img, search_range):
   best_match = None
   best_sad = np.inf
   accum_best_match = (0, 0)
   # get the center pixel of the current macroblock
   fix_cy = x*macroblock_size + macroblock_size // 2
   fix_cx = y*macroblock_size + macroblock_size // 2
   mb_y_low_limit = fix_cy - search_range*2
   mb_y_up_limit = fix_cy + search_range*2
   mb_x_low_limit = fix_cx - search_range*2
   mb_x_up_limit = fix_cx + search_range*2
   target_mb = target_img[int(fix_cy-macroblock_size//2):int(fix_cy+macroblock_size//2), int(fix_cx-macroblock_size//2):int(fix_cx+macroblock_size//2)]
   cy = fix_cy
   cx = fix_cx
   # set the initial search step size
   n_prime = math.floor(math.log2(search_range))
   n = max(2, 2**(n_prime-1))
   # iterate until the step size is 1
   while n > 1:
      # step 2
      m = [(0, 0), (n, 0), (0, n), (-n, 0), (0, -n)]
      while True:
         # step 3
         for dx, dy in m:
            # compute the coordinates of the corresponding macroblock in the reference frame
            ry = cy + dy
            rx = cx + dx
            # check if the macroblock is within the target image
            if ry - macroblock_size//2 >= 0 and ry + macroblock_size//2 <= target_img.shape[0] and rx - macroblock_size//2 >= 0 and rx + macroblock_size//2 <= target_img.shape[1]:
                  # extract the corresponding macroblock from the reference frame
                  reference_mb = reference_img[int(ry-macroblock_size//2):int(ry+macroblock_size//2), int(rx-macroblock_size//2):int(rx+macroblock_size//2)]
                  # compute the sum of absolute differences (SAD) between the macroblocks
                  sad = np.abs(target_mb - reference_mb)
                  sad = np.sum(sad)
                  # update the best match if the current SAD is smaller
                  if sad <= best_sad:
                     best_match = (dx, dy)
                     best_sad = sad
         accum_best_match = tuple([accum_best_match[0] + best_match[0], accum_best_match[1] + best_match[1]])
         if best_match == (0, 0):
            # print('out')
            break
         # step 4
         cy = cy + best_match[1]
         cx = cx + best_match[0]
         reversed_best_match = tuple([-best_match[0], -best_match[1]])
         if reversed_best_match in m:
            m.remove(reversed_best_match)
      # step 5
      n = n / 2
   # step 6
   final_m = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
   for dx, dy in final_m:
      # compute the coordinates of the corresponding macroblock in the reference frame
      ry = cy + dy
      rx = cx + dx
      # check if the macroblock is within the target image
      if ry - macroblock_size//2 >= 0 and ry + macroblock_size//2 <= target_img.shape[0] and rx - macroblock_size//2 >= 0 and rx + macroblock_size//2 <= target_img.shape[1]:
            # extract the corresponding macroblock from the reference frame
            reference_mb = reference_img[int(ry-macroblock_size//2):int(ry+macroblock_size//2), int(rx-macroblock_size//2):int(rx+macroblock_size//2)]
            # compute the sum of absolute differences (SAD) between the macroblocks
            sad = np.abs(target_mb - reference_mb)
            sad = np.sum(sad)
            # update the best match if the current SAD is smaller
            if sad < best_sad:
               best_match = (dx, dy)
               best_sad = sad
   accum_best_match = tuple([accum_best_match[0] + best_match[0], accum_best_match[1] + best_match[1]])
   cy += best_match[1]
   cx += best_match[0]

   return accum_best_match, best_sad

motion_vectors = np.zeros((num_macroblocks_x, num_macroblocks_y, 2), dtype=np.int8)
macroblocks_sad = np.zeros((num_macroblocks_x, num_macroblocks_y))
#get start time
start_time = time.time()
for y in range(num_macroblocks_y):
   for x in range(num_macroblocks_x):
      if search_method == 1:
         motion_vectors[x, y], macroblocks_sad[x, y] = motion_estimation_full_search(x, y, ref_img, p)
      elif search_method == 2:
         motion_vectors[x, y], macroblocks_sad[x, y] = motion_estimation_2d_search(x, y, ref_img, p)
#calculate execution time
execution_time = time.time() - start_time
print('Execution Time:', execution_time)

# predicted image
pred_img = np.zeros((x_lim, y_lim, 3), dtype=np.uint8)
for y in range(num_macroblocks_y):
    for x in range(num_macroblocks_x):
        # calculate the center pixel coordinates of the current macroblock
        cx = y * macroblock_size + macroblock_size//2
        cy = x * macroblock_size + macroblock_size//2
        # get the motion vector for the current macroblock
        motion_vector = (dx, dy) = motion_vectors[x, y]
        # calculate the corresponding macroblock in the reference frame image1
        rx = cx + dx
        ry = cy + dy
        ref_macroblock = ref_img[ry-macroblock_size//2:ry+macroblock_size//2, rx-macroblock_size//2:rx+macroblock_size//2]
        # replace the macroblock in the predicted image with the reference macroblock
        pred_img[(cy - macroblock_size//2):(cy + macroblock_size//2), (cx - macroblock_size//2):cx + macroblock_size//2] = ref_macroblock
imshow(pred_img)
cv2.imwrite('./out/'+method_name+'_predicted_r'+str(p)+'_b'+str(macroblock_size)+'.png', cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))


# draw motion vectors
img_with_vectors = target_img.copy()
for j in range(num_macroblocks_y):
   for i in range(num_macroblocks_x):
      # Get the center of the current macroblock
        x_center = j * macroblock_size + macroblock_size//2
        y_center = i * macroblock_size + macroblock_size//2
        # Get the motion vector for the current macroblock
        motion_x = motion_vectors[i,j,0]
        motion_y = motion_vectors[i,j,1]
        # Draw a line representing the motion vector on the current frame
        cv2.arrowedLine(img_with_vectors, (x_center, y_center), (x_center+motion_x, y_center+motion_y), (255, 0, 0), 1, tipLength=0.3)

# show the img with motion vectors
imshow(img_with_vectors)
cv2.imwrite('./out/'+method_name+'_motion_vector_r'+str(p)+'_b'+str(macroblock_size)+'.png', cv2.cvtColor(img_with_vectors, cv2.COLOR_RGB2BGR))


# residual image
residual_img = cv2.absdiff(target_img, pred_img)
imshow(residual_img)
cv2.imwrite('./out/'+method_name+'_residual_r'+str(p)+'_b'+str(macroblock_size)+'.png', cv2.cvtColor(residual_img, cv2.COLOR_RGB2BGR))


#compute total SAD values
total_sad = np.sum(macroblocks_sad)
print('Total SAD values:', total_sad)

#compute PSNR value
mse = np.mean((target_img.astype("float")-pred_img.astype("float"))**2)
max_pixel = 255.0
psnr = 20*math.log10(max_pixel / math.sqrt(mse))
print('PSNR value:', psnr)


# (2)
p = 8
macroblock_size = 8
search_method = 1 # full search

ref_img = cv2.imread('./img/40.jpg')
ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

target_img = cv2.imread('./img/51.jpg')
target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)


# divide input images into macroblocks
x_lim = ref_img.shape[0]
y_lim = ref_img.shape[1]

num_macroblocks_x = x_lim // macroblock_size
num_macroblocks_y = y_lim // macroblock_size
target_macroblocks = np.zeros((num_macroblocks_x, num_macroblocks_y, macroblock_size, macroblock_size, 3), dtype=float)
ref_macroblocks = np.zeros((num_macroblocks_x, num_macroblocks_y, macroblock_size, macroblock_size, 3), dtype=float)

for y in range(num_macroblocks_y):
   for x in range(num_macroblocks_x):
      target_macroblocks[x, y] = target_img[x*macroblock_size:(x+1)*macroblock_size, y*macroblock_size:(y+1)*macroblock_size]

for y in range(num_macroblocks_y):
   for x in range(num_macroblocks_x):
      ref_macroblocks[x, y] = ref_img[x*macroblock_size:(x+1)*macroblock_size, y*macroblock_size:(y+1)*macroblock_size]
      

motion_vectors = np.zeros((num_macroblocks_x, num_macroblocks_y, 2), dtype=np.int8)
macroblocks_sad = np.zeros((num_macroblocks_x, num_macroblocks_y))
#get start time
start_time = time.time()
for y in range(num_macroblocks_y):
   for x in range(num_macroblocks_x):
      if search_method == 1:
         motion_vectors[x, y], macroblocks_sad[x, y] = motion_estimation_full_search(x, y, ref_img, p)
      elif search_method == 2:
         motion_vectors[x, y], macroblocks_sad[x, y] = motion_estimation_2d_search(x, y, ref_img, p)
#calculate execution time
execution_time = time.time() - start_time
print('Execution Time:', execution_time)

# predicted image
pred_img = np.zeros((x_lim, y_lim, 3), dtype=np.uint8)
for y in range(num_macroblocks_y):
    for x in range(num_macroblocks_x):
        # calculate the center pixel coordinates of the current macroblock
        cx = y * macroblock_size + macroblock_size//2
        cy = x * macroblock_size + macroblock_size//2
        # get the motion vector for the current macroblock
        motion_vector = (dx, dy) = motion_vectors[x, y]
        # calculate the corresponding macroblock in the reference frame image1
        rx = cx + dx
        ry = cy + dy
        ref_macroblock = ref_img[ry-macroblock_size//2:ry+macroblock_size//2, rx-macroblock_size//2:rx+macroblock_size//2]
        # replace the macroblock in the predicted image with the reference macroblock
        pred_img[(cy - macroblock_size//2):(cy + macroblock_size//2), (cx - macroblock_size//2):cx + macroblock_size//2] = ref_macroblock
imshow(pred_img)
# cv2.imwrite('predicted_experiment.png', cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))


# draw motion vectors
img_with_vectors = target_img.copy()
for j in range(num_macroblocks_y):
   for i in range(num_macroblocks_x):
      # Get the center of the current macroblock
        x_center = j * macroblock_size + macroblock_size//2
        y_center = i * macroblock_size + macroblock_size//2
        # Get the motion vector for the current macroblock
        motion_x = motion_vectors[i,j,0]
        motion_y = motion_vectors[i,j,1]
        # Draw a line representing the motion vector on the current frame
        cv2.arrowedLine(img_with_vectors, (x_center, y_center), (x_center+motion_x, y_center+motion_y), (255, 0, 0), 1, tipLength=0.3)


# show the img with motion vectors
imshow(img_with_vectors)
# cv2.imwrite('motion_vector_experiment.png', cv2.cvtColor(img_with_vectors, cv2.COLOR_RGB2BGR))

# residual image
residual_img = cv2.absdiff(target_img, pred_img)
imshow(residual_img)
# cv2.imwrite('residual_experiment.png', cv2.cvtColor(residual_img, cv2.COLOR_RGB2BGR))

#compute total SAD values
total_sad = np.sum(macroblocks_sad)
print('Total SAD values:', total_sad)

#compute PSNR value
mse = np.mean((target_img.astype("float")-pred_img.astype("float"))**2)
max_pixel = 255.0
psnr = 20*math.log10(max_pixel / math.sqrt(mse))
print('PSNR value:', psnr)
