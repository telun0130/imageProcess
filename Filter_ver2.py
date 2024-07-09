import numpy
import time
import math
import cv2
def filter_mediation(kernel_pxl, kernel_size):
    surrounds = 0.
    center_x = []
    center_y = []
    if kernel_size % 2 == 0:
        ceiling = math.ceil((kernel_size-1) / 2)
        floor = math.floor((kernel_size-1) / 2)
        center_x.append(floor)
        center_x.append(ceiling)
        center_y.append(floor)
        center_y.append(ceiling)
    else:
        center_x.append((kernel_size-1) / 2)
        center_y.append((kernel_size-1) / 2)
    for i in range(kernel_size):
        for j in range(kernel_size):
            if i in center_x and j in center_y:
                pass
            else:
                surrounds += kernel_pxl[i][j]
    result = surrounds / ((kernel_size ** 2)-1)
    return result

def filter_2(kernel_pxl, kernel_size):
    surrounds = 0.
    greater100 = numpy.sum(kernel_pxl >= 100)
    center_x = []
    center_y = []
    if kernel_size % 2 == 0:
        ceiling = math.ceil((kernel_size-1) / 2)
        floor = math.floor((kernel_size-1) / 2)
        center_x.append(floor)
        center_x.append(ceiling)
        center_y.append(floor)
        center_y.append(ceiling)
    else:
        center_x.append((kernel_size-1) / 2)
        center_y.append((kernel_size-1) / 2)
    for i in range(kernel_size):
        for j in range(kernel_size):
            if kernel_pxl[i][j] >= 100:
                surrounds += kernel_pxl[i][j]
    determind = surrounds // greater100
    if determind > 120:
        determind = 120

    for i in range(kernel_size):
        for j in range(kernel_size):
            if kernel_pxl[i][j] >= 100:
                kernel_pxl[i][j] = determind
            else:
                pass
    return kernel_pxl

def condition_blur(path, kernel_size, slide):
    img = cv2.imread(path)
    img = cv2.resize(img, (400, 800))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    padding = math.floor(kernel_size / 2)
    padding_type = cv2.BORDER_CONSTANT
    gray_image = cv2.copyMakeBorder(gray_img,
                                    top = padding,
                                    bottom = padding,
                                    left = padding,
                                    right = padding, borderType = padding_type, value=0)
    height, width = gray_image.shape[:2]
    pixel_indices = numpy.argwhere(gray_image > 200)
    filtered_locations = []
    for loc in pixel_indices:
        if not any(numpy.linalg.norm(loc - other_loc) < slide for other_loc in filtered_locations):
            filtered_locations.append(loc)
    for idx in filtered_locations:
        i = idx[0]
        j = idx[1]
        start_x = max(j - kernel_size // 2, 0)
        end_x = min(j + kernel_size // 2, width)
        start_y = max(i - kernel_size // 2, 0)
        end_y = min(i + kernel_size // 2, height)
        # If n is even, shift the region to the upper-left corner
        if kernel_size % 2 == 0:
            if end_x - start_x < kernel_size:  # Adjust end_x
                end_x += 1
            if end_y - start_y < kernel_size:  # Adjust end_y
                end_y += 1
        kernel_area = gray_image[start_y:end_y + 1, start_x:end_x + 1]
        gray_image[start_y:end_y + 1, start_x:end_x + 1] = filter_2(kernel_area, kernel_size)
    result = gray_image[padding:-padding, padding:-padding]
    cv2.imshow("before", gray_img)
    cv2.imshow("after", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result

# def time_test(func, param):
#     start_time = time.time()
#     func(param)
#     end_time = time.time()
#     Time = end_time - start_time
#     return Time

if __name__ == '__main__':
    # start_time = time.time()
    # condition_blur('image/cutting_4_0.jpg', 9, 3)
    # end_time = time.time()
    # Time2 = end_time - start_time
    # print(Time2)
    output = condition_blur('image/cutting_5_0.jpg', 9, 4)
    # cv2.imwrite('image/filter_kernel9_cutting_2_0.jpg', output)
