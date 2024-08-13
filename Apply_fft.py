import cv2
import cv2 as cv
import numpy
from patchify import patchify
from scipy.stats import entropy

distance_threshold = 12
center_width = 400
center_height = 800

def Hist(image):
    hist = cv.calcHist([image], [0], None, [256], [0, 256])
    return hist
def cumHalf(image):
    hist = Hist(image)
    total = numpy.sum(hist)
    cumHist = numpy.cumsum(hist)
    threshold_position = total // 2
    median_pixel_index = numpy.argmax(cumHist >= threshold_position)
    return  median_pixel_index

# calculate minimum distance of two frame
def calculate_min_distance(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # 水平距離
    horizontal_distance = max(x1, x2) - min(w1, w2)
    # 垂直距離
    vertical_distance = max(y1, y2) - min(h1, h2)
    # 返回歐幾里得距離
    return (horizontal_distance ** 2 + vertical_distance ** 2) ** 0.5

# merge the closest rect
def merge_rects(rects, distance_threshold):
    merged_rects = []
    for rect in rects:
        # 如果還沒有合併的矩形，直接添加
        if not merged_rects:
            merged_rects.append(rect)
            continue

        # 計算距離，決定是否合併
        merge_with = -1
        for i, merged_rect in enumerate(merged_rects):
            if calculate_min_distance(rect, merged_rect) <= distance_threshold:
                merge_with = i
                break

        # 如果找到合併對象，合併矩形
        if merge_with >= 0:
            x1, y1, w1, h1 = merged_rects[merge_with]
            x2, y2, w2, h2 = rect

            new_x = min(x1, x2)
            new_y = min(y1, y2)
            new_w = max(x1 + w1, x2 + w2) - new_x
            new_h = max(y1 + h1, y2 + h2) - new_y

            merged_rects[merge_with] = (new_x, new_y, new_w, new_h)
        else:
            merged_rects.append(rect)

    return merged_rects

def location_pin(patches, patch_size):
    entList = [[0 for _ in range(patches.shape[1])] for _ in range(patches.shape[0])]
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i][j]
            patch_fft = numpy.fft.fft2(patch)
            patch_fft = numpy.fft.fftshift(patch_fft)
            fft_magnitude = numpy.abs(patch_fft)
            Ent = entropy(fft_magnitude.flatten(), base=2)
            entList[i][j] = Ent
    midEnt = numpy.median(entList)
    output = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            if entList[i][j] >= midEnt:
                pinList = []
                height = i * patch_size
                width = j * patch_size
                pinList.append(height)
                pinList.append(width)
                pinList.append(int(height + patch_size))
                pinList.append(int(width + patch_size))
                output.append(pinList)
    return output

def label(frame):
    # 先侵蝕，再膨脹
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                      (11,
                                       11))  # MORPH_RECT(函数返回矩形卷积核) MORPH_CROSS(函数返回十字形卷积核) MORPH_ELLIPSE(函数返回椭圆形卷积核)
    temp_img = cv.erode(frame, kernel)
    temp_img = cv.dilate(temp_img, kernel)

    pixel_threshold = cumHalf(temp_img)
    # 二質化
    ret, temp_img = cv.threshold(temp_img, pixel_threshold, 255, cv.THRESH_TRUNC)
    # 侵蝕使裂縫變寬
    kernel = numpy.ones((3, 3), numpy.uint8)
    temp_img = cv.erode(temp_img, kernel, iterations=1)
    # 自適應二質化
    temp_img = cv.adaptiveThreshold(temp_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 2)
    # 開運算
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    temp_img = cv.morphologyEx(temp_img, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(temp_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 儲存所有矩形的列表
    rectangles = []

    # 找到輪廓並添加到矩形列表
    for contour in contours:
        if cv.contourArea(contour) > 100:
            x, y, w, h = cv.boundingRect(contour)
            rectangles.append((x, y, x + w, y + h))
    merged_rects = merge_rects(rectangles, distance_threshold)
    return merged_rects

# main core of crack detection
def preprocess(captured):
    gray_img = cv.cvtColor(captured, cv.COLOR_BGR2GRAY)
    median_img = cv.medianBlur(gray_img, ksize=5)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,
                                      (3,
                                       3))  # MORPH_RECT(函数返回矩形卷积核) MORPH_CROSS(函数返回十字形卷积核) MORPH_ELLIPSE(函数返回椭圆形卷积核)
    erode_img = cv.erode(median_img, kernel)
    temp = erode_img.copy()
    patches = patchify(temp, (100, 100), step=100)
    pinned_location = location_pin(patches, patch_size=100) # use fft to calculate average gray entropy of whole image and
                                                            # do average filter on each patch
    crack_num = 0
    for L in pinned_location:
        frame = erode_img[L[0]:L[2] + 1, L[1]:L[3] + 1]
        merged_rects = label(frame)
        for i, (x1, y1, x2, y2) in enumerate(merged_rects):
            text = str(i + 1)
            cv.rectangle(captured, (x1 + L[1], y1 + L[0]), (x2 + L[1], y2 + L[0]), (0, 0, 255), 2)
            cv.putText(captured, text, (x1 + L[1], y1 + L[0]), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        crack_num += len(merged_rects)
    return captured, crack_num

# main execution
def capture_frame():
    image = cv2.imread("./Data/cutting_5_0.jpg")
    image = cv2.resize(image, (center_width, center_height))
    result = preprocess(image)
    print('裂縫數：%s' % str(result[1]))
    cv2.imshow('Frame', result[0])
    cv2.imwrite("./Result/labelResult_fft_cutting5.png",result[0])

if __name__ == "__main__":
    capture_frame()