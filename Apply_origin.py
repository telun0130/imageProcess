import cv2
import cv2 as cv
import numpy

# 距離閾值
distance_threshold = 12
center_width = 400
center_height = 800

def calculate_min_distance(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # 水平距離
    horizontal_distance = max(x1, x2) - min(w1, w2)
    # print(horizontal_distance)
    # 垂直距離
    vertical_distance = max(y1, y2) - min(h1, h2)
    # print(vertical_distance)

    # 返回歐幾里得距離
    # print((horizontal_distance**2 + vertical_distance**2)**0.5)
    return (horizontal_distance ** 2 + vertical_distance ** 2) ** 0.5
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
def preprocess(img):
    image_blur = cv.GaussianBlur(img, (5, 5), 2)
    gray_img2 = cv.cvtColor(image_blur, cv.COLOR_BGR2GRAY)

    threshold = 60
    img_transformed = numpy.where(gray_img2 < threshold, gray_img2, 60)
    img_transformed = numpy.clip(img_transformed, 0, 255)
    img_transformed = img_transformed.astype(numpy.uint8)

    # 做侵蝕
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    img_erode = cv.erode(img_transformed, kernel)

    # 做膨脹
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    img_dilate = cv.dilate(img_erode, kernel)

    # 二值法
    binaryimg = cv.adaptiveThreshold(img_dilate, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

    dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    img_dilate = cv.dilate(binaryimg, dilate_kernel)

    erode_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    img_erode = cv.erode(img_dilate, erode_kernel, iterations=2)
    dilate_kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    img_dilate2 = cv.dilate(img_erode, dilate_kernel2, iterations=7)

    contours, _ = cv.findContours(img_dilate2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 儲存所有矩形的列表
    rectangles = []

    # 找到輪廓並添加到矩形列表
    for contour in contours:
        if cv.contourArea(contour) > 100:
            x, y, w, h = cv.boundingRect(contour)
            rectangles.append((x, y, x + w, y + h))
    merged_rects = merge_rects(rectangles, distance_threshold)

    for i, (x1, y1, x2, y2) in enumerate(merged_rects):
        text = str(i + 1)
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv.putText(img, text, (x1, y1), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
    crack_num = len(merged_rects)
    return img, crack_num
def capture_frame():
    image = cv2.imread("./Data/cutting_5_0.jpg")
    image = cv2.resize(image, (center_width, center_height))
    result = preprocess(image)
    print('裂縫數：%s' % str(result[1]))
    cv2.imshow('Frame', result[0])
    cv2.imwrite("./Result/labelResult_cutting5.png", result[0])


if __name__ == "__main__":
    capture_frame()