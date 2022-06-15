import numpy as np
import cv2 as cv2
import math
from flask import jsonify
from skimage import exposure
from statistics import mean
import os

minimum_commutative_image_diff = 1
IMAGE_SIZE_X = 600
IMAGE_SIZE_Y = 450


def get_gray(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_low = np.array([100, 105, 50])
    red_high = np.array([255, 135, 255])
    curr_mask = cv2.inRange(hsv_img, red_low, red_high)
    hsv_img[curr_mask > 0] = ([255, 50, 50])
    RGB_again = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    gray = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)
    return gray


def get_threshold(image):
    gray = get_gray(image)
    ret, threshold = cv2.threshold(gray, 120, 255, 0)
    return threshold


def get__biggest_contour(threshold):
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest_contour = max(contours, key=lambda x: x.size)
    return biggest_contour


def compare_image(image_1, image_2):
    commutative_image_diff = get_image_difference(image_1, image_2)
    if commutative_image_diff < minimum_commutative_image_diff:
        return commutative_image_diff
    return 10000


def get_image_difference(image_1, image_2):
    first_image_hist = cv2.calcHist([image_1], [0], None, [256], [0, 256])
    second_image_hist = cv2.calcHist([image_2], [0], None, [256], [0, 256])

    img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)
    img_template_probability_match = cv2.matchTemplate(first_image_hist, second_image_hist, cv2.TM_CCOEFF_NORMED)[0][0]
    img_template_diff = 1 - img_template_probability_match

    # проверяем 10%
    commutative_image_diff = (img_hist_diff / 10) + img_template_diff
    return commutative_image_diff


def get_asymmentry(main_image):
    image = main_image
    differences = []
    for i in range(1, 35):
        h, w = image.shape
        ch, cw = h // 2, w // 2
        left = image[:, :cw]
        right = image[:, cw:]
        center = (w / 2, h / 2)

        rows, cols = right.shape
        # матрица для поворота по У на 180
        M = np.float32([[-1, 0, cols],
                        [0, 1, 0],
                        [0, 0, 1]])
        # ЗЕРКАЛИТЬ ПРАВОЕ ИЗОБРАЖЕНИЕ
        reflected_right = cv2.warpPerspective(right, M, (int(cols), int(rows)))

        image_1 = left
        image_2 = reflected_right
        image_difference = compare_image(image_1, image_2)
        differences.append(image_difference)
        M = cv2.getRotationMatrix2D(center, 10 * i, 1.0)
        image = cv2.warpAffine(main_image, M, (w, h), borderValue=(255, 255, 255))

    diff = mean(differences)
    return diff


def get_uneven_borders(biggest_contour, h, w):
    area = cv2.contourArea(biggest_contour)
    perimeter = cv2.arcLength(biggest_contour, True)
    border_irregularity = (perimeter * perimeter) / (2 * math.pi * area) / (h * w / 1000)
    return border_irregularity


def get_color_scattered(img):
    gamma_corrected = exposure.adjust_gamma(img, 2)

    y = len(gamma_corrected)
    arr = []
    for i in range(y):
        for j in gamma_corrected[i]:
            if (j > 60):
                arr.append(j)

    arr.sort()
    size = len(arr)
    borders_count = math.ceil(size / 4)
    bottom = arr[:borders_count]
    top = arr[size - borders_count:]

    mean_b = np.mean(bottom)
    mean_t = np.mean(top)
    brightness_difference = (mean_t / (mean_b + mean_t))
    #     т.к. идеал это 50% отнимаем и смотрим по модулю, если 0 это полностью однотонное
    result = 0.5 - brightness_difference
    return abs(result)


def get_filled_percent(threshold):
    h, w = threshold.shape
    count_black = np.sum(threshold == 0)
    filled_percent = count_black / (w * h)
    return filled_percent


def getMetrics(filename):
    print('start detection')
    image = cv2.imread(filename)

    threshold = get_threshold(image)
    biggest_contour = get__biggest_contour(threshold)

    x, y, w, h = cv2.boundingRect(biggest_contour)
    crop_img = image[y:y + h, x:x + w]

    gray_crop = get_gray(crop_img)

    threshold_crop = get_threshold(crop_img)

    he, we = threshold.shape
    asymmetry = get_asymmentry(threshold_crop)
    uneven_borders = get_uneven_borders(biggest_contour, he, we)
    color_scattered = get_color_scattered(gray_crop)
    filled_percent = get_filled_percent(threshold)
    print(f'asymmetry {asymmetry}, uneven_borders {uneven_borders}, color_scattered {color_scattered}, '
          f'filled_percent {filled_percent}')
    os.remove(filename)

    return jsonify(
        asymmetry=asymmetry,
        unevenBorders=uneven_borders,
        colorScattered=color_scattered,
        filledPercent=filled_percent
    )

