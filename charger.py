#!/usr/bin/python3

import cv2
import numpy as np
import requests
import time
import os
import logging
import configparser
import datetime


def archive_current_log(log_file="charger.log"):
    if os.path.exists(log_file):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        archived_log = f"{log_file}_{timestamp}.old"
        os.rename(log_file, archived_log)


def setup_logger(log_file="charger.log", log_level="INFO"):
    archive_current_log(log_file)
    logger = logging.getLogger()
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S,%f')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def fetch_and_analyze_image(image_url_str, min_area, max_area, brightness_threshold, charging_lower_bound, charging_upper_bound, aspect_ratio_range, morph_kernel_height,
                            morph_kernel_width):

    try:
        response = requests.get(image_url_str)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Error fetching image: {e}")
        return

    img_np_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_np_array, cv2.IMREAD_COLOR)

    height, width = img.shape[:2]
    crop_width = int(0.05 * width)
    img = img[:, :-crop_width]
    logger.debug(f"The image has size x={width} y={height}")

    # Increase the contrast of the image
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    avg_brightness = round(np.mean(gray), 2)
    logger.debug(f"Garage Average Brightness: {avg_brightness}")

    if avg_brightness < brightness_threshold:
        cv2.imwrite("closed.jpg", gray)
        if charging_lower_bound <= avg_brightness < charging_upper_bound:
            logger.info(f"Garage Door Closed: State is Charging:")
        else:
            logger.info(f"Garage Door Closed: State is Not Charging:")
    else:
        logger.info("Garage Door Open:")

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_kernel_width, morph_kernel_height))
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite("thresh_and_morph.jpg", closing)

        contours, hierarchy = cv2.findContours(
            closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h
            logger.debug(
                f"Contour found at ({x}, {y}) with area: {area} and aspect ratio: {aspect_ratio}")

            # Add vertical position check (assuming y=0 is the top)
            vertical_threshold = int(
                0.50 * img.shape[0])  # adjust as necessary

            if (min_area <= area <= max_area and
                aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1] and
                    y > vertical_threshold):  # Only consider contours below this threshold
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                logger.info("Possible charger detected!")

        closing_resized = cv2.resize(
            closing, (closing.shape[1] * 2, closing.shape[0] * 2))
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))

        cv2.imwrite("contours.jpg", closing_resized)
        cv2.imwrite("final_image.jpg", img)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    min_area = int(config['DEFAULT']['min_area'])
    max_area = int(config['DEFAULT']['max_area'])
    brightness_threshold = float(config['DEFAULT']['brightness_threshold'])
    charging_lower_bound = float(config['DEFAULT']['charging_lower_bound'])
    charging_upper_bound = float(config['DEFAULT']['charging_upper_bound'])
    image_interval = int(config['DEFAULT']['image_interval'])
    aspect_ratio_range = tuple(map(float, config['DEFAULT'].get(
        'aspect_ratio_range', '2.2,2.6').split(',')))
    log_level_str = config['DEFAULT'].get('log_level', 'INFO')
    morph_kernel_width = int(config['DEFAULT']['morph_kernel_width'])
    morph_kernel_height = int(config['DEFAULT']['morph_kernel_height'])
    image_url_str = config['DEFAULT'].get('image_url')

    logger = setup_logger(log_level=log_level_str)

    while True:
        fetch_and_analyze_image(image_url_str, min_area, max_area, brightness_threshold,
                                charging_lower_bound, charging_upper_bound, aspect_ratio_range,
                                morph_kernel_width, morph_kernel_height)
        time.sleep(image_interval)
