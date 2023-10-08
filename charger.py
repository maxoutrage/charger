#!/usr/bin/python3
#
import cv2
import numpy as np
import requests
import time
import os
import logging
import configparser
import datetime


def archive_current_log(log_file="charger.log"):
    # Check if the log_file already exists
    if os.path.exists(log_file):
        # Rename the current log file with the current timestamp
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


def fetch_and_analyze_image(min_area, max_area, brightness_threshold, charging_lower_bound, charging_upper_bound):

    response = requests.get("http://192.168.200.22:8080/?action=snapshot")
    response.raise_for_status()
    img_np_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_np_array, cv2.IMREAD_COLOR)

    height, width = img.shape[:2]
    crop_width = int(0.05 * width)
    img = img[:, :-crop_width]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    avg_brightness = round(np.mean(blurred), 2)
    logger.info(f"Garage Average Brightness: {avg_brightness}")

    if avg_brightness < brightness_threshold:
        logger.info(f"Garage door is closed")
        logger.info(
            f"Debug: avg_brightness={avg_brightness}, charging_lower_bound={charging_lower_bound}, charging_upper_bound={charging_upper_bound}")
        if charging_lower_bound <= avg_brightness < charging_upper_bound:
            logger.info(f"Garage Door Closed: State is Charging:")
        else:
            logger.info(f"Garage Door is Closed: State is Not Charging:")
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
        cv2.imwrite("closed_image.jpg", img)
        logger.info("Garage Closed image saved:")
        return

    logger.info("Garage Door is Open, Searching for contours:")
    edged = cv2.Canny(blurred, 100, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 60))
    dilated = cv2.dilate(edged, kernel, iterations=2)
    closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(
        closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    logger.info(f"Found {len(contours)} relevant contours")

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        logger.info(f"Contour Area: {contour_area}")
        if min_area < contour_area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            logger.debug(f"Aspect Ratio: {aspect_ratio}")
            if 1 < aspect_ratio < 10:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if img[y:y + h, x:x + w].mean() > 200:
                    logger.info("Charging!")
                else:
                    logger.info("Not Charging!")
            else:
                logger.info("Charger not found!")
        else:
            logger.info(
                "Contour area does not fall within expected range for the charger.")

    # Resize and then save images for analysis
    edged = cv2.resize(closing, (closing.shape[1] * 2, closing.shape[0] * 2))
    img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
    cv2.imwrite("contours.jpg", edged)
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
    log_level_str = config['DEFAULT'].get('log_level', 'INFO')
    logger = setup_logger(log_level=log_level_str)

    while True:
        fetch_and_analyze_image(
            min_area, max_area, brightness_threshold, charging_lower_bound, charging_upper_bound)
        time.sleep(image_interval)
