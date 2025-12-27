import cv2
import os
import numpy as np
import pytesseract
import re
from skimage.metrics import structural_similarity as ssim

# ---------------------------
# Paths
# ---------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "..", "data", "sample_screenshots")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# ---------------------------
# Load templates (edges)
# ---------------------------

templates = []
for filename in os.listdir(TEMPLATES_DIR):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img = cv2.imread(os.path.join(TEMPLATES_DIR, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            templates.append(cv2.Canny(img, 50, 150))

# ---------------------------
# Load sample screenshots (edges)
# ---------------------------

samples = []
for filename in os.listdir(SAMPLES_DIR):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img = cv2.imread(os.path.join(SAMPLES_DIR, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            samples.append(cv2.Canny(img, 50, 150))

# ---------------------------
# Hard rejection filters
# ---------------------------

def is_low_information(gray):
    if np.std(gray) < 10:
        return True

    edges = cv2.Canny(gray, 50, 150)
    return (np.count_nonzero(edges) / edges.size) < 0.01


def has_text_or_numbers(gray):
    text = pytesseract.image_to_string(gray, config="--psm 6").strip()
    return len(text) >= 3 and bool(re.search(r"\d|[A-Za-z]{3,}", text))

# ---------------------------
# YPT semantic signals
# ---------------------------

YPT_KEYWORDS = [
    "insights", "max focus", "started", "finished",
    "period", "trend", "today", "yesterday"
]

def semantic_text_score(gray):
    text = pytesseract.image_to_string(gray, config="--psm 6").lower()
    return sum(1 for kw in YPT_KEYWORDS if kw in text)


def time_pattern_score(gray):
    text = pytesseract.image_to_string(gray, config="--psm 6")
    has_hms = bool(re.search(r"\d{1,2}:\d{2}:\d{2}", text))
    has_ampm = bool(re.search(r"\b(am|pm)\b", text, re.I))
    return 2 if has_hms and has_ampm else 0

# ---------------------------
# Donut chart detection
# ---------------------------

def donut_chart_score(gray):
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=40,
        maxRadius=200
    )

    if circles is None:
        return 0

    text = pytesseract.image_to_string(gray, config="--psm 6")
    return 3 if "%" in text else 1

# ---------------------------
# Color fingerprint
# ---------------------------

def color_fingerprint_score(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))

    red_ratio = np.count_nonzero(red_mask) / red_mask.size
    green_ratio = np.count_nonzero(green_mask) / green_mask.size

    return 2 if red_ratio > 0.01 and green_ratio > 0.01 else 0

# ---------------------------
# Similarity helpers
# ---------------------------

def match_template(image_edges, template_edges, threshold=0.6):
    template_edges = cv2.resize(template_edges, image_edges.shape[::-1])
    res = cv2.matchTemplate(image_edges, template_edges, cv2.TM_CCOEFF_NORMED)
    return np.any(res >= threshold)


def compare_with_samples(image_edges, threshold=0.75):
    for sample in samples:
        resized = cv2.resize(sample, image_edges.shape[::-1])
        if ssim(image_edges, resized) >= threshold:
            return True
    return False

# ---------------------------
# Main detector (score-based)
# ---------------------------

def is_ypt_screenshot(image_path, score_threshold=7):
    img = cv2.imread(image_path)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if is_low_information(gray) or not has_text_or_numbers(gray):
        return False

    score = 0
    score += semantic_text_score(gray)
    score += time_pattern_score(gray)
    score += donut_chart_score(gray)
    score += color_fingerprint_score(img)

    edges = cv2.Canny(gray, 50, 150)

    if any(match_template(edges, t) for t in templates):
        score += 1

    if compare_with_samples(edges):
        score += 1

    return score >= score_threshold

# ---------------------------
# Manual test
# ---------------------------

if __name__ == "__main__":
    tests = ["../data/sample_screenshots/ypt-21.jpeg"]

    for img in tests:
        print(img, "→", "YPT ✅" if is_ypt_screenshot(img) else "NOT YPT ❌")
