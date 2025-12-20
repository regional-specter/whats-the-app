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
        path = os.path.join(TEMPLATES_DIR, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        edges = cv2.Canny(img, 50, 150)
        templates.append(edges)

# ---------------------------
# Load sample screenshots (edges)
# ---------------------------

samples = []
for filename in os.listdir(SAMPLES_DIR):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(SAMPLES_DIR, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        edges = cv2.Canny(img, 50, 150)
        samples.append(edges)

# ---------------------------
# Hard rejection filters
# ---------------------------

def is_low_information(gray):
    """
    Reject blank / near-blank images.
    """
    # Pixel variance (solid colors fail here)
    if np.std(gray) < 10:
        return True

    # Edge density (UI must have structure)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.count_nonzero(edges) / edges.size

    return edge_ratio < 0.01


def has_text_or_numbers(gray):
    """
    Require visible UI text or numbers.
    """
    text = pytesseract.image_to_string(gray, config="--psm 6").strip()

    if len(text) < 3:
        return False

    has_number = bool(re.search(r"\d", text))
    has_word = bool(re.search(r"[A-Za-z]{3,}", text))

    return has_number or has_word

# ---------------------------
# Similarity helpers
# ---------------------------

def match_template(image_edges, template_edges, threshold=0.6):
    """
    Template match using normalized correlation.
    """
    if template_edges.shape[0] > image_edges.shape[0] or template_edges.shape[1] > image_edges.shape[1]:
        template_edges = cv2.resize(
            template_edges,
            (image_edges.shape[1], image_edges.shape[0])
        )

    res = cv2.matchTemplate(image_edges, template_edges, cv2.TM_CCOEFF_NORMED)
    return np.any(res >= threshold)


def compare_with_samples(image_edges, threshold=0.75):
    """
    Structural similarity check against known YPT samples.
    """
    for sample_edges in samples:
        resized = cv2.resize(sample_edges, (image_edges.shape[1], image_edges.shape[0]))
        score = ssim(image_edges, resized)
        if score >= threshold:
            return True
    return False

# ---------------------------
# Main detector
# ---------------------------

def is_ypt_screenshot(image_path):
    """
    Strict YPT screenshot detector.
    Returns True ONLY if UI + YPT similarity are present.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🚨 HARD GATES (non-negotiable)
    if is_low_information(gray):
        return False

    if not has_text_or_numbers(gray):
        return False

    # ---- YPT-specific checks ----
    edges = cv2.Canny(gray, 50, 150)

    # Template voting
    template_hits = sum(match_template(edges, t) for t in templates)
    if template_hits >= 1:
        return True

    # SSIM fallback
    if compare_with_samples(edges):
        return True

    return False

# ---------------------------
# Manual test
# ---------------------------

if __name__ == "__main__":
    test_images = [
        "../data/sample_screenshots/ypt-21.jpeg",
    ]

    for img_path in test_images:
        result = is_ypt_screenshot(img_path)
        print(f"{img_path} → {'YPT ✅' if result else 'NOT YPT ❌'}")
