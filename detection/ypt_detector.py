import cv2
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Get the folder where this script resides
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
SAMPLES_DIR = os.path.join(BASE_DIR, "..", "data", "sample_screenshots")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Load templates (grayscale)
templates = []
for filename in os.listdir(TEMPLATES_DIR):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        template_path = os.path.join(TEMPLATES_DIR, filename)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        # Precompute edges
        template_edges = cv2.Canny(template, 50, 150)
        templates.append(template_edges)

# Load sample screenshots (grayscale)
samples = []
for filename in os.listdir(SAMPLES_DIR):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        sample_path = os.path.join(SAMPLES_DIR, filename)
        sample = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
        # Precompute edges
        sample_edges = cv2.Canny(sample, 50, 150)
        samples.append(sample_edges)

def match_template(image_edges, template_edges, threshold=0.6):
    """Return True if template matches image edges above threshold"""
    # Resize template if bigger than image
    if template_edges.shape[0] > image_edges.shape[0] or template_edges.shape[1] > image_edges.shape[1]:
        template_edges = cv2.resize(template_edges, (image_edges.shape[1], image_edges.shape[0]))
    
    res = cv2.matchTemplate(image_edges, template_edges, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    return len(loc[0]) > 0

def compare_with_samples(image_edges, threshold=0.75):
    """Return True if image edges are similar to any sample edges using SSIM"""
    for sample_edges in samples:
        resized_sample = cv2.resize(sample_edges, (image_edges.shape[1], image_edges.shape[0]))
        score = ssim(image_edges, resized_sample)
        if score >= threshold:
            return True
    return False

def is_ypt_screenshot(image_path):
    """Return True if the image is detected as a YPT screenshot"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Template matching with multi-template voting
    matches = sum(match_template(edges, t) for t in templates)
    if matches >= 1:  # At least 1 template matches
        return True

    # SSIM fallback on edges
    if compare_with_samples(edges):
        return True

    return False

# Example test
if __name__ == "__main__":
    test_images = ["../data/sample_screenshots/ypt-21.jpeg"]  # Replace with test paths
    for img_path in test_images:
        if is_ypt_screenshot(img_path):
            print(f"{img_path} is a YPT screenshot ✅")
        else:
            print(f"{img_path} is NOT a YPT screenshot ❌")
