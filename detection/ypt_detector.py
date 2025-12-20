import cv2
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Get the folder where this script resides
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
SAMPLES_DIR = os.path.join(BASE_DIR, "..", "data", "sample_screenshots")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Load templates
templates = []
for filename in os.listdir(TEMPLATES_DIR):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        template = cv2.imread(os.path.join(TEMPLATES_DIR, filename), cv2.IMREAD_GRAYSCALE)
        templates.append(template)

# Load sample screenshots
samples = []
for filename in os.listdir(SAMPLES_DIR):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        sample = cv2.imread(os.path.join(SAMPLES_DIR, filename), cv2.IMREAD_GRAYSCALE)
        samples.append(sample)

def match_template(image_gray, template, threshold=0.8):
    """Return True if template matches image above threshold"""
    # Resize template if it's bigger than the image
    if template.shape[0] > image_gray.shape[0] or template.shape[1] > image_gray.shape[1]:
        template = cv2.resize(template, (image_gray.shape[1], image_gray.shape[0]))
    
    res = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    return len(loc[0]) > 0


def compare_with_samples(image_gray, threshold=0.85):
    """Return True if image is similar to any sample using SSIM"""
    for sample in samples:
        resized_sample = cv2.resize(sample, (image_gray.shape[1], image_gray.shape[0]))
        score = ssim(image_gray, resized_sample)
        if score >= threshold:
            return True
    return False

def is_ypt_screenshot(image_path):
    """Return True if the image is detected as a YPT screenshot"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Template matching
    if any(match_template(gray, t) for t in templates):
        return True

    # Optional SSIM check
    if compare_with_samples(gray):
        return True

    return False

# Example test
if __name__ == "__main__":
    test_images = ["../data/sample_screenshots/ypt-21.jpeg"]  # Replace with paths to test images
    for img_path in test_images:
        if is_ypt_screenshot(img_path):
            print(f"{img_path} is a YPT screenshot ✅")
        else:
            print(f"{img_path} is NOT a YPT screenshot ❌")
