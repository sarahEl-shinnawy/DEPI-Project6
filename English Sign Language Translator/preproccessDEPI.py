# preprocess.py — augmented, safe, and documented
import os
import cv2
import mediapipe as mp
import pickle
import random
import numpy as np
from collections import Counter
import json
import matplotlib.pyplot as plt

# -------------------- CONFIG --------------------
DATA_DIR = './data'                   # input: folders per class (A/, B/, ...)
OUTPUT_PKL = 'data_augmented.pickle'  # output
META_JSON = 'dataset_meta.json'
IMG_SIZE = (256, 256)                 # resize images to this before augmentation
AUG_PER_IMAGE = 2                     # how many augmentations PER original image (0 => only original)
TARGET_SAMPLES_PER_CLASS = None       # e.g. 1000 to force balance; None to not force
APPLY_SKIN_MASK = False               # simple color-based mask (try True if many failures)
ALLOW_HORIZONTAL_FLIP = False         # be careful: flips change handedness
RANDOM_SEED = 42
# ------------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.35)

def apply_skin_mask(img_bgr):
    """Simple HSV skin-color mask — optional, adjust thresholds as needed."""
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Typical skin color ranges — may need tuning per lighting / camera
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([25, 200, 255], dtype=np.uint8)
    mask = cv2.inRange(img_hsv, lower, upper)
    # morphological clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    return result

def random_augment(img):
    """Return an augmented image (rotation, scale, brightness)."""
    h, w = img.shape[:2]

    # rotation
    angle = random.uniform(-15, 15)
    # scale
    scale = random.uniform(0.95, 1.05)

    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    aug = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # brightness
    factor = random.uniform(0.85, 1.15)
    aug = np.clip(aug * factor, 0, 255).astype(np.uint8)

    # horizontal flip occasionally (only if allowed)
    if ALLOW_HORIZONTAL_FLIP and random.random() < 0.5:
        aug = cv2.flip(aug, 1)

    return aug

def extract_landmarks_from_image(img_bgr):
    """Runs Mediapipe and returns normalized landmark vector or None if detection fails."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None
    # if multiple hands are detected, we take the first one (you can change this)
    hand_landmarks = results.multi_hand_landmarks[0]
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    data_aux = []
    min_x, min_y = min(x_coords), min(y_coords)
    for lm in hand_landmarks.landmark:
        data_aux.append(lm.x - min_x)
        data_aux.append(lm.y - min_y)
    return data_aux

# ---------------- main preprocessing ----------------
all_data = []
all_labels = []
class_counts_before = {}
class_counts_after = {}

class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print("Classes found:", class_names)

# First, count files per class (raw)
raw_counts = {c: len([f for f in os.listdir(os.path.join(DATA_DIR, c)) if f.lower().endswith(('.jpg','.png'))]) for c in class_names}
print("Raw image counts per class:", raw_counts)

for class_name in class_names:
    class_dir = os.path.join(DATA_DIR, class_name)
    image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg','.png'))]
    class_counts_before[class_name] = len(image_files)
    print(f"\nProcessing class {class_name} ({len(image_files)} images)...")
    extracted_this_class = 0

    # Prepare list to potentially balance later
    processed_samples_for_class = []

    for img_fn in image_files:
        img_path = os.path.join(class_dir, img_fn)
        img = cv2.imread(img_path)
        if img is None:
            continue
        # resize to fixed size
        img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)

        # optionally apply skin mask
        candidates = [img_resized]
        if AUG_PER_IMAGE > 0:
            for _ in range(AUG_PER_IMAGE):
                candidates.append(random_augment(img_resized))

        for cand in candidates:
            if APPLY_SKIN_MASK:
                cand_proc = apply_skin_mask(cand)
            else:
                cand_proc = cand

            lm = extract_landmarks_from_image(cand_proc)
            if lm is not None:
                processed_samples_for_class.append(lm)

    # If target balancing requested, augment processed samples until target reached
    if TARGET_SAMPLES_PER_CLASS is not None:
        needed = max(0, TARGET_SAMPLES_PER_CLASS - len(processed_samples_for_class))
        print(f"  -> {len(processed_samples_for_class)} valid samples, need {needed} more to reach target {TARGET_SAMPLES_PER_CLASS}")
        # naive augmentation loop: re-augment existing originals (may produce duplicates)
        original_files = [os.path.join(class_dir, f) for f in image_files]
        idx = 0
        while needed > 0 and original_files:
            img = cv2.imread(original_files[idx % len(original_files)])
            img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
            aug_img = random_augment(img_resized)
            if APPLY_SKIN_MASK:
                aug_img = apply_skin_mask(aug_img)
            lm = extract_landmarks_from_image(aug_img)
            if lm is not None:
                processed_samples_for_class.append(lm)
                needed -= 1
            idx += 1

    # Add processed samples to global arrays
    all_data.extend(processed_samples_for_class)
    all_labels.extend([class_name]*len(processed_samples_for_class))
    class_counts_after[class_name] = len(processed_samples_for_class)
    print(f"  -> kept {len(processed_samples_for_class)} samples for class {class_name}")

# Save dataset and metadata
with open(OUTPUT_PKL, 'wb') as f:
    pickle.dump({'data': all_data, 'labels': all_labels}, f)

meta = {
    'raw_counts': raw_counts,
    'after_counts': class_counts_after,
    'img_size': IMG_SIZE,
    'aug_per_image': AUG_PER_IMAGE,
    'target_per_class': TARGET_SAMPLES_PER_CLASS,
    'apply_skin_mask': APPLY_SKIN_MASK,
    'allow_horizontal_flip': ALLOW_HORIZONTAL_FLIP,
    'random_seed': RANDOM_SEED,
    'classes': class_names
}
with open(META_JSON, 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\nPreprocessing complete. {len(all_data)} total samples saved to {OUTPUT_PKL}")
print("Per-class counts (after preprocessing):")
for k,v in class_counts_after.items():
    print(f"  {k}: {v}")

# Simple bar chart
plt.figure(figsize=(12,5))
plt.bar(list(class_counts_after.keys()), list(class_counts_after.values()))
plt.xticks(rotation=45)
plt.xlabel('Class')
plt.ylabel('Samples (after preprocessing)')
plt.title('Samples per class after preprocessing')
plt.tight_layout()
plt.savefig('class_counts_after.png')
print("Saved class distribution plot to class_counts_after.png")
