import os
import cv2

# Dataset location
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Class names (A–Y, Space, Backspace) → 27 total
class_names = [
    "A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q",
    "R","S","T","U","V","W","X","Y","Space","Backspace"
]

dataset_size = 1000  # images per class

cap = cv2.VideoCapture(0)

for class_name in class_names:
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # Count existing images (resume support)
    existing_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
    counter = len(existing_files)

    if counter >= dataset_size:
        print(f"Skipping {class_name} (already has {counter} images)")
        continue

    print(f'Collecting data for class {class_name}')

    # Wait for user to press Q to start
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # ✅ Do NOT mirror the frame → this is real camera view
        cv2.putText(frame, f'Ready for {class_name}? Press "Q"', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Start capturing
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue

        # ✅ Still non-mirrored
        cv2.putText(frame, f'{class_name}: {counter}/{dataset_size}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        cv2.waitKey(1)

        filename = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(filename, frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
