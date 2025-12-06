import cv2
import os

# Get the folder where THIS script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build path to:  ../data/test.jpeg
img_path = os.path.join(script_dir, "..", "data", "test.jpeg")
img_path = os.path.abspath(img_path)  # Normalize the path

print("Loading image from:", img_path)

# Load image
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not load image at: {img_path}")

# Image processing
blurred = cv2.GaussianBlur(img, (15, 15), 0)
edges = cv2.Canny(img, 100, 200)

# Display
cv2.imshow("Original", img)
cv2.imshow("Blurred", blurred)
cv2.imshow("Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
