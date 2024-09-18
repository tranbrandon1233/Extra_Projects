import cv2
import pytesseract
import os

def extract_mileage(image_path):
    """
    Extracts the 6-digit mileage from a car dashboard image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: The extracted mileage as a 6-digit string, or None if not found.
    """

    # Load the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to handle varying lighting
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours in the image
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours
    for c in cnts:
        # Approximate the contour to a polygon
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Consider contours with 4 sides (potential rectangular regions)
        if len(approx) == 4:
            # Calculate the bounding box of the contour
            x, y, w, h = cv2.boundingRect(approx)

            # Define a reasonable aspect ratio range for mileage digits
            aspect_ratio = w / float(h)
            if 0.5 <= aspect_ratio <= 1.5:
                # Extract the ROI (Region of Interest)
                roi = thresh[y:y + h, x:x + w]

                # Use pytesseract to perform OCR on the ROI
                text = pytesseract.image_to_string(roi, config='--psm 6 digits')

                # Check if the extracted text is a 6-digit number
                if len(text.strip()) == 6 and text.strip().isdigit():
                    return text.strip()

    # If no suitable mileage found, return None
    return None

# Get the directory path from the user
image_dir = input("Enter the directory containing the mileage images: ")

# Iterate through all JPEG files in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        image_path = os.path.join(image_dir, filename)
        mileage = extract_mileage(image_path)
        if mileage:
            print(f"Mileage in {filename}: {mileage}")
        else:
            print(f"Mileage not found in {filename}")