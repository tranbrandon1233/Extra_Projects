
import cv2
import numpy as np

def measure_object(image_path, known_width=None):
    """
    Measure the largest object in an image.
    
    Args:
        image_path (str): Path to the input image
        known_width (float, optional): Known width of an object in the image (in inches/cm)
                                     for calibration
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image")
    
    # Get the original image dimensions
    original_height, original_width = image.shape[:2]
    
    # Calculate window size (max 800px width while maintaining aspect ratio)
    max_width = 800
    scale_ratio = max_width / original_width
    window_width = int(original_width * scale_ratio)
    window_height = int(original_height * scale_ratio)
    
    # Resize image to fit window
    image = cv2.resize(image, (window_width, window_height))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Perform edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No objects detected")
        return
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Calculate width and height in pixels
    width = rect[1][0]
    height = rect[1][1]
    
    # Draw the rectangle
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    
    # Calculate pixel to unit ratio if known_width is provided
    if known_width is not None:
        pixels_per_unit = width / known_width
        actual_height = height / pixels_per_unit
        print(f"Estimated object dimensions:")
        print(f"Width: {known_width:.2f} units (reference)")
        print(f"Height: {actual_height:.2f} units")
    else:
        print(f"Object dimensions in pixels:")
        print(f"Width: {width:.2f} pixels")
        print(f"Height: {height:.2f} pixels")
    
    # Add text to image
    cv2.putText(image, f"Width: {width:.1f}px", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Height: {height:.1f}px", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the image
    cv2.imshow("Object Measurement", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image_path = "img.jpg"

    known_width = None  # Example: 10 (inches/cm/etc)
    
    try:
        measure_object(image_path, known_width)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
