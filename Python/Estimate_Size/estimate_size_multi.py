import cv2
import numpy as np
from scipy.spatial.distance import euclidean

def resize_image(image, width=1024):
    """
    Resize image to fit within specified width while maintaining aspect ratio.
    
    Args:
        image: Input image
        width: Target width
    
    Returns:
        Resized image and scale factor
    """
    h, w = image.shape[:2]
    scale = width / float(w)
    
    # If image is smaller than target width, keep original size
    if scale > 1:
        return image, 1.0
        
    dimensions = (int(w * scale), int(h * scale))
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA), scale

def get_object_size(image_path, reference_width, debug=False, window_width=1024):
    """
    Estimate sizes of objects in an image using a reference object.
    
    Args:
        image_path: Path to the input image
        reference_width: Known width of reference object in millimeters
        debug: If True, shows intermediate processing steps
        window_width: Target width for display window
    
    Returns:
        List of detected objects with their estimated dimensions
    """
    # Read image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError("Could not read image from path")
        
    # Resize image for processing while maintaining aspect ratio
    image, scale_factor = resize_image(original_image, window_width)
    
    # Create processing copy
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return []
    
    # Sort contours from left to right
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    # Get reference object (leftmost object)
    reference_object = sorted_contours[0]
    ref_box = cv2.minAreaRect(reference_object)
    ref_pixels = max(ref_box[1])  # Get the longest dimension
    
    # Calculate pixels per metric (accounting for scale factor)
    pixels_per_metric = (ref_pixels / scale_factor) / reference_width
    
    objects = []
    
    # Process each contour
    for contour in sorted_contours[1:]:
        # Get rotated rectangle
        box = cv2.minAreaRect(contour)
        box_points = cv2.boxPoints(box)
        box_points = np.intp(box_points)
        
        # Calculate real-world dimensions (accounting for scale factor)
        width = (box[1][0] / scale_factor) / pixels_per_metric
        height = (box[1][1] / scale_factor) / pixels_per_metric
        
        # Store object information
        objects.append({
            'width_mm': round(width, 1),
            'height_mm': round(height, 1),
            'contour': box_points
        })
        
        if debug:
            # Draw rotated rectangle
            cv2.drawContours(image, [box_points], 0, (0, 255, 0), 2)
            
            # Add dimensions text
            center = np.mean(box_points, axis=0).astype(int)
            text = f"{width:.1f}mm x {height:.1f}mm"
            cv2.putText(image, text, tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.65, (255, 0, 0), 2)
    
    if debug:
        # Create window with proper sizing
        window_name = "Object Size Estimation"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Get screen resolution
        screen = cv2.getWindowImageRect(window_name)
        if screen is not None:
            screen_height = screen[3]
            # Calculate window height maintaining aspect ratio
            window_height = int(window_width * (image.shape[0] / image.shape[1]))
            # Ensure window fits on screen
            if window_height > screen_height:
                scale = screen_height / window_height
                window_width = int(window_width * scale)
                window_height = screen_height
            
            cv2.resizeWindow(window_name, window_width, window_height)
        
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return objects

# def calibrate_camera():
#     """
#     Optional: Calibrate camera to handle lens distortion.
#     Returns camera matrix and distortion coefficients.
#     """
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#     objp = np.zeros((6*7,3), np.float32)
#     objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    
#     objpoints = []  # 3d points in real world space
#     imgpoints = []  # 2d points in image plane
    
#     def process_calibration_image(img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
        
#         if ret:
#             objpoints.append(objp)
#             corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
#             imgpoints.append(corners2)
            
#         return ret, gray.shape[::-1]
    
#     return np.eye(3), np.zeros(5)

# Example usage
if __name__ == "__main__":
    # Example settings
    image_path = "img.jpg"
    reference_width_mm = 100  # Known width of reference object in millimeters
    window_width = 1024  # Default window width in pixels
    
    try:
        objects = get_object_size(
            image_path, 
            reference_width_mm, 
            debug=True,
            window_width=window_width
        )
        
        print("\nDetected Objects:")
        for i, obj in enumerate(objects, 1):
            print(f"Object {i}:")
            print(f"Width: {obj['width_mm']}mm")
            print(f"Height: {obj['height_mm']}mm")
            print("---")
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")