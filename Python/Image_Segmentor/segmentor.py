import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_rocks(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Reshape the image into a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Define criteria and apply K-means clustering
    # Criteria: (type, max_iter, epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 3  # Number of clusters (may need to adjust this based on the image)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert to integers and reshape to the original image shape
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    # Create a mask for one of the clusters (e.g., rocks)
    # Adjust the cluster index based on the segmentation result
    labels = labels.reshape(image.shape[:2])
    cluster_index = 1  # Assume rocks are in cluster 1 (adjust as necessary)
    mask = (labels == cluster_index).astype(np.uint8) * 255

    # Apply the mask to extract the rocks
    rocks = cv2.bitwise_and(image, image, mask=mask)

    # Display the results
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Segmented Image")
    plt.imshow(segmented_image)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Rocks")
    plt.imshow(rocks)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Example usage
image_path = "image.png"  # Replace with your image path
segment_rocks(image_path)
