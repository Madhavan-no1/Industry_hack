import cv2
import numpy as np
from sklearn.cluster import KMeans

def load_and_preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image_rgb

def segment_image(image):
    # Assuming the image is divided into two halves vertically
    height, width = image.shape[:2]
    parent_strip = image[:, :width//2]
    test_strip = image[:, width//2:]
    
    return parent_strip, test_strip

def extract_dominant_colors(image, n_colors):
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    
    # Get the dominant colors
    dominant_colors = kmeans.cluster_centers_
    
    return dominant_colors.astype(int)

def find_nearest_color(test_color, parent_colors):
    # Calculate Euclidean distances
    distances = np.sqrt(np.sum((parent_colors - test_color)**2, axis=1))
    
    # Find the index of the minimum distance
    nearest_index = np.argmin(distances)
    
    return nearest_index, parent_colors[nearest_index]

def colorimetry_comparison(image_path, n_parent_colors=10):
    # Load and preprocess the image
    image = load_and_preprocess_image(image_path)
    
    # Segment the image
    parent_strip, test_strip = segment_image(image)
    
    # Extract dominant colors from parent strip
    parent_colors = extract_dominant_colors(parent_strip, n_parent_colors)
    
    # Extract the dominant color from the test strip
    test_color = extract_dominant_colors(test_strip, 1)[0]
    
    # Find the nearest color in the parent strip
    nearest_index, nearest_color = find_nearest_color(test_color, parent_colors)
    
    return nearest_index, nearest_color, test_color

# Example usage
image_path = r'C:\Users\lewis\Downloads\image_recognition_using_visionai-main\Screenshot_26-9-2024_234233_.jpeg'
result_index, result_color, test_color = colorimetry_comparison(image_path)

print(f"Test strip color: {test_color}")
print(f"Nearest parent strip color index: {result_index}")
print(f"Nearest parent strip color: {result_color}")