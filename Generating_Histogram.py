from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image_path_1 = "/Users/snehaagrawal/Documents/SEM 3/Advance ML/Assignments/1/Task-1/First_Image.png"
image_path_2 = "/Users/snehaagrawal/Documents/SEM 3/Advance ML/Assignments/1/Task-1/Second_Image.png"

# Load and convert images to grayscale
image1 = Image.open(image_path_1).convert("L")
image2 = Image.open(image_path_2).convert("L")

# Image to histogram
histogram1 = image1.histogram()
histogram2 = image2.histogram()

# Normalize histograms
hist1_normalized = np.array(histogram1) / np.sum(histogram1)
hist2_normalized = np.array(histogram2) / np.sum(histogram2)

# Plot histograms
plt.figure(figsize=(12, 6))

# Histogram for the first image
plt.subplot(1, 2, 1)
plt.title("Histogram of First Image")
plt.bar(range(256), histogram1, color='black')
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

# Histogram for the second image
plt.subplot(1, 2, 2)
plt.title("Histogram of Second Image")
plt.bar(range(256), histogram2, color='black')
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

# Adjust layout
plt.tight_layout()

# Save
save_path = "/Users/snehaagrawal/Documents/SEM 3/Advance ML/Assignments/1/Task-1/Histograms.png"
plt.savefig(save_path)

# Show
plt.show()

# Save histograms to binary files for better performance
np.save("/Users/snehaagrawal/Documents/SEM 3/Advance ML/Assignments/1/Task-1/histogram1.npy", histogram1)
np.save("/Users/snehaagrawal/Documents/SEM 3/Advance ML/Assignments/1/Task-1/histogram2.npy", histogram2)




