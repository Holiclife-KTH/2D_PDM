import numpy as np
import cv2

# Load the depth image
depth_image = np.load(
    "/home/irol/workspace/2D_PDM/src/output/pen_1/scene/depth_dis_map/depthmap_001.npy"
)

print(f"Depth image shape: {depth_image.shape}")
print(f"Depth range: {depth_image.min():.4f} to {depth_image.max():.4f}")
print(f"Depth dtype: {depth_image.dtype}")

print(depth_image)


# Normalize depth to 0-255 for visualization
depth_normalized = (depth_image - depth_image.min()) / (
    depth_image.max() - depth_image.min()
)
depth_visual = (depth_normalized * 255).astype(np.uint8)

# Apply colormap for better visualization
depth_colored = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)

# Display
cv2.imshow("Depth Image (Grayscale)", depth_visual)
cv2.imshow("Depth Image (Colored)", depth_colored)
print("Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()


# import cv2
# import numpy as np
# img = cv2.imread('src/output/pen_1/target/seg/1.png', cv2.IMREAD_UNCHANGED)
# print('Image shape:', img.shape)
# print('Image dtype:', img.dtype)
# print('Unique values:', np.unique(img))
# print('First 5x5 pixels:')
# print(img[0:5, 0:5])
