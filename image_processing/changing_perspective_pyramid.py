
"""Cambio de la perspectiva de una pirámide para que el plano focal sea paralelo a su cara frontal."""

#%% Importar las librerías necesarias
import cv2
import numpy as np
import matplotlib.pyplot as plt

#%% Función para cambiar la perspectiva de la pirámide
def correct_pyramid_perspective(image_path, output_path=None):
    """
    Transform the perspective of a pyramid image to make it appear as if
    the focal plane is parallel to the pyramid's front face.
    """

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image. Check the file path.")

    # Convert BGR to RGB for matplotlib display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get image dimensions
    height, width = img.shape[:2]

    # Define source points (corners of the pyramid base/area to be corrected)
    # These coordinates need to be adjusted based on your specific image
    # Order: top-left, top-right, bottom-right, bottom-left
    src_points = np.array([
        [width * 0.35, height * 0.45],  # Top-left of pyramid base
        [width * 0.65, height * 0.45],  # Top-right of pyramid base
        [width * 0.75, height * 0.85],  # Bottom-right of pyramid base
        [width * 0.25, height * 0.85]   # Bottom-left of pyramid base
    ], dtype=np.float32)

    # Define destination points (rectangular perspective)
    # This creates a rectangular view
    dst_width = int(width * 0.6)
    dst_height = int(height * 0.6)
    dst_points = np.array([
        [width//2 - dst_width//2, height//2 - dst_height//2],  # Top-left
        [width//2 + dst_width//2, height//2 - dst_height//2],  # Top-right
        [width//2 + dst_width//2, height//2 + dst_height//2],  # Bottom-right
        [width//2 - dst_width//2, height//2 + dst_height//2]   # Bottom-left
    ], dtype=np.float32)

    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transformation
    corrected_img = cv2.warpPerspective(img_rgb, matrix, (width, height))

    # Display the results
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Original image with source points marked
    axes[0].imshow(img_rgb)
    axes[0].plot([src_points[0][0], src_points[1][0], src_points[2][0], src_points[3][0], src_points[0][0]],
                 [src_points[0][1], src_points[1][1], src_points[2][1], src_points[3][1], src_points[0][1]],
                 'r-', linewidth=2, label='Source area')
    axes[0].scatter(src_points[:, 0], src_points[:, 1], c='red', s=50, zorder=5)
    axes[0].set_title('Original Image with Source Points')
    axes[0].legend()
    axes[0].axis('off')

    # Corrected image with destination points marked
    axes[1].imshow(corrected_img)
    axes[1].plot([dst_points[0][0], dst_points[1][0], dst_points[2][0], dst_points[3][0], dst_points[0][0]],
                 [dst_points[0][1], dst_points[1][1], dst_points[2][1], dst_points[3][1], dst_points[0][1]],
                 'g-', linewidth=2, label='Destination area')
    axes[1].scatter(dst_points[:, 0], dst_points[:, 1], c='green', s=50, zorder=5)
    axes[1].set_title('Perspective Corrected Image')
    axes[1].legend()
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    # Save the corrected image if output path is provided
    if output_path:
        corrected_bgr = cv2.cvtColor(corrected_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, corrected_bgr)
        print(f"Corrected image saved to: {output_path}")

    return corrected_img, matrix

#%% Función interactiva para seleccionar puntos
def interactive_point_selection(image_path):
    """
    Interactive function to help select the correct source points
    by clicking on the image.
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    points = []

    def onclick(event):
        if event.inaxes and len(points) < 4:
            points.append([event.xdata, event.ydata])
            plt.plot(event.xdata, event.ydata, 'ro', markersize=8)
            plt.text(event.xdata + 10, event.ydata, f'Point {len(points)}',
                    fontsize=12, color='red', weight='bold')
            plt.draw()

            if len(points) == 4:
                # Draw lines connecting the points
                x_coords = [p[0] for p in points] + [points[0][0]]
                y_coords = [p[1] for p in points] + [points[0][1]]
                plt.plot(x_coords, y_coords, 'r-', linewidth=1)
                plt.draw()
                print("All 4 points selected!")
                print("Points (in order):", points)
                plt.title("All 4 points selected! Close the window.")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_rgb)
    ax.set_title("Click to select 4 corners of the pyramid base\n(Order: top-left, top-right, bottom-right, bottom-left)")
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    while len(points) < 4:
        plt.pause(0.1)

    fig.canvas.mpl_disconnect(cid)
    plt.close(fig)
    # print(f"Se han seleccionado {len(points)} puntos")
    return np.float32(points) if len(points) == 4 else None

#%% Prueba de escritorio de la función anterior
image_path = "../img/paisaje.jpg"
%matplotlib qt
puntos = interactive_point_selection(image_path)
print("Puntos seleccionados:", puntos)

#%% Carga y visualiza la imagen
image_path = "../img/paisaje.jpg"
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 7))
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()




#%% Alternative function for fine-tuned correction
def fine_tune_perspective(image_path, src_points_custom=None):
    """
    Apply perspective correction with custom source points
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]

    if src_points_custom is not None:
        src_points = np.float32(src_points_custom)
    else:
        # Default points - adjust these based on your pyramid location
        src_points = np.array([
            [width * 0.35, height * 0.45],
            [width * 0.65, height * 0.45],
            [width * 0.75, height * 0.85],
            [width * 0.25, height * 0.85]
        ], dtype=np.float32)

    # Calculate destination rectangle to fill more of the image
    margin = 50
    dst_points = np.array([
        [margin, margin],
        [width - margin, margin],
        [width - margin, height - margin],
        [margin, height - margin]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    corrected_img = cv2.warpPerspective(img_rgb, matrix, (width, height))

    return corrected_img, matrix

#%% Example usage:
if __name__ == "__main__":

    # Replace with your image path
    image_path = "../img/paisaje.jpg"

    # # Method 1: Use predefined points (adjust the coordinates in the function above)
    # try:
    #     corrected_img, transformation_matrix = correct_pyramid_perspective(image_path, "corrected_pyramid.jpg")
    # except Exception as e:
    #     print(f"Error: {e}")

    # Method 2: Interactive point selection (uncomment to use)
    %matplotlib qt
    selected_points = interactive_point_selection(image_path)
    if selected_points is not None:
        print("Use these coordinates in the src_points variable:")
        for i, point in enumerate(selected_points):
            print(f"Point {i+1}: [{point[0]:.1f}, {point[1]:.1f}]")
        # Now apply the fine-tuned perspective correction
        try:
            corrected_img, transformation_matrix = fine_tune_perspective(image_path, selected_points)
        except Exception as e:
            print(f"Error: {e}")
        plt.figure(figsize=(10, 7))
        plt.imshow(corrected_img)
        plt.title("Fine-tuned Perspective Corrected Image")
        plt.axis('off')
        plt.show()

# %%
