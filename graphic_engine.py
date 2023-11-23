import cv2
import numpy as np

class GraphicEngine:
    def __init__(self):
        # Constructor to initialize the GraphicEngine
        pass

    def draw(self, pixel_matrix):
        """
        Display an image based on the given pixel matrix.

        Args:
            pixel_matrix (list of lists): A matrix representing the image with pixel values (0-255).
        """
        # Ensure pixel matrix is a valid 2D or 3D list
        if not isinstance(pixel_matrix, list) or not all(isinstance(row, list) for row in pixel_matrix):
            raise ValueError("Invalid pixel matrix format")

        # Check if the matrix is 2D or 3D
        if len(pixel_matrix[0]) == 1:
            # 2D matrix (black and white)
            pixel_array = np.array(pixel_matrix, dtype=np.uint8)
        else:
            # 3D matrix (colorful)
            pixel_array = np.array(pixel_matrix, dtype=np.uint8, order='C')

        # Create a window to display the image
        cv2.imshow("Image", pixel_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def matrixify(self, image_path):
        """
        Load an image from the given file path and convert it into a pixel matrix.

        Args:
            image_path (str): Path to the image file.

        Returns:
            list of lists: A 2D pixel matrix for black and white images or a 3D pixel matrix for colorful images.
        """
        # Load the image using OpenCV
        image = cv2.imread(image_path)

        # Ensure the image was loaded successfully
        if image is None:
            raise FileNotFoundError("Image file not found or could not be loaded")

        if len(image.shape) == 2:
            # 2D image (black and white)
            pixel_matrix = image.tolist()
        else:
            # 3D image (colorful)
            pixel_matrix = image.tolist()

        return pixel_matrix

# Example usage:
if __name__ == "__main__":
    graphic_engine = GraphicEngine()

    # # Example 1: Display a black and white image from a pixel matrix
    # bw_pixel_matrix = [[0, 128, 255], [64, 192, 32], [96, 16, 160]]
    # graphic_engine.draw(bw_pixel_matrix)

    # # Example 2: Display a colorful image from a pixel matrix
    # colorful_pixel_matrix = [[[0, 0, 255], [0, 255, 0]], [[255, 0, 0], [255, 255, 255]]
    # graphic_engine.draw(colorful_pixel_matrix)

    # Example 3: Convert an image into a pixel matrix (2D or 3D) and display it
    image_path = "dv.jpeg"
    image_pixel_matrix = graphic_engine.matrixify(image_path)
    graphic_engine.draw(image_pixel_matrix)

