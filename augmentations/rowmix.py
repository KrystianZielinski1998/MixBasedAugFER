import random
import cv2
import numpy as np

class RowMix:
    def __init__(self,
                row_height_min: int=4,
                row_height_max: int=8,
                gap_min: int=4,
                gap_max: int=8,
                start_offset_min: int=0,
                start_offset_max: int=8,
                seed: int=42
               ):

        self.row_height_min = row_height_min
        self.row_height_max = row_height_max
        self.gap_min = gap_min
        self.gap_max = gap_max
        self.start_offset_min = start_offset_min
        self.start_offset_max = start_offset_max
        self.seed = seed

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"RowMix initialized with parameters: "
                          f"row_height=({row_height_min},{row_height_max}), "
                          f"gap=({gap_min},{gap_max}), "
                          f"start_offset=({start_offset_min},{start_offset_max})")


    def __call__(self, img1, img2):
        """
        Mix two images row-wise, alternating strips of img2 onto img1.

        Args:
            img1: First input image (base image).
            img2: Second input image (to mix into img1).

        Returns:
            result: Augmented image with rows from img2 mixed into img1.
        """

        # Convert images from BGR to RGB (common in OpenCV)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Get image dimensions: height (h) and width (w)
        h, w = img1.shape[:2]

        # Initialize the result image as a copy of the first image
        result = img1.copy()

        # Random starting vertical offset for the first row to mix
        y = random.randint(self.start_offset_min, self.start_offset_max)

        # Continue mixing rows until the bottom of the image is reached
        while y < h:
            # Randomly select the height of the current row to copy from img2
            row_height = random.randint(self.row_height_min, self.row_height_max)

            # Randomly select the gap after the current row before the next one
            gap = random.randint(self.gap_min, self.gap_max)

            # Calculate the end of the current row, but do not exceed image height
            end = min(y + row_height, h)

            # Copy the row from img2 into the result image
            result[y:end, :] = img2[y:end, :]

            # Move the y-coordinate to the start of the next row, adding the gap
            y += row_height + gap

        # Return the final mixed image
        return result


    



            
    
    
    