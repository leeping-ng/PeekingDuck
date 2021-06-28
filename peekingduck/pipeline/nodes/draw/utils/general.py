from typing import List, Tuple, Any, Iterable, Union
import numpy as np


def get_image_size(frame: np.array) -> Tuple[int, int]:
    """ Obtain image size of input frame

    Args:
        frame (np.array): image of current frame
        
    Returns: 
        image_size (Tuple[int, int]): Width and height of image
    """
    image_size = (frame.shape[1], frame.shape[0])  # width, height
    return image_size


def project_points_onto_original_image(points: np.ndarray,
                                        image_size: Tuple[int, int]) -> np.ndarray:
    """ Project points from relative value (0, 1) to absolute values in original
    image. Note that coordinate (0, 0) starts from image top-left.

    Args:
        points (np.array): points on an image
        image_size (Tuple[int, int]): Width and height of image

    Returns:
        porject_points (np.ndarray): projected points on the original image
    """
    if len(points) == 0:
        return []

    points = points.reshape((-1, 2))

    projected_points = np.array(points, dtype=np.float32)

    width, height = image_size[0], image_size[1]
    projected_points[:, 0] *= width
    projected_points[:, 1] *= height

    return projected_points