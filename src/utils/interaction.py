from dataclasses import dataclass

import cv2
import os
from screeninfo import get_monitors,Monitor

from src.logger import logger
from src.utils.image import ImageUtils

#monitor_window = get_monitors()[0]
monitor_window = Monitor(0, 0, 1000, 1000, 100, 100, 'FakeMonitor', False)


@dataclass
class ImageMetrics:
    # TODO: Move TEXT_SIZE, etc here and find a better class name
    window_width, window_height = monitor_window.width, monitor_window.height
    # for positioning image windows
    window_x, window_y = 0, 0
    reset_pos = [0, 0]


class InteractionUtils:
    """Perform primary functions such as displaying images and reading responses"""

    image_metrics = ImageMetrics()

    @staticmethod
    def show(name, origin, pause=1, resize=False, reset_pos=None, config=None, save_dir='./saved_images'):
        if origin is None:
            logger.info(f"'{name}' - NoneType image to save!")
            return

        if resize:
            if not config:
                raise Exception("config not provided for resizing the image to save")
            img = ImageUtils.resize_util(origin, config.dimensions.display_width)
        else:
            img = origin

        if reset_pos:
            InteractionUtils.image_metrics.window_x = reset_pos[0]
            InteractionUtils.image_metrics.window_y = reset_pos[1]

        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Save image to file
        save_path = os.path.join(save_dir, f"{name}.png")
        cv2.imwrite(save_path, img)
        logger.info(f"Image '{name}' saved at {save_path}")

        # Update window position metrics (if necessary)
        h, w = img.shape[:2]
        margin = 25
        w += margin
        h += margin

        w, h = w // 2, h // 2
        if InteractionUtils.image_metrics.window_x + w > InteractionUtils.image_metrics.window_width:
            InteractionUtils.image_metrics.window_x = 0
            if InteractionUtils.image_metrics.window_y + h > InteractionUtils.image_metrics.window_height:
                InteractionUtils.image_metrics.window_y = 0
            else:
                InteractionUtils.image_metrics.window_y += h
        else:
            InteractionUtils.image_metrics.window_x += w

        # Reset window position metrics if pause is enabled
        if pause:
            logger.info(f"Image '{name}' saved and window position reset.")
            InteractionUtils.image_metrics.window_x = 0
            InteractionUtils.image_metrics.window_y = 0


@dataclass
class Stats:
    # TODO Fill these for stats
    # Move qbox_vals here?
    # badThresholds = []
    # veryBadPoints = []
    files_moved = 0
    files_not_moved = 0


def wait_q():
    esc_key = 27
    while cv2.waitKey(1) & 0xFF not in [ord("q"), esc_key]:
        pass
    cv2.destroyAllWindows()
