# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
functions for drawing legend related UI components
"""

from typing import Any, Dict, List, Union

import cv2
import numpy as np
from cv2 import FONT_HERSHEY_SIMPLEX, LINE_AA

from peekingduck.pipeline.nodes.draw.utils.constants import (
    BLACK,
    FILLED,
    PRIMARY_PALETTE,
    PRIMARY_PALETTE_LENGTH,
    WHITE,
)
from peekingduck.pipeline.nodes.draw.utils.general import get_image_size

ZONE_COUNTS_HEADING = "-ZONE COUNTS-"


class Legend:  # pylint: disable=too-many-instance-attributes
    """Legend class that uses available info to draw legend box on frame"""

    def __init__(
        self,
        items: List[str],
        position: str,
        box_opacity: float,
        font: Dict[str, Union[float, int]],
    ) -> None:
        self.items = items  # list of items to be drawn in legend box
        self.position = position
        self.box_opacity = box_opacity
        self.font_size = font["size"]
        self.font_thickness = font["thickness"]

        self.frame = None
        self.legend_left_x = 15
        self.legend_starting_y = 0
        self.delta_y = 0
        self.legend_height = 0
        self.item_height = cv2.getTextSize(
            "",
            FONT_HERSHEY_SIMPLEX,
            self.font_size,
            self.font_thickness,
        )[0][1]
        self.item_padding = self.item_height // 2

    def draw(
        self,
        inputs: Dict[str, Any],
    ) -> np.ndarray:
        """Draw legends onto image

        Args:
            inputs (dict): dictionary of all available inputs for drawing the legend
        """
        self.frame = inputs["img"]

        self.legend_height = self._get_legend_height(inputs)
        self.legend_width = self._get_legend_width(inputs)
        self._set_legend_variables()

        self._draw_legend_box(self.frame)
        y_pos = self.legend_starting_y + self.item_height
        for item in self.items:
            if item == "zone_count":
                self._draw_zone_count(self.frame, y_pos, inputs[item])
            else:
                self.draw_item_info(self.frame, y_pos, item, inputs[item])
            y_pos += self.item_height + self.item_padding

    def draw_item_info(
        self,
        frame: np.ndarray,
        y_pos: int,
        item_name: str,
        item_info: Union[int, float, str],
    ) -> None:
        """Draw item name followed by item info onto frame. If item info is
        of float type, it will be displayed in 2 decimal places.

        Args:
            frame (np.array): image of current frame
            y_pos (int): y_position to draw the count text
            item_name (str): name of the legend item
            item_info: Union[int, float, str]: info contained by the legend item
        """
        if isinstance(item_info, (int, float, str)):
            pass
        else:
            raise TypeError(
                f"With the exception of the 'zone_count' data type, "
                f"the draw.legend node only draws values that are of type 'int', 'float' or 'str' "
                f"within the legend box. The value: {item_info} from the data type: {item_name} "
                f"is of type: {type(item_info)} and is unable to be drawn."
            )

        if isinstance(item_info, float):
            text = f"{item_name.upper()}: {item_info:.2f}"
        else:
            text = f"{item_name.upper()}: {str(item_info)}"
        cv2.putText(
            frame,
            text,
            (self.legend_left_x + self.item_padding, y_pos),
            FONT_HERSHEY_SIMPLEX,
            self.font_size,
            WHITE,
            self.font_thickness,
            LINE_AA,
        )

    def _draw_zone_count(
        self, frame: np.ndarray, y_pos: int, counts: List[int]
    ) -> None:
        """Draw zone counts of all zones onto frame image

        Args:
            frame (np.array): image of current frame
            y_pos (int): y position to draw the count info text
            counts (list): list of zone counts
        """
        cv2.putText(
            frame,
            ZONE_COUNTS_HEADING,
            (self.legend_left_x + self.item_padding, y_pos),
            FONT_HERSHEY_SIMPLEX,
            self.font_size,
            WHITE,
            self.font_thickness,
            LINE_AA,
        )
        for i, count in enumerate(counts):
            y_pos += self.item_height + self.item_padding
            cv2.rectangle(
                frame,
                (self.legend_left_x + self.item_padding, y_pos),
                (
                    self.legend_left_x + self.item_padding + self.item_height,
                    y_pos - self.item_height,
                ),
                PRIMARY_PALETTE[(i + 1) % PRIMARY_PALETTE_LENGTH],
                FILLED,
            )
            text = f" ZONE-{i+1}: {count}"
            cv2.putText(
                frame,
                text,
                (self.legend_left_x + self.item_padding + self.item_height, y_pos),
                FONT_HERSHEY_SIMPLEX,
                self.font_size,
                WHITE,
                self.font_thickness,
                LINE_AA,
            )

    def _draw_legend_box(self, frame: np.ndarray) -> None:
        """draw pts of selected object onto frame

        Args:
            frame (np.array): image of current frame
        """
        assert self.legend_height is not None
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (self.legend_left_x, self.legend_starting_y - self.item_padding),
            (
                self.legend_left_x + self.legend_width,
                self.legend_starting_y + self.legend_height,
            ),
            BLACK,
            FILLED,
        )
        # apply the overlay
        cv2.addWeighted(
            overlay, self.box_opacity, frame, 1 - self.box_opacity, 0, frame
        )

    def _get_legend_height(self, inputs: Dict[str, Any]) -> int:
        """Get height of legend box needed to contain all items drawn"""
        no_of_items = len(self.items)
        if "zone_count" in self.items:
            # increase the number of items according to number of zones
            no_of_items += len(inputs["zone_count"])
        return (self.item_height + self.item_padding) * no_of_items

    def _get_item_width(
        self,
        item_name: str,
        item_info: Union[int, float, str],
    ) -> int:
        """Get width of the text to be drawn. If item info is
        of float type, it will be displayed in 2 decimal places.

        Args:
            item_name (str): name of the legend item
            item_info: Union[int, float, str]: info contained by the legend item
        """
        if not isinstance(item_info, (int, float, str)):
            raise TypeError(
                f"With the exception of the 'zone_count' data type, "
                f"the draw.legend node only draws values that are of type 'int', 'float' or 'str' "
                f"within the legend box. The value: {item_info} from the data type: {item_name} "
                f"is of type: {type(item_info)} and is unable to be drawn."
            )

        if isinstance(item_info, float):
            text = f"{item_name.upper()}: {item_info:.2f}"
        else:
            text = f"{item_name.upper()}: {str(item_info)}"

        text_size = cv2.getTextSize(
            text,
            FONT_HERSHEY_SIMPLEX,
            self.font_size,
            self.font_thickness,
        )

        return text_size[0][0]

    def _get_legend_width(self, inputs: Dict[str, Any]) -> int:
        """Get width of legened box needed to contain all items drawn"""
        max_width = 0
        for item in self.items:
            if item != "zone_count":
                max_width = max(max_width, self._get_item_width(item, inputs[item]))
            else:
                max_width = cv2.getTextSize(
                    ZONE_COUNTS_HEADING,
                    FONT_HERSHEY_SIMPLEX,
                    self.font_size,
                    self.font_thickness,
                )[0][0]

        return max_width + 2 * self.item_padding

    def _set_legend_variables(self) -> None:
        assert self.legend_height != 0
        if self.position == "top":
            self.legend_starting_y = self.item_padding
        else:
            _, image_height = get_image_size(self.frame)
            self.legend_starting_y = (
                image_height - self.item_padding - self.legend_height
            )
