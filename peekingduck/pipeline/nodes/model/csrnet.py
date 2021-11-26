# Copyright 2021 AI Singapore
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
Crowd Counting model
"""

from typing import Any, Dict

from peekingduck.pipeline.nodes.node import AbstractNode

from peekingduck.pipeline.nodes.model.csrnetv1 import csrnet_model


class Node(AbstractNode):
    """Initialises and uses CSRNet model to predict the density map and crowd
    count.

    The csrnet node is capable of predicting the number of people in dense and
    sparse crowds. The dense and sparse crowd models were trained using data from
    ShanghaiTech Part A and ShanghaiTech Part B respectively. The node uses the
    sparse crowd model by default and can be changed to using the dense crowd model.

    Inputs:
        |img|

    Outputs:
        |density_map|

        |count|

    Configs:
        model_type (:obj:`str`): **{"dense", "sparse"}, default="sparse"**. |br|
            Defines the type of CSRNet model to be used.
        weights_parent_dir (:obj:`Optional[str]`): **default = null**. |br|
            Change the parent directory where weights will be stored by replacing
            ``null`` with an absolute path to the desired directory.
        width (:obj:`int`): **default = 640**. |br|
            By default, the width of an image will be resized to 640 for inference.
            The height of the image will be resized proportionally to preserve its
            aspect ratio. In general, decreasing the width of an image will improve
            inference speed. However, this might impair the accuracy of the results.

    References:
        CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes:
        https://arxiv.org/pdf/1802.10062.pdf

        Model weights trained by https://github.com/Neerajj9/CSRNet-keras

        Inference code adapted from https://github.com/Neerajj9/CSRNet-keras
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = csrnet_model.CsrnetModel(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reads in image frames and returns the density map and crowd count.

        Args:
            inputs (dict): Dictionary of inputs with key "img".

        Returns:
            outputs (dict): csrnet output in dictionary format with keys
            "density_map" and "count".
        """
        density_map, crowd_count = self.model.predict(inputs["img"])
        outputs = {"density_map": density_map, "count": crowd_count}
        return outputs
