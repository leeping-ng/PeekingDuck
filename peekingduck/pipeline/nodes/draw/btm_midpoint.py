from typing import Dict
from peekingduck.pipeline.nodes.node import AbstractNode
from .utils.drawfunctions import draw_pts


class Node(AbstractNode):
    def __init__(self, config):
        super().__init__(config, name='draw_btm_midpoint')

    def run(self, inputs: Dict):
        draw_pts(inputs[self.inputs[1]],
                   inputs[self.inputs[0]])
        return {}
