"""
Copyright 2021 AI Singapore

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import yaml
import pytest
import numpy as np
import numpy.testing as npt
import cv2
import tensorflow as tf
from unittest import mock, TestCase
from peekingduck.pipeline.nodes.model.yolo import Node
from peekingduck.pipeline.nodes.model.yolov4.yolo_files.detector import Detector


@pytest.fixture
def yolo_config():
    filepath = os.path.join(
        os.getcwd(), 'tests/pipeline/nodes/model/yolov4/test_yolo.yml')
    with open(filepath) as file:
        node_config = yaml.safe_load(file)
    node_config['root'] = os.getcwd()

    return node_config


@pytest.fixture(params=['v4', 'v4tiny'])
def yolo(request, yolo_config):
    yolo_config['model_type'] = request.param
    node = Node(yolo_config)

    return node


@pytest.fixture()
def yolo_detector(yolo_config):
    yolo_config['model_type'] = 'v4tiny'
    detector = Detector(yolo_config)

    return detector


def replace_download_weights(root, blob_file):
    return False


@pytest.mark.mlmodel
class TestYolo:

    # def test_no_human_image(self, test_no_human_images, yolo):
    #     blank_image = cv2.imread(test_no_human_images)
    #     output = yolo.run({'img': blank_image})
    #     expected_output = {'bboxes': np.empty((0, 4), dtype=np.float32),
    #                        'bbox_labels': np.empty((0)),
    #                        'bbox_scores': np.empty((0), dtype=np.float32)}
    #     assert output.keys() == expected_output.keys()
    #     npt.assert_equal(output['bboxes'], expected_output['bboxes'])
    #     npt.assert_equal(output['bbox_labels'], expected_output['bbox_labels'])
    #     npt.assert_equal(output['bbox_scores'], expected_output['bbox_scores'])

    # def test_return_at_least_one_person_and_one_bbox(self, test_human_images, yolo):
    #     test_img = cv2.imread(test_human_images)
    #     output = yolo.run({'img': test_img})
    #     assert 1 == 1
    #     # assert 'bboxes' in output
    #     # assert output['bboxes'].size != 0

    def test_no_weights(self, yolo_config):
        with mock.patch('peekingduck.weights_utils.checker.has_weights',
                        return_value=False):
            with mock.patch('peekingduck.weights_utils.downloader.download_weights',
                            wraps=replace_download_weights):
                with TestCase.assertLogs(
                        'peekingduck.pipeline.nodes.model.yolov4.yolo_model.logger') \
                        as captured:

                    yolo = Node(yolo_config)

                    assert captured.records[0].getMessage(
                    ) == '---no yolo weights detected. proceeding to download...---'
                    assert captured.records[1].getMessage(
                    ) == '---yolo weights download complete.---'

    def test_get_detect_ids(self, yolo):
        assert yolo.model.get_detect_ids() == [0]
