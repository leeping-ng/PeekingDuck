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
from pathlib import Path

from typing import Any, Dict

from google.cloud import storage
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """This is a template class of how to write a node for peekingduck."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)

        self.bucket_name = config["bucket_name"]
        self.blob_dir = config["blob_dir"]
        self.dest_dir = config["dest_dir"]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """ This node does ___.

        Args:
            inputs (dict): Dict with keys "__", "__".

        Returns:
            outputs (dict): Dict with keys "__".
        """

        storage_client = storage.Client()
        blobs = storage_client.list_blobs(
            self.bucket_name, prefix=self.blob_dir)
        for blob in blobs:
            if blob.name.endswith("/"):
                continue

            filename = blob.name.split("/")[-1]
            file_split = blob.name.split("/")
            sub_dir = "/".join(file_split[0:-1])

            Path(os.path.join(self.dest_dir, sub_dir)).mkdir(
                parents=True, exist_ok=True)
            blob.download_to_filename(os.path.join(
                self.dest_dir, sub_dir, filename))
            self.logger.info("Blob %s downloaded", filename)

        return {}
