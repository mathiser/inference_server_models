import json
import logging
import os

import SimpleITK as sitk
import monai.deploy.core as md
import numpy as np
from monai.deploy.core import ExecutionContext, DataPath, InputContext, IOType, Operator, OutputContext

from .timer import TimeOP
import json

@md.input("seg", np.ndarray, IOType.IN_MEMORY)
@md.input("ref_image", sitk.Image, IOType.IN_MEMORY)
@md.output("", DataPath, IOType.DISK)
@md.env(pip_packages=["monai==0.6.0", "simpleitk", "numpy"])

class DataWriter(Operator):
    def __init__(self):
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__()

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        timer = TimeOP(__name__)

        arr = op_input.get("seg")
        out_path = op_output.get().path

        img = sitk.GetImageFromArray(arr)
        sitk.WriteImage(img, os.path.join(out_path, "predictions.nii.gz"))
        labels = {
            1: "GTVt",
            2: "GTVn"
        }
        with open(os.path.join(out_path, "labels.json"), "w") as f:
            f.write(json.dumps(labels))
        print(timer.report())
