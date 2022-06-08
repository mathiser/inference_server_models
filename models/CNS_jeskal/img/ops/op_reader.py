import json
import logging
import os.path
from typing import Dict

import SimpleITK as sitk
import monai.deploy.core as md
import numpy as np
from monai.deploy.core import ExecutionContext, DataPath, InputContext, IOType, Operator, OutputContext, Image
from .timer import TimeOP


@md.input("arr", DataPath, IOType.DISK)
@md.output("arr", np.ndarray, IOType.IN_MEMORY)
@md.output("ref_img", sitk.Image, IOType.IN_MEMORY)
@md.output("meta", Dict, IOType.IN_MEMORY)
@md.env(pip_packages=["monai==0.6.0", "simpleitk", "numpy"])
class DataLoader(Operator):
    def __init__(self):
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__()

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        timer = TimeOP(__name__)

        in_path = op_input.get().path

        for f in os.listdir(in_path):
            path = os.path.join(in_path, f)
            if "0000.nii.gz" in f:
                img = sitk.ReadImage(path)
                op_output.set(value=img, label="ref_img")
                op_output.set(value=sitk.GetArrayFromImage(img), label="arr")
            if f == "meta.json":
                with open(path) as r:
                    meta = dict(json.loads(r.read()))
                op_output.set(value=meta, label="meta")

        print(timer.report())
