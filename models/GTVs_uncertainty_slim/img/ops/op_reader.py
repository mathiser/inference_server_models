import logging
import os.path
from typing import Dict

import SimpleITK as sitk
import monai.deploy.core as md
import numpy as np
from monai.deploy.core import ExecutionContext, DataPath, InputContext, IOType, Operator, OutputContext, Image
from .timer import TimeOP

@md.input("", DataPath, IOType.DISK)
@md.output("label_array_dict", Dict[str, np.ndarray], IOType.IN_MEMORY)
@md.output("ref_image", sitk.Image)
@md.env(pip_packages=["monai==0.6.0", "simpleitk", "numpy"])
class DataLoader(Operator):
    def __init__(self):
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__()

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        timer = TimeOP(__name__)

        in_path = op_input.get().path
        out_dict = {}

        for f in os.listdir(in_path):
            path = os.path.join(in_path, f)
            if f == "tmp_0000.nii.gz":
                out_dict[f] = self.get_array_from_path(path)
                op_output.set(value=sitk.ReadImage(path), label="ref_image")
            if f == "tmp_0001.nii.gz":
                out_dict[f] = self.get_array_from_path(path)
            if f == "tmp_0002.nii.gz":
                out_dict[f] = self.get_array_from_path(path)
            if f == "tmp_0003.nii.gz":
                out_dict[f] = self.get_array_from_path(path)

        op_output.set(value=out_dict, label="label_array_dict")
        print(timer.report())
    def get_array_from_path(self, p):
        return sitk.GetArrayFromImage(sitk.ReadImage(p))
