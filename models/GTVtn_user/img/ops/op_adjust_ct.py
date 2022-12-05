import logging
from typing import Dict

import SimpleITK as sitk
import monai.deploy.core as md
from monai.deploy.core import ExecutionContext, InputContext, IOType, Operator, OutputContext

from .timer import TimeOP


@md.input("label_array_dict", Dict[str, sitk.Image], IOType.IN_MEMORY)
@md.output("label_array_dict", Dict[str, sitk.Image], IOType.IN_MEMORY)
@md.env(pip_packages=["monai==0.6.0", "simpleitk", "numpy"])

class AdjustCT(Operator):
    def __init__(self):
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__()

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        timer = TimeOP(__name__)
        label_array_dict = op_input.get("label_array_dict")
        label_array_dict["tmp_0000.nii.gz"] = label_array_dict["tmp_0000.nii.gz"] - 1000
        op_output.set(value=label_array_dict, label="label_array_dict")

        print(timer.report())

