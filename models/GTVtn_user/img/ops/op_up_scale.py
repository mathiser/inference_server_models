import logging
import os
from typing import Dict

import monai.deploy.core as md
import numpy as np
from monai.deploy.core import ExecutionContext, InputContext, IOType, Operator, OutputContext
import SimpleITK as sitk
from .timer import TimeOP
@md.input("seg", sitk.Image, IOType.IN_MEMORY)
@md.input("ref_contour_meta", Dict, IOType.IN_MEMORY)
@md.output("seg", sitk.Image, IOType.IN_MEMORY)
@md.env(pip_packages=["monai==0.6.0", "numpy"])
class UpScaleSegmentation(Operator):
    def __init__(self):
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

        super().__init__()

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        timer = TimeOP(__name__)

        seg = op_input.get("seg")
        ref_contour_meta = op_input.get("ref_contour_meta")
        print(ref_contour_meta)
        rev = [n for n in reversed(ref_contour_meta["multiplier"])]

        seg_arr = sitk.GetArrayFromImage(seg)
        up_scaled_seg_arr = np.kron(seg_arr, np.ones(rev))

        up_scaled_seg = sitk.GetImageFromArray(up_scaled_seg_arr)
        up_scaled_seg.SetSpacing(ref_contour_meta["spacing"])
        up_scaled_seg.SetOrigin(seg.GetOrigin())
        up_scaled_seg.SetDirection(seg.GetDirection())


        op_output.set(up_scaled_seg, "seg")

        print(timer.report())
