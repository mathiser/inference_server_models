import logging
from typing import Dict, Tuple

import SimpleITK as sitk
import monai.deploy.core as md
import numpy as np
import skimage.measure
from monai.deploy.core import ExecutionContext, InputContext, IOType, Operator, OutputContext

from .timer import TimeOP


@md.input("label_array_dict", Dict[str, sitk.Image], IOType.IN_MEMORY)
@md.input("ref_contour_meta", Dict, IOType.IN_MEMORY)
@md.output("label_array_dict", Dict[str, sitk.Image], IOType.IN_MEMORY)
@md.env(pip_packages=["monai==0.6.0", "simpleitk", "numpy", "scikit-image"])
class DownScaleContour(Operator):
    def __init__(self):
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__()

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        timer = TimeOP(__name__)

        arrs = op_input.get("label_array_dict")
        ref_contour_meta = op_input.get("ref_contour_meta")

        arr = sitk.GetArrayFromImage(arrs["tmp_0004.nii.gz"])
        down_scaled = self.down_scale(arr, tuple([n for n in reversed(ref_contour_meta["multiplier"])]))
        down_scaled_img = sitk.GetImageFromArray(down_scaled)
        down_scaled_img.SetSpacing(tuple(np.array(ref_contour_meta["spacing"]) * np.array(ref_contour_meta["multiplier"])))
        down_scaled_img.SetOrigin(ref_contour_meta["origin"])

        arrs["tmp_0004.nii.gz"] = down_scaled_img

        op_output.set(arrs, label="label_array_dict")

        print(timer.report())

    def down_scale(self, img_arr, scale_factor: Tuple):
        reduced_arr = skimage.measure.block_reduce(img_arr, scale_factor, np.max)
        return reduced_arr
