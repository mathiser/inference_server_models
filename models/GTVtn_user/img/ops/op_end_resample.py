import logging
import os
from typing import Dict

import SimpleITK as sitk
import monai.deploy.core as md
from monai.deploy.core import ExecutionContext, InputContext, IOType, Operator, OutputContext
import numpy as np
from .timer import TimeOP


@md.input("seg", sitk.Image, IOType.IN_MEMORY)
@md.input("ref_contour_meta", Dict, IOType.IN_MEMORY)
@md.output("seg", sitk.Image, IOType.IN_MEMORY)
@md.env(pip_packages=["monai==0.6.0", "simpleitk", "numpy"])

class EndResample(Operator):
    def __init__(self):
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__()

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        timer = TimeOP(__name__)

        img = op_input.get("seg")
        ref_contour_meta = op_input.get('ref_contour_meta')

        resample_back_img = self.resample_back(img, ref_contour_meta, use_nn=True)

        op_output.set(resample_back_img, "seg")

        print(timer.report())

    def resample_back(self, itk_image, ref_contour_meta, use_nn=True):
        # Resample images to 1mm spacing with SimpleITK #2022.11.25
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(np.array(ref_contour_meta["spacing"]) * np.array(ref_contour_meta["multiplier"]))

        resample.SetSize([int(n) for n in (np.array(ref_contour_meta["dimensions"]) / np.array(ref_contour_meta["multiplier"]))])
        resample.SetOutputDirection(itk_image.GetDirection())
        resample.SetOutputOrigin(itk_image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

        if use_nn:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkBSpline)

        return resample.Execute(itk_image)