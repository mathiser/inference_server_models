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
@md.input("original_spacing", tuple, IOType.IN_MEMORY)
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
        original_spacing = op_input.get('original_spacing')
        img = sitk.GetImageFromArray(arr)
        resample_back_img = self.resemple_back(img, out_spacing=original_spacing, use_nn=True)

        sitk.WriteImage(resample_back_img, os.path.join(out_path, "pred.nii.gz"))
        labels = {
            1: "GTVt",
            2: "GTVn"
        }
        with open(os.path.join(out_path, "pred.json"), "w") as f:
            f.write(json.dumps(labels))
        print(timer.report())

    def resemple_back(self, itk_image, out_spacing=[1.0, 1.0, 1.0], use_nn=True):
        # Resample images to 1mm spacing with SimpleITK #2022.11.25

        original_spacing = itk_image.GetSpacing()
        original_size = itk_image.GetSize()

        out_size = [
            int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
            int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size)
        resample.SetOutputDirection(itk_image.GetDirection())
        resample.SetOutputOrigin(itk_image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

        if use_nn:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkBSpline)

        return resample.Execute(itk_image)