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
@md.output("original_spacing", tuple, IOType.IN_MEMORY)
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
            if "0000.nii" in f:
                out_dict[f], img = self.get_array_and_resampled_from_path(path, use_nn=False)
                op_output.set(value=sitk.ReadImage(path).GetSpacing(), label="original_spacing")
                op_output.set(value=img, label="ref_image")

            if "0001.nii" in f:
                out_dict[f], _ = self.get_array_and_resampled_from_path(path, use_nn=False)
            if "0002.nii" in f:
                out_dict[f], _ = self.get_array_and_resampled_from_path(path, use_nn=False)
            if "0003.nii" in f:
                out_dict[f], _ = self.get_array_and_resampled_from_path(path, use_nn=False)
            if "0004.nii" in f:
                out_dict[f], _ = self.get_array_and_resampled_from_path(path, use_nn=True) ## user input should stay as tinary images (expecting, 0 as background, 1 as GTV-T seed and 2 as GTV-N seed)

        op_output.set(value=out_dict, label="label_array_dict")
        print(timer.report())

    def get_array_and_resampled_from_path(self, p, use_nn=False):
        # returns: array and itk_image

        img = sitk.ReadImage(p)
        resampled_img = self.resemple1mm(img, out_spacing=[1.0, 1.0, 1.0], use_nn=use_nn)

        return sitk.GetArrayFromImage(resampled_img), resampled_img


    def resemple1mm(self, itk_image, out_spacing=[1.0, 1.0, 1.0], use_nn=False):
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