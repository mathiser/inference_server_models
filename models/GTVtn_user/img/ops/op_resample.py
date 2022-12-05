import logging
from multiprocessing.pool import ThreadPool
from typing import Dict

import SimpleITK as sitk
import monai.deploy.core as md
import numpy as np
from monai.deploy.core import ExecutionContext, InputContext, IOType, Operator, OutputContext

from .timer import TimeOP


@md.input("label_array_dict", Dict[str, sitk.Image], IOType.IN_MEMORY)
@md.output("label_array_dict", Dict[str, sitk.Image], IOType.IN_MEMORY)
@md.env(pip_packages=["monai==0.6.0", "simpleitk", "numpy"])

class Resample1mm(Operator):
    def __init__(self):
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__()

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        timer = TimeOP(__name__)
        label_array_dict = op_input.get("label_array_dict")
        t = ThreadPool(4)
        results = t.starmap(self.resample1mm, label_array_dict.items())
        t.close()
        t.join()


        #for key, img in label_array_dict.items():
        #    label_array_dict[key] = self.resemple1mm(img, use_nn=False)
        return_dict = {}
        for kv in results:
            k, v = kv
            return_dict[k] = v
        op_output.set(value=return_dict, label="label_array_dict")

        print(timer.report())

    def resample1mm(self, label, itk_image, out_spacing=(1.0, 1.0, 1.0), use_nn=True):
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

        return label, resample.Execute(itk_image)