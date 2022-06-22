import json
import logging
import os

import SimpleITK as sitk
import monai.deploy.core as md
import numpy as np
from monai.deploy.core import ExecutionContext, DataPath, InputContext, IOType, Operator, OutputContext
from rt_utils import RTStructBuilder

from .timer import TimeOP
import json


@md.input("seg", np.ndarray, IOType.IN_MEMORY)
@md.input("dcm_dir", str, IOType.IN_MEMORY)
@md.output("", DataPath, IOType.DISK)
@md.env(pip_packages=["monai==0.6.0", "simpleitk", "numpy"])
class DataWriter(Operator):
    def __init__(self):
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__()

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        timer = TimeOP(__name__)

        arr = op_input.get("seg")
        #sitk.WriteImage(sitk.GetImageFromArray(arr), "/home/mathis/writer_pre_transform.nii.gz")
        arr = np.transpose(arr, (2, 0, 1))
        #sitk.WriteImage(sitk.GetImageFromArray(arr), "/home/mathis/writer_post_transform.nii.gz")

        dcm_dir = op_input.get("dcm_dir")
        out_path = op_output.get().path

        contour = {
            1: {"name": "BrainStem", "color": [187, 255, 187]},
            2: {"name": "Chiasm", "color": [250, 20, 250]},
            3: {"name": "Pituitary", "color": [128, 0, 0]},
            4: {"name": "Hippocampus_R", "color": [0, 64, 0]},
            5: {"name": "Hippocampus_L", "color": [0, 127, 0]},
            6: {"name": "OpticNerve_R", "color": [0, 0, 255]},
            7: {"name": "OpticNerve_L", "color": [0, 64, 128]},
            8: {"name": "OpticTract_R", "color": [255, 0, 128]},
            9: {"name": "OpticTract_L", "color": [250, 128, 114]}
        }

        # contour = {
        #     1: {"name": "BrainStem", "color": [187, 255, 187]},
        #     2: {"name": "Chiasm", "color": [250, 20, 250]},
        #     3: {"name": "Pituitary", "color": [128, 0, 0]},
        #     4: {"name": "Hippocampus_LR", "color": [0, 64, 0]},
        #     5: {"name": "OpticNerve_LR", "color": [0, 0, 255]},
        #     6: {"name": "OpticTract_LR", "color": [255, 0, 128]},
        # }
        rtstruct = RTStructBuilder.create_new(dicom_series_path=dcm_dir)

        for i, detail in contour.items():
            mask = self.get_boolean_array(arr, i)
            if True in mask:
                rtstruct.add_roi(
                    mask=mask,
                    color=detail["color"],
                    name=detail["name"]
                )

        rtstruct.save(os.path.join(out_path, "rtstruct_predictions.dcm"))
        print(timer.report())

    def get_boolean_array(self, pred_array: np.array, label_i: int):
        return pred_array == label_i
