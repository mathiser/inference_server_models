import logging
import os

import monai.deploy.core as md
import numpy as np
from monai.deploy.core import ExecutionContext, DataPath, InputContext, IOType, Operator, OutputContext
from rt_utils import RTStructBuilder

from .timer import TimeOP


@md.input("seg", np.ndarray, IOType.IN_MEMORY)
@md.input("dcm_dir", str, IOType.IN_MEMORY)
@md.output("", DataPath, IOType.DISK)
@md.env(pip_packages=["monai", "simpleitk", "numpy"])
class DataWriter(Operator):
    def __init__(self):
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__()

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        timer = TimeOP(__name__)

        arr_oars = op_input.get("seg")
        arr_oars = np.transpose(arr_oars, (2, 0, 1))

        dcm_dir = op_input.get("dcm_dir")
        out_path = op_output.get().path

        contour = {
            1: {"name": "Brain", "color": [187, 255, 187]},
            2: {"name": "BrainStem", "color": [250, 20, 250]},
            3: {"name": "SpinalCord", "color": [128, 0, 0]},
            4: {"name": "Lips", "color": [0, 64, 0]},
            6: {"name": "Esophagus", "color": [0, 127, 0]},
            8: {"name": "PCM_Low", "color": [0, 0, 255]},
            9: {"name": "PCM_Mid", "color": [0, 64, 128]},
            10: {"name": "PCM_Up", "color": [255, 0, 128]},
            11: {"name": "Mandible", "color": [250, 128, 114]},
            13: {"name": "Thyroid", "color": [0, 127, 0]},
            20: {"name": "Cochlea_L", "color": [0, 0, 255]},
            21: {"name": "Cochlea_R", "color": [0, 64, 128]},
            22: {"name": "Parotid_L", "color": [255, 0, 128]},
            23: {"name": "Parotid_R", "color": [250, 128, 114]},
            24: {"name": "Submandibular_L", "color": [255, 0, 128]},
            25: {"name": "Submandibular_R", "color": [250, 128, 114]},
            26: {"name": "OpticNerve_L", "color": [0, 64, 128]},
            27: {"name": "OpticNerve_R", "color": [255, 0, 128]},
            28: {"name": "EyeFront_L", "color": [250, 128, 114]},
            29: {"name": "EyeFront_R", "color": [255, 0, 128]},
            30: {"name": "EyeBack_L", "color": [250, 128, 114]},
            31: {"name": "EyeBack_R", "color": [250, 128, 114]}

        }

        rtstruct = RTStructBuilder.create_new(dicom_series_path=dcm_dir)

        for i, detail in contour.items():
            mask = self.get_boolean_array(arr_oars, i)
            if True in mask:
                rtstruct.add_roi(
                    mask=mask,
                    #color=detail["color"],
                    name=detail["name"]
                )

        rtstruct.save(os.path.join(out_path, "rtstruct_predictions.dcm"))
        print(timer.report())

    def get_boolean_array(self, pred_array: np.array, label_i: int):
        return pred_array == label_i
