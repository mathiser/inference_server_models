import logging
import os
import shutil
import tempfile
from multiprocessing.pool import ThreadPool
from typing import Dict

import SimpleITK as sitk
import monai.deploy.core as md
import numpy as np
from monai.deploy.core import ExecutionContext, InputContext, IOType, Operator, OutputContext
from .timer import TimeOP


@md.input("img", sitk.Image, IOType.IN_MEMORY)
@md.output("seg", np.ndarray, IOType.IN_MEMORY)
@md.env(pip_packages=["monai", "simpleitk", "numpy", "nnunet"])
class Predict(Operator):
    def __init__(self):
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__()

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        timer = TimeOP(__name__)
        img = op_input.get("img")

        oral_cavity = self.run_inference(img, 508)
        oars = self.run_inference(img, 504)

        oars[oral_cavity == 9] = 12  # Load submandibular (9) from oralcavity model into oars_arr on 12 for postprocess split
        op_output.set(oars, "seg")

        print(timer.report())

    def run_inference(self, image: sitk.Image, task_id: int) -> np.ndarray:
        try:
            with tempfile.TemporaryDirectory() as tmp_in, tempfile.TemporaryDirectory() as tmp_out:
                task_id = str(task_id)

                sitk.WriteImage(image, os.path.join(tmp_in, "tmp_0000.nii.gz"))
                os.system(f"nnUNet_predict -t {task_id} -tr nnUNetTrainerV2 -f 0 -i {tmp_in} -o {tmp_out}")

                for f in os.listdir(tmp_out):
                    if f.endswith(".nii.gz"):
                        pred_img = sitk.ReadImage(os.path.join(tmp_out, f))
                        pred_img.CopyInformation(image)
                        pred_arr = sitk.GetArrayFromImage(pred_img)

                        return pred_arr

                else:
                    raise Exception("No prediction found")

        except Exception as e:
            self.logger.error(e)
            raise e

