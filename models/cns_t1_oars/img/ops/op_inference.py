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
@md.env(pip_packages=["monai==0.6.0", "simpleitk", "numpy", "nnunet"])
class Predict(Operator):
    def __init__(self):
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__()

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        timer = TimeOP(__name__)
        img = op_input.get("img")

        tmp_in = tempfile.mkdtemp()
        tmp_out = tempfile.mkdtemp()

        try:
            sitk.WriteImage(img, os.path.join(tmp_in, "tmp_0000.nii.gz"))
            os.system(f"nnUNet_predict -t 504 -tr nnUNetTrainerV2 -f 0 -i {tmp_in} -o {tmp_out}")

            for f in os.listdir(tmp_out):
                if f.endswith(".nii.gz"):
                    pred_img = sitk.ReadImage(os.path.join(tmp_out, f))
                    pred_img.CopyInformation(img)
                    pred_arr = sitk.GetArrayFromImage(pred_img)
                    op_output.set(pred_arr, "seg")
                    break
            else:
                raise Exception("No prediction found")

        except Exception as e:
            self.logger.error(e)
            raise e
        finally:
            pass
            #shutil.rmtree(tmp_in)
            #shutil.rmtree(tmp_out)

        print(timer.report())
