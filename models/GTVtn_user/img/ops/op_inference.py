import logging
import os
import tempfile
from multiprocessing.pool import ThreadPool
from typing import Dict

import SimpleITK as sitk
import monai.deploy.core as md
from monai.deploy.core import ExecutionContext, InputContext, IOType, Operator, OutputContext

from .timer import TimeOP


@md.input("label_array_dict", Dict[str, sitk.Image], IOType.IN_MEMORY)
@md.output("seg", sitk.Image, IOType.IN_MEMORY)
@md.env(pip_packages=["monai==0.6.0", "simpleitk", "numpy", "nnunet"])
class Predict(Operator):
    def __init__(self):
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

        super().__init__()

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        timer = TimeOP(__name__)
        label_array_dict = op_input.get("label_array_dict")

        with tempfile.TemporaryDirectory() as tmp_in, tempfile.TemporaryDirectory() as tmp_out:
            tasks = [(label, array, tmp_in) for label, array in label_array_dict.items()]
            tp = ThreadPool(4)
            tp.starmap(self.save_array_as_image, tasks)
            tp.close()
            tp.join()

            if not os.environ.get("DEBUG"):
                os.system(f"nnUNet_predict -t 101 -tr nnUNetTrainerV2 -f 0 -i {tmp_in} -o {tmp_out}")
            else:
                self.generate_dummy_prediction(label_array_dict, tmp_out)

            for f in os.listdir(tmp_out):
                if f.endswith(".nii.gz"):
                    pred_img = sitk.ReadImage(os.path.join(tmp_out, f))
                    op_output.set(pred_img, "seg")
                    break
            else:
                raise Exception("No prediction found")
            print(timer.report())

    def save_array_as_image(self, label, img, to_dir):
        sitk.WriteImage(img, os.path.join(to_dir, label))

    def generate_dummy_prediction(self, label_array_dict, tmp_out):
        dummy_seg = label_array_dict["tmp_0004.nii.gz"]
        sitk.WriteImage(dummy_seg, os.path.join(tmp_out, "prediction.nii.gz"))