import json
import logging
import os.path
from typing import Dict

import SimpleITK as sitk
import monai.deploy.core as md
import numpy as np
from monai.deploy.core import ExecutionContext, DataPath, InputContext, IOType, Operator, OutputContext
from .timer import TimeOP

@md.input("", DataPath, IOType.DISK)
@md.output("label_array_dict", Dict[str, sitk.Image], IOType.IN_MEMORY)
@md.output("ref_contour_meta", Dict)
@md.env(pip_packages=["monai==0.6.0", "simpleitk", "numpy"])

class DataLoader(Operator):
    """
    This operator loads all images and meta files in a raw unprocessed format.
    """
    def __init__(self):
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__()

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        timer = TimeOP(__name__)

        in_path = op_input.get().path
        out_dict = {}

        for f in os.listdir(in_path):
            path = os.path.join(in_path, f)
            if "0000.nii.gz" in f:
                out_dict["tmp_0000.nii.gz"] = sitk.ReadImage(path)
            if "0001.nii.gz" in f:
                out_dict["tmp_0001.nii.gz"] = sitk.ReadImage(path)
            if "0002.nii.gz" in f:
                out_dict["tmp_0002.nii.gz"] = sitk.ReadImage(path)
            if "0003.nii.gz" in f:
                out_dict["tmp_0003.nii.gz"] = sitk.ReadImage(path)

        gtv_t_path = os.path.join(in_path, "GTVt_feed.nii.gz")
        gtv_t_arr = sitk.GetArrayFromImage(sitk.ReadImage(gtv_t_path))
        with open(gtv_t_path.replace(".nii.gz", ".json")) as r:
            meta_json = dict(json.loads(r.read()))
            op_output.set(value=meta_json, label="ref_contour_meta")

        gtv_n_path = os.path.join(in_path, "GTVn_feed.nii.gz")
        gtv_n_arr = sitk.GetArrayFromImage(sitk.ReadImage(gtv_n_path))

        user_input = np.zeros_like(gtv_t_arr)
        user_input[gtv_t_arr.astype(bool)] = 1
        user_input[gtv_n_arr.astype(bool)] = 2
        user_input_img = sitk.GetImageFromArray(user_input)
        out_dict["tmp_0004.nii.gz"] = user_input_img  ## user input should stay as tinary images (expecting, 0 as background, 1 as GTV-T seed, and 2 as GTV-N seed)

        op_output.set(value=out_dict, label="label_array_dict")
        print(timer.report())

