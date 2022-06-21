import logging

import SimpleITK
import numpy as np
from typing import Dict
import monai.deploy.core as md
from monai.deploy.core import ExecutionContext, InputContext, IOType, Operator, OutputContext
from .timer import TimeOP


@md.input("seg", np.ndarray, IOType.IN_MEMORY)
@md.output("seg", np.ndarray, IOType.IN_MEMORY)
@md.env(pip_packages=["monai==0.6.0", "numpy", "nnunet"])
class SeparateStructure(Operator):
    def __init__(self, label_dict):
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__()
        self.label_dict = label_dict

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        timer = TimeOP(__name__)
        arr = op_input.get("seg")

        try:

            arr_sep = self.sep_structs(arr, self.label_dict["mirrored"])
            arr_out = self.combine_arrays(arr_sep, arr, self.label_dict["non_mirrored"])

            SimpleITK.WriteImage(SimpleITK.GetImageFromArray(arr_out), "/home/mathis/pred_post_processed.nii.gz")

            op_output.set(arr_out)
            print(timer.report())

        except Exception as e:
            self.logger.error(e)
            raise e

    def get_centroid(self, arr, val) -> float:
        arr_bin = np.where(arr == val, 1, 0)
        arr_1d = arr_bin.sum(axis = 1).sum(axis = 1)
        return (arr_1d*np.arange(len(arr_1d))).sum()/arr_1d.sum()

    def get_sep_arr(self, arr : np.ndarray, center: float, values) -> np.ndarray:
        arr_o = np.zeros_like(arr)
        #Right structure
        arr_o[round(center):len(arr[:, 0, 0]), :, :] = np.where(arr[round(center):len(arr[:, 0, 0]), :, :] == values["pre_val"], values["post_val"][0], 0)
        #Left structure
        arr_o[0:round(center), :, :] = np.where(arr[0:round(center), :, :] == values["pre_val"], values["post_val"][1], 0)
        return arr_o

    def split_struct(self, arr : np.ndarray, values) -> np.ndarray:
        if values["pre_val"] in arr:
            center = self.get_centroid(arr, values["pre_val"])
            sep_arr = self.get_sep_arr(arr, center, values)
        else:
            self.logger.warning(f'Structure "{values["name"]}" is empty.')
            sep_arr = np.zeros_like(arr)
            
        return sep_arr

    def sep_structs(self, arr: np.ndarray, value_cfg: dict) -> np.ndarray:
        sep_arrs = [self.split_struct(arr, values) for values in value_cfg]
        arr_out = np.zeros_like(arr)
        for sep_arr in sep_arrs:
            arr_out = np.where(arr_out == 0, sep_arr, arr_out)
        return arr_out

    def combine_arrays(self,arr_a: np.ndarray, arr_b: np.ndarray, value_cfg) -> np.ndarray:
        for struct in value_cfg:
            arr_b = np.where(arr_b == struct["pre_val"], struct["post_val"], arr_b)
        return np.where(arr_a == 0, arr_b, arr_a)