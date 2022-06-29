import logging

import SimpleITK as sitk
import monai.deploy.core as md
from SimpleITK import DICOMOrientImageFilter
from monai.deploy.core import ExecutionContext, DataPath, InputContext, IOType, Operator, OutputContext

from .timer import TimeOP


@md.input("", DataPath, IOType.DISK)
@md.output("img", sitk.Image, IOType.IN_MEMORY)
@md.output("dcm_dir", str, IOType.IN_MEMORY)
@md.env(pip_packages=["monai", "simpleitk", "numpy"])
class DataLoader(Operator):
    def __init__(self):
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__()

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        timer = TimeOP(__name__)

        in_path = str(op_input.get().path)

        reader = sitk.ImageSeriesReader()
        reader.LoadPrivateTagsOn()
        dicom_names = reader.GetGDCMSeriesFileNames(in_path)
        reader.SetFileNames(dicom_names)

        img = reader.Execute()

        op_output.set(value=img, label="img")
        op_output.set(value=in_path, label="dcm_dir")

        print(timer.report())
