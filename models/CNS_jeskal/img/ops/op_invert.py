import logging
from multiprocessing.pool import ThreadPool
from typing import Dict

import monai.deploy.core as md
import numpy as np
from monai.deploy.core import ExecutionContext, InputContext, IOType, Operator, OutputContext
from .timer import TimeOP


@md.input("", np.ndarray, IOType.IN_MEMORY)
@md.output("", np.ndarray, IOType.IN_MEMORY)
@md.env(pip_packages=["monai==0.6.0", "numpy"])
class InvertImages(Operator):
    def __init__(self):
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

        super().__init__()

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        timer = TimeOP(__name__)
        arr = op_input.get("")

        op_output.set(arr[::-1, :, :])
        print(timer.report())
