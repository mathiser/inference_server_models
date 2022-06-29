import json
import os.path

from monai.deploy.core import Application

from ops.op_postprocessing import SeparateStructure
from ops.op_inference import Predict
from ops.op_reader import DataLoader
from ops.op_writer import DataWriter


class CNSOARApplication(Application):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compose(self):
        with open(os.path.join(os.path.dirname(__file__), "label_dict_split_labels.json"), "r") as r:
            label_dict = json.loads(r.read())

        dataloader_op = DataLoader()

        predict_op = Predict()
        postprocessing_op = SeparateStructure(label_dict=label_dict)
        writer_op = DataWriter()

        # Flows
        self.add_flow(dataloader_op, predict_op, {"img": "img"})

        self.add_flow(predict_op, postprocessing_op, {"seg": "seg"})

        self.add_flow(dataloader_op, writer_op, {"dcm_dir": "dcm_dir"})
        self.add_flow(postprocessing_op, writer_op, {"seg": "seg"})


        # self.add_flow(dataloader_op, writer_op, {"dcm_dir": "dcm_dir"})
        # self.add_flow(predict_op, writer_op, {"seg": "seg"})

if __name__ == "__main__":
    # Creates the app and test it standalone. When running is this mode, please note the following:
    #     -i <DICOM folder>, for input DICOM CT series folder
    #     -o <output folder>, for the output folder, default $PWD/output
    #     -m <model file>, for model file path
    # e.g.
    #     python3 app.py -i input -m model.ts
    #
    CNSOARApplication(do_run=True)

