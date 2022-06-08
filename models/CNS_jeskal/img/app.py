from monai.deploy.core import Application

from ops.op_invert import InvertImages
from ops.op_reader import DataLoader
from ops.op_writer import DataWriter
from ops.op_inference import Predict
from ops.op_up_scale import Upscale

class CNSOARApplication(Application):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compose(self):
        dataloader_op = DataLoader()
        #inverter_op = InvertImages()
        predict_op = Predict()
        #end_invert_op = InvertImages()
        scale_up_op = Upscale()
        writer_op = DataWriter()

        # Flows
        #self.add_flow(dataloader_op, inverter_op, {"arr": ""})

        #self.add_flow(inverter_op, predict_op, {"": "arr"})
        self.add_flow(dataloader_op, predict_op, {"arr": "arr",
                                                  "ref_img": "ref_img"})

        #self.add_flow(predict_op, end_invert_op, {"seg": ""})
        #self.add_flow(end_invert_op, scale_up_op,  {"": "seg"})

        self.add_flow(predict_op, scale_up_op, {"seg": "seg"})
        self.add_flow(dataloader_op, scale_up_op, {"meta": "meta"})

        self.add_flow(scale_up_op, writer_op, {"seg": "seg"})


if __name__ == "__main__":
    # Creates the app and test it standalone. When running is this mode, please note the following:
    #     -i <DICOM folder>, for input DICOM CT series folder
    #     -o <output folder>, for the output folder, default $PWD/output
    #     -m <model file>, for model file path
    # e.g.
    #     python3 app.py -i input -m model.ts
    #
    CNSOARApplication(do_run=True)

