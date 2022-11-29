from monai.deploy.core import Application

from ops.op_reader import DataLoader
from ops.op_writer import DataWriter
from ops.op_inference import Predict
from ops.op_up_scale import UpScaleSegmentation
from ops.op_down_scale import DownScaleContour
from ops.op_resample import Resample1mm
from ops.op_end_resample import EndResample
from ops.op_adjust_ct import AdjustCT

class GTVApplication(Application):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compose(self):
        # Creates the DAG by link the operators
        # 0 Dataloader
        dataloader_op = DataLoader() # load with resample to 1mm

        # 1 Down scale the user input feed according to the meta.json
        down_scale_op = DownScaleContour()

        adjust_ct_op = AdjustCT()
        # 2 Resample
        resample_op = Resample1mm()

        # 3 Predict
        predict_op = Predict()

        # 4 Upscale
        scale_up_op = UpScaleSegmentation()

        # 5 Resample_end
        resample_end_op = EndResample()

        # 6 Write to out
        writer_op = DataWriter()

        # Flows
        self.add_flow(dataloader_op, down_scale_op, {"label_array_dict": "label_array_dict",
                                                     "ref_contour_meta": "ref_contour_meta"})

        self.add_flow(down_scale_op, resample_op, {"label_array_dict": "label_array_dict"})

        self.add_flow(resample_op, adjust_ct_op, {"label_array_dict": "label_array_dict"})

        self.add_flow(adjust_ct_op, predict_op, {"label_array_dict": "label_array_dict"})

        self.add_flow(predict_op, resample_end_op, {"seg": "seg"})
        self.add_flow(dataloader_op, resample_end_op, {"ref_contour_meta": "ref_contour_meta"})

        self.add_flow(resample_end_op, scale_up_op, {"seg": "seg"})
        self.add_flow(dataloader_op, scale_up_op, {"ref_contour_meta": "ref_contour_meta"})

        self.add_flow(scale_up_op, writer_op, {"seg": "seg"})

if __name__ == "__main__":
    # Creates the app and test it standalone. When running is this mode, please note the following:
    #     -i <DICOM folder>, for input DICOM CT series folder
    #     -o <output folder>, for the output folder, default $PWD/output
    #     -m <model file>, for model file path
    # e.g.
    #     python3 app.py -i input -m model.ts
    #
    GTVApplication(do_run=True)

