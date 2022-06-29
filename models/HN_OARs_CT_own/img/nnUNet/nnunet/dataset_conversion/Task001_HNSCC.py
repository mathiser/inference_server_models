#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
from collections import OrderedDict

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
import SimpleITK as sitk
import shutil

"""
# ! only required for BraTS dataset.
def copy_BraTS_segmentation_and_convert_labels(in_file, out_file):
    # use this for segmentation only!!!
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 4] = 3
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 1] = 2
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)
"""

if __name__ == "__main__":
    """
    REMEMBER TO CONVERT LABELS BACK TO BRATS CONVENTION AFTER PREDICTION!
    """

    task_name = "Task001_HNSCC"
    downloaded_data_dir = "/mnt/faststorage/hndata/Scans1mm_new/train/"

    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")
    #target_imagesVal = join(target_base, "imagesVal")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")


    maybe_mkdir_p(target_imagesTr)
    #maybe_mkdir_p(target_imagesVal)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)
    maybe_mkdir_p(target_labelsTs)


    patient_names = []
    
    cur = downloaded_data_dir
    for p in subdirs(cur, join=False):
        patdir = join(cur, p)
        patient_name = p
        patient_names.append(patient_name)

        ct = join(patdir, "CT.nii.gz")
        pt = join(patdir, "PT.nii.gz")
        t1 = join(patdir, "T1.nii.gz")
        t2 = join(patdir, "T2.nii.gz")
        gtv = join(patdir, "GTV.nii.gz")
        print(ct)
        assert all([
            isfile(ct),
            isfile(pt),
            isfile(t2),
            isfile(t1),
            isfile(gtv)
        ]), "%s" % patient_name

        shutil.copy(ct, join(target_imagesTr, patient_name + "_0000.nii.gz"))
        shutil.copy(pt, join(target_imagesTr, patient_name + "_0001.nii.gz"))
        shutil.copy(t1, join(target_imagesTr, patient_name + "_0002.nii.gz"))
        shutil.copy(t2, join(target_imagesTr, patient_name + "_0003.nii.gz"))


        shutil.copy(gtv, join(target_labelsTr, patient_name + ".nii.gz"))

        #copy_BraTS_segmentation_and_convert_labels(seg, join(target_labelsTr, patient_name + ".nii.gz"))

    downloaded_data_dir_test = "/mnt/faststorage/hndata/Scans1mm_new/test/"

    test_patient_names = []
    cur = downloaded_data_dir_test
    for p in subdirs(cur, join=False):
        patdir = join(cur, p)
        patient_name = p
        test_patient_names.append(patient_name)

        ct = join(patdir, "CT.nii.gz")
        pt = join(patdir, "PT.nii.gz")
        t1 = join(patdir, "T1.nii.gz")
        t2 = join(patdir, "T2.nii.gz")
        gtv = join(patdir, "GTV.nii.gz")

        assert all([
            isfile(ct),
            isfile(pt),
            isfile(t2),
            isfile(t1),
            isfile(gtv)
        ]), "%s" % patient_name

        shutil.copy(ct, join(target_imagesTs, patient_name + "_0000.nii.gz"))
        shutil.copy(pt, join(target_imagesTs, patient_name + "_0001.nii.gz"))
        shutil.copy(t1, join(target_imagesTs, patient_name + "_0002.nii.gz"))
        shutil.copy(t2, join(target_imagesTs, patient_name + "_0003.nii.gz"))

        shutil.copy(gtv, join(target_labelsTs, patient_name + ".nii.gz"))

    json_dict = OrderedDict()
    json_dict['name'] = "HNSCC"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "NA"
    json_dict['licence'] = "NA"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
        "1": "PT",
        "2": "T1",
        "3": "T2"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "GTVt",
        "2": "GTVn"
    }
    json_dict['numTraining'] = len(patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             patient_names]


    json_dict['test'] =  [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in
                          test_patient_names]
   
    save_json(json_dict, join(target_base, "dataset.json"))