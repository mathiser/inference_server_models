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




if __name__ == "__main__":
    """
    REMEMBER TO CONVERT LABELS BACK TO BRATS CONVENTION AFTER PREDICTION!
    """

    task_name = "Task010_GBM"
    downloaded_data_dir = "/mnt/faststorage/jintao/Glioblastoma/copied/new_train/"
    downloaded_data_dir_val = None

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
        t1 = join(patdir, "t1.nii.gz")
        t1c = join(patdir, "t1c.nii.gz")
        flair = join(patdir,"flair.nii.gz")
        seg = join(patdir, "seg.nii.gz")

        assert all([
            isfile(t1),
            isfile(t1c),
            isfile(flair),
            isfile(seg)
        ]), "%s" % patient_name

        shutil.copy(t1c, join(target_imagesTr, patient_name + "_0000.nii.gz"))
        shutil.copy(t1, join(target_imagesTr, patient_name + "_0001.nii.gz"))
        shutil.copy(flair, join(target_imagesTr, patient_name + "_0002.nii.gz"))
         
        shutil.copy(seg, join(target_labelsTr, patient_name + ".nii.gz"))
         

    if downloaded_data_dir_val is not None:
        for p in subdirs(downloaded_data_dir_val, join=False):
            patdir = join(downloaded_data_dir_val, p)
            patient_name = p
            t1 = join(patdir, "t1.nii.gz")
            t1c = join(patdir, "t1c.nii.gz")
            flair = join(patdir, "flair.nii.gz")

            assert all([
                isfile(t1),
                isfile(t1c),
                isfile(flair),
            ]), "%s" % patient_name

            shutil.copy(t1c, join(target_imagesVal, patient_name + "_0000.nii.gz"))
            shutil.copy(t1, join(target_imagesVal, patient_name + "_0001.nii.gz"))
            shutil.copy(flair, join(target_imagesVal, patient_name + "_0002.nii.gz"))

  
    downloaded_data_dir_test = "/mnt/faststorage/jintao/Glioblastoma/copied/new_test"
    test_patient_names = []
    for p in subdirs(downloaded_data_dir_test, join=False):
        patdir = join(downloaded_data_dir_test, p)
        patient_name = p
        test_patient_names.append(patient_name)

        t1 = join(patdir, "t1.nii.gz")
        t1c = join(patdir, "t1c.nii.gz")
        flair = join(patdir, "flair.nii.gz")
        seg = join(patdir, "seg.nii.gz")

        assert all([
            isfile(t1),
            isfile(t1c),
            isfile(flair),
            isfile(seg)
        ]), "%s" % patient_name

        shutil.copy(t1c, join(target_imagesTs, patient_name + "_0000.nii.gz"))
        shutil.copy(t1, join(target_imagesTs, patient_name + "_0001.nii.gz"))
        shutil.copy(flair, join(target_imagesTs, patient_name + "_0002.nii.gz"))
        shutil.copy(seg, join(target_labelsTs, patient_name + ".nii.gz"))


    json_dict = OrderedDict()
    json_dict['name'] = "Glioblastoma"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "NA"
    json_dict['licence'] = "NA"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "T1c",
        "1": "T1",
        "2": "FLAIR"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "GTV"

    }
    json_dict['numTraining'] = len(patient_names)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             patient_names]
    json_dict['test'] =  [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in
                          test_patient_names]

    save_json(json_dict, join(target_base, "dataset.json"))