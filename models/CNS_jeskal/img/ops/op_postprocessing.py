## Seperate merged data
from genericpath import isfile
import os
import nibabel as nib
import numpy as np
from skimage.measure import regionprops as regionprops
import nibabel as nib
from pathlib import Path
from datetime import datetime

# import toml

mask_resolution = np.uint8


def _seperate_structures(mask: np.array, mean_plane_: np.double):
    # Should mean_plane_ be included in both halves?
    left_structure = mask.copy()
    right_structure = mask.copy()

    left_structure[round(mean_plane_):len(left_structure[:, 0, 0]), :, :] = 0
    right_structure[0:round(mean_plane_), :, :] = 0

    # left_structure  = np.where(mask[0,:,:] >= round(mean_plane_), 0, mask[0,:,:])
    # right_structure  = np.where(mask[0,:,:] < round(mean_plane_), 0, mask[0,:,:])
    # left_structure  = ([[0 for x in mask if x >= round(mean_plane_)] for y in mask] for z in mask)
    # right_structure = ([[0 for x in mask if x >= 1 and x <= round(mean_plane_)] for y in mask] for z in mask)
    return left_structure, right_structure


class ScanStructure:
    def __init__(self, mask_arr, id, name):
        self.mask_arr = mask_arr
        self.id = id
        self.name = name


def trim_empty_masks(mask_list, structure_array):
    valid_struct_bool = True

    # xlist,ylist = map(list,zip(*[f(value) for value in x]))
    # somelist = [ScanStructure(mask, structure_ID, structure_name) for mask, structure_ID, structure_name in zip(mask_list, structure_array[0], structure_array[1]) if np.any(mask)]
    # somelist = [mask_class for mask_class in mask_class if mask_class.mask.any()]

    # ScanStructList = []
    valid_mask_list = []
    for mask, structure_ID, structure_name in zip(mask_list, structure_array[0], structure_array[1]):
        if (np.any(mask)):
            # valid_mask_list.append(ScanStructure(mask, structure_ID, structure_name))
            valid_mask_list.append(mask)
        else:
            print(f'WARNING: Struct {structure_ID} with ID {structure_name} is non-existent.')
    return valid_mask_list

    # Check for empty layers
    # for i in range(len(mask_list)-1):
    # empty_masks = np.where(mask_list.sum() > 0, 1, 0)
    # structure_array[0] = np.where(empty_masks > 0, structure_array[0], 0)
    # structure_array[1] = np.where(empty_masks > 0, structure_array[1], 0)
    # for i in reversed(range(len(mask_list[:]))):
    #     if not np.any(mask_list[i]):
    #         print(f'Struct {structure_array[0][i]} with ID {structure_array[1][i]} is non-existent.')
    #         del mask_list[i]
    #         del structure_array[0][i]
    #         del structure_array[1][i]

    # for mask, structure_ID, structure_name in zip(mask_list, structure_array[0], structure_array[1]):
    #     if not np.any(mask):
    #         print(f'Struct {structure_name} with ID {structure_ID} is non-existent.')
    #         mask_list.remove(mask)
    #         structure_array[0].remove(structure_ID)
    #         structure_array[1].remove(structure_name)
    #         #valid_struct_bool = False

    if (len(mask_list) != len(structure_array[0]) or len(mask_list) != len(structure_array[1])):
        print(
            f'ERROR: Length of structure lists do not match between config (length of {len(structure_array[0])} and {len(structure_array[1])}) and predicted mask (length of {len(mask_list)}')
        return False

    return valid_struct_bool


def find_medial_plane(symmetrical_structure_list):
    # coords_for_sep = [np.round(regionprops(structure.mask)[0].centroid) for structure in symmetrical_structure_list]
    coords_for_sep = [np.round(regionprops(structure)[0].centroid) for structure in symmetrical_structure_list if
                      np.any(structure)]

    medial_plane_coords = [coords[0] for coords in coords_for_sep]
    medial_plane = np.mean(medial_plane_coords)

    return medial_plane


def get_structure_mask_list(nifti_img, structure_ID_list):
    structure_masks = ([np.where(nifti_img == mask_not_for_sep_ID, 1, 0)
                        for mask_not_for_sep_ID
                        in structure_ID_list])
    # structure_masks = []
    # for mask_not_for_sep_ID in structure_ID_list:
    #     masktemp = np.where(nifti_img == mask_not_for_sep_ID, 1, 0)
    #     # masktemp.dtype = mask_resolution
    #     structure_masks.append(masktemp)
    return structure_masks


def seperate_multiple_labels_from_nifti(nifti_input_file_paths_list, nifti_output_paths_list, config_structures_list):
    save_file_paths_list = []
    for nifti_in, nifti_out, config_struct in zip(nifti_input_file_paths_list, nifti_output_paths_list,
                                                  config_structures_list):
        result = seperate_labels_from_nifti(nifti_in, nifti_out, nifti_out)
        if (isinstance(result, Exception)):
            return result
        else:
            save_file_paths_list.append(result)
    return save_file_paths_list


#   Load data and find structures
def seperate_labels_from_nifti(nifti_input_file_path, nifti_output_path, config_structure):
    # Since the function uses binary masks, high resolution is not required

    try:
        if not (os.path.isfile(nifti_input_file_path) and Path(nifti_input_file_path).suffixes == ['.nii', '.gz']):
            return 0

        nifti_img = nib.load(Path((nifti_input_file_path)))
        # current_nifti_img = nib.load(Path(nifti_input_file_path)).get_fdata().astype(mask_resolution)
        current_nifti_img = np.array(nifti_img.dataobj)

        # current_nifti_img.dtype = mask_resolution

        seperated_masks = get_structure_mask_list(current_nifti_img, config_structure['strucs_not_for_seperation'][0])

        masks_for_seperation = get_structure_mask_list(current_nifti_img, config_structure['strucs_for_seperation'][0])

        valid_masks_flag = True
        # seperated_masks = trim_empty_masks(seperated_masks, config_structure['strucs_not_for_seperation'])
        # if(len(seperated_masks) == 0):
        #     print("WARNING: No seperated masks")
        # masks_for_seperation = trim_empty_masks(masks_for_seperation, config_structure['strucs_for_seperation'])

        # if valid_masks_flag == False:
        #     print(f"Failed to seperate structures for {config_structure['name']} scan")
        #     return 0

        # print("WARNING: No masks for seperation"
        if (len(masks_for_seperation) != 0):
            # Find Center of Mass between symmetrical structures
            medial_plane = find_medial_plane(masks_for_seperation)

            # Seperate structures into two
            seperated_structs = [_seperate_structures(mask_for_sep, medial_plane) for mask_for_sep in
                                 masks_for_seperation]
            seperated_masks.extend(
                [single_structure for structure_pair in seperated_structs for single_structure in structure_pair])

        seperated_masks = np.asarray(seperated_masks, dtype=mask_resolution)

        # structure_ID = np.array(config_structure['values'], dtype=mask_resolution)
        structure_ID = np.array(range(1, len(seperated_masks) + 1), dtype=mask_resolution)
        sep_mask = np.zeros_like(current_nifti_img)
        sep_mask = np.sum(
            np.multiply(structure_ID, seperated_masks.transpose(), dtype=mask_resolution, casting='unsafe').transpose(),
            0, dtype=mask_resolution)

        # Prediction delivers flipped segmentation. This fixes the issue. Alternative:
        # sep_mask = np.array(sep_mask)[:, ::-1, :]

        mask_image = nib.Nifti1Image(sep_mask, affine=nifti_img.affine)  # affine=np.eye(4))

        i = 1
        save_file_path = os.path.join(nifti_output_path,
                                      Path(Path(nifti_input_file_path).stem).stem + '_sep' + '.nii.gz')
        while (os.path.isfile(save_file_path)):
            save_file_path = os.path.join(nifti_output_path,
                                          Path(Path(nifti_input_file_path).stem).stem + '_sep_' + str(i) + '.nii.gz')
            i = i + 1

        # Save as nii.gz
        try:
            nib.save(mask_image, save_file_path)
        except OSError as ose:
            print(f'Cannot access {nifti_output_path}. Probably a permissions error.  Full error: \r\n{ose}')
            return ose

        return save_file_path

    except OSError as ose:
        print(f'Cannot access {Path(nifti_input_file_path)}. Probably a permissions error. Full error: \r\n{ose}')
        return ose
    except FileNotFoundError as fnf:
        print(f'{nifti_input_file_path} not found. Full error: \r\n{fnf}')
        return fnf

    # Return pass
    return 0


if __name__ == "__main__":
    # import sys
    import toml
    import os
    from pathlib import PurePath

    config = toml.load('config.toml')

    config_struct = config['fingerprints']['csi']['structures']
    test_data_path = "N:/Begraenset/AUHDCENP_DeepLearningDCPT/scripts_dokumentation/asbbor_scripting_dev_branch/Test_data"
    # patient_ID = "0911LNR_20211111_T1wC_CSI"
    patient_ID = "0911LNR_20211111_T1wC_CSI"
    # seperate_brain_labels_from_nifti(sys.argv[1], sys.argv[2])
    return_path = seperate_labels_from_nifti(
        PurePath(test_data_path, "DCPT_clin_label", patient_ID + ".nii.gz"),
        PurePath(test_data_path, "seperated_labels"),
        config_struct)

    print(f'Seperated labels: {return_path}')
    # seperate_labels_from_nifti(nifti_input_file_path, nifti_output_path, config_structure)