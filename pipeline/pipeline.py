import os
import sys

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

global COMP_DIR, num_cores

# Set up AAL labels
aal_labels = {1: 'precentral gyrus', 2: 'precentral gyrus', 3: 'superior frontal gyrus, dorsolateral',
              4: 'superior frontal gyrus, dorsolateral', 5: 'superior frontal gyrus, orbital part',
              6: 'superior frontal gyrus, orbital part', 7: 'middle frontal gyrus, lateral part',
              8: 'middle frontal gyrus, lateral part', 9: 'middle frontal gyrus, orbital part',
              10: 'middle frontal gyrus, orbital part', 11: 'opercular part of inferior frontal gyrus',
              12: 'opercular part of inferior frontal gyrus', 13: 'area triangularis', 14: 'area triangularis',
              15: 'orbital part of inferior frontal gyrus', 16: 'orbital part of inferior frontal gyrus',
              17: 'rolandic operculum', 18: 'rolandic operculum', 19: 'supplementary motor area',
              20: 'supplementary motor area', 21: 'olfactory cortex', 22: 'olfactory cortex',
              23: 'superior frontal gyrus, medial part', 24: 'superior frontal gyrus, medial part',
              25: 'superior frontal gyrus, medial orbital part', 26: 'superior frontal gyrus, medial orbital part',
              27: 'gyrus rectus', 28: 'gyrus rectus', 29: 'insula', 30: 'insula', 31: 'anterior cingulate gyrus',
              32: 'anterior cingulate gyrus', 33: 'middle cingulate', 34: 'middle cingulate',
              35: 'posterior cingulate gyrus', 36: 'posterior cingulate gyrus', 37: 'hippocampus', 38: 'hippocampus',
              39: 'parahippocampal gyrus', 40: 'parahippocampal gyrus', 41: 'amygdala', 42: 'amygdala',
              43: 'calcarine sulcus', 44: 'calcarine sulcus', 45: 'cuneus', 46: 'cuneus', 47: 'lingual gyrus',
              48: 'lingual gyrus', 49: 'superior occipital', 50: 'superior occipital', 51: 'middle occipital',
              52: 'middle occipital', 53: 'inferior occipital', 54: 'inferior occipital', 55: 'fusiform gyrus',
              56: 'fusiform gyrus', 57: 'postcentral gyrus', 58: 'postcentral gyrus', 59: 'superior parietal lobule',
              60: 'superior parietal lobule', 61: 'inferior parietal lobule', 62: 'inferior parietal lobule',
              63: 'supramarginal gyrus', 64: 'supramarginal gyrus', 65: 'angular gyrus', 66: 'angular gyrus',
              67: 'precuneus', 68: 'precuneus', 69: 'paracentral lobule', 70: 'paracentral lobule',
              71: 'caudate nucleus', 72: 'caudate nucleus', 73: 'putamen', 74: 'putamen', 75: 'globus pallidus',
              76: 'globus pallidus', 77: 'thalamus', 78: 'thalamus', 79: 'transverse temporal gyri',
              80: 'transverse temporal gyri', 81: 'superior temporal gyrus', 82: 'superior temporal gyrus',
              83: 'superior temporal pole', 84: 'superior temporal pole', 85: 'middle temporal gyrus',
              86: 'middle temporal gyrus', 87: 'middle temporal pole', 88: 'middle temporal pole',
              89: 'inferior temporal gyrus', 90: 'inferior temporal gyrus', 91: 'crus I of cerebellar hemisphere',
              92: 'crus I of cerebellar hemisphere', 93: 'crus II of cerebellar hemisphere',
              94: 'crus II of cerebellar hemisphere', 95: 'Lobule III of cerebellar hemisphere',
              96: 'Lobule III of cerebellar hemisphere', 97: 'lobule IV, V of cerebellar hemisphere',
              98: 'lobule IV, V of cerebellar hemisphere', 99: 'Lobule VI of cerebellar hemisphere',
              100: 'Lobule VI of cerebellar hemisphere', 101: 'lobule VIIB of cerebellar hemisphere',
              102: 'lobule VIIB of cerebellar hemisphere', 103: 'lobule VIII of cerebellar hemisphere',
              104: 'lobule VIII of cerebellar hemisphere', 105: 'lobule IX of cerebellar hemisphere',
              106: 'lobule IX of cerebellar hemisphere', 107: 'lobule X of cerebellar hemisphere (flocculus)',
              108: 'lobule X of cerebellar hemisphere (flocculus)', 109: 'Lobule I, II of vermis',
              110: 'Lobule III of vermis', 111: 'Lobule IV, V of vermis', 112: 'Lobule VI of vermis',
              113: 'Lobule VII of vermis', 114: 'Lobule VIII of vermis', 115: 'Lobule IX of vermis',
              116: 'Lobule X of vermis (nodulus)'}

ssp_labels = [u'*', u'Angular Gyrus', u'Anterior Cingulate', u'Caudate', u'Cerebellar Lingual', u'Cerebellar Tonsil',
              u'Cingulate Gyrus', u'Culmen', u'Culmen of Vermis', u'Cuneus', u'Declive', u'Declive of Vermis',
              u'Extra-Nuclear', u'Fusiform Gyrus', u'Inferior Frontal Gyrus', u'Inferior Occipital Gyrus',
              u'Inferior Parietal Lobule', u'Inferior Semi-Lunar Lobule', u'Inferior Temporal Gyrus', u'Lingual Gyrus',
              u'Medial Frontal Gyrus', u'Middle Frontal Gyrus', u'Middle Occipital Gyrus', u'Middle Temporal Gyrus',
              u'Nodule', u'Orbital Gyrus', u'Paracentral Lobule', u'Parahippocampal Gyrus', u'Postcentral Gyrus',
              u'Posterior Cingulate', u'Precentral Gyrus', u'Precuneus', u'Pyramis', u'Pyramis of Vermis',
              u'Rectal Gyrus', u'Subcallosal Gyrus', u'Superior Frontal Gyrus', u'Superior Occipital Gyrus',
              u'Superior Parietal Lobule', u'Superior Temporal Gyrus', u'Supramarginal Gyrus', u'Thalamus',
              u'Transverse Temporal Gyrus', u'Tuber', u'Tuber of Vermis', u'Uncus', u'Uvula', u'Uvula of Vermis']


def create_ai_image(patient_id):
    """
    Creates the AI image file in the original PET image space. This is computed by registering
      the flipped image with the original PET image. The AI index is computed at each voxel.
    """

    # Load PET image
    try:
        pet_nii = nib.load('%s/%s/%s_PET.nii.gz' % (COMP_DIR, patient_id, patient_id))
    except Exception:
        raise IOError('Patient image %s/%s/%s_PET.nii.gz does not exist' % (COMP_DIR, patient_id, patient_id))
    pet = pet_nii.get_data()

    # Flip PET image across sagittal plane to get mirror image
    pet = np.flipud(pet)
    if not (os.path.exists('%s/%s/%s_PET_flipped.nii.gz' % (COMP_DIR, patient_id, patient_id))):
        nib.save(nib.Nifti1Image(pet, pet_nii.get_affine()),
                 '%s/%s/%s_PET_flipped.nii.gz' % (COMP_DIR, patient_id, patient_id))

    # Register flipped PET image to original PET image
    if not (os.path.exists('%s/%s/%s_PET_flipped_to_%s_PET_deformed.nii.gz' % (
            COMP_DIR, patient_id, patient_id, patient_id))):
        os.system(
            "$ANTSPATH/antsIntroduction.sh "
            "-d 3 "
            "-i %s/%s/%s_PET_flipped.nii.gz "
            "-r %s/%s/%s_PET.nii.gz "
            "-s MI "
            "-t RA "
            "-o %s/%s/%s_PET_flipped_to_%s_PET_" % (
                COMP_DIR, patient_id, patient_id, COMP_DIR, patient_id, patient_id, COMP_DIR, patient_id, patient_id,
                patient_id))

    # Get original and flipped PET image data
    orig = pet_nii.get_data().astype(np.float32)
    flip = nib.load('%s/%s/%s_PET_flipped_to_%s_PET_deformed.nii.gz' % (COMP_DIR, patient_id, patient_id, patient_id))
    flip = flip.get_data().astype(np.float32)
    # Clean images for negative values as a result of registration
    idx1 = np.where(orig < 0)
    idx2 = np.where(flip < 0)
    orig[idx1] = 0
    flip[idx1] = 0
    orig[idx2] = 0
    flip[idx2] = 0
    # Compute AI index
    ai = 2 * np.divide(orig - flip, orig + flip)
    # Save AI image which is in the original PET image space
    nib.save(nib.Nifti1Image(ai, pet_nii.get_affine()), '%s/%s/%s_AI.nii.gz' % (COMP_DIR, patient_id, patient_id))
    assert os.path.exists('%s/%s/%s_AI.nii.gz' % (COMP_DIR, patient_id, patient_id))


def create_symmetric_ai_image(patient_id, n_cores=2):
    """
    Creates the symmetric AI image file in the original PET image space. This is computed by first
      creating an iterative template image from the original PET image and its mirror image. Both
      images are then registered to this template image space and the AI index is computed at each
      voxel. This symmetric AI index image is then mapped back to the original PET image space.
    """

    # Create a template using original PET image and its mirror image in subdirectory with suffix _template
    if not os.path.exists('%s/%s/%s_template/%s_%s_PETdeformed.nii.gz' % (
            COMP_DIR, patient_id, patient_id, patient_id, patient_id)):
        os.chdir('%s/%s' % (COMP_DIR, patient_id))
        if not os.path.exists('%s/%s/%s_template' % (COMP_DIR, patient_id, patient_id)):
            os.mkdir('%s/%s/%s_template' % (COMP_DIR, patient_id, patient_id))
        os.chdir('%s/%s/%s_template' % (COMP_DIR, patient_id, patient_id))
        os.system('rm -f *.nii.gz')
        os.system('rm -rf GR*')
        os.system('cp %s/%s/%s_PET.nii.gz %s/%s/%s_PET_flipped.nii.gz %s/%s/%s_template/.' % (
            COMP_DIR, patient_id, patient_id, COMP_DIR, patient_id, patient_id, COMP_DIR, patient_id, patient_id))
        os.system('chmod 775 -R %s/%s/%s_template' % (COMP_DIR, patient_id, patient_id))
        # NOTE: COMPUTATIONALLY EXPENSIVE STEP!
        os.system('$ANTSPATH/buildtemplateparallel.sh -d 3 -o %s_ -m 30x50x20 -c 2 -j %i -i 12 -t GR -s CC *.nii.gz' % (
            patient_id, n_cores))

    # Load original PET image in template space
    try:
        orig_nii = nib.load(
            '%s/%s/%s_template/%s_%s_PETdeformed.nii.gz' % (COMP_DIR, patient_id, patient_id, patient_id, patient_id))
    except Exception:
        raise IOError(
            '%s/%s/%s_template/%s_%s_PETdeformed.nii.gz was not successfully generated!' % (
                COMP_DIR, patient_id, patient_id, patient_id, patient_id))

    # Load original and flipped PET image data in template space
    orig = orig_nii.get_data().astype(np.float32)
    flip = nib.load('%s/%s/%s_template/%s_%s_PET_flippeddeformed.nii.gz' % (
        COMP_DIR, patient_id, patient_id, patient_id, patient_id))
    flip = flip.get_data().astype(np.float32)
    # Clean images for negative values as a result of registration
    idx1 = np.where(orig < 0)
    idx2 = np.where(flip < 0)
    orig[idx1] = 0
    flip[idx1] = 0
    orig[idx2] = 0
    flip[idx2] = 0
    # Compute AI index
    ai = 2 * np.divide(orig - flip, orig + flip)
    # Save Symmetric AI image and warp it back to original PET image space
    nib.save(nib.Nifti1Image(ai, orig_nii.get_affine()),
             '%s/%s/%s_template/%s_symmTemplate_AI.nii.gz' % (COMP_DIR, patient_id, patient_id, patient_id))
    if not os.path.exists('%s/%s/%s_symmTemplate_AI.nii.gz' % (COMP_DIR, patient_id, patient_id)):
        os.system(
            '$ANTSPATH/WarpImageMultiTransform '
            '3 '
            '%s/%s/%s_template/%s_symmTemplate_AI.nii.gz '
            '%s/%s/%s_symmTemplate_AI.nii.gz '
            '-R %s/%s/%s_PET.nii.gz  '
            '-i %s/%s/%s_template/%s_%s_PETAffine.txt %s/%s/%s_template/%s_%s_PETInverseWarp.nii.gz' % (
                COMP_DIR, patient_id, patient_id, patient_id, COMP_DIR, patient_id, patient_id, COMP_DIR, patient_id,
                patient_id, COMP_DIR, patient_id, patient_id, patient_id, patient_id, COMP_DIR, patient_id, patient_id,
                patient_id,
                patient_id))
    assert os.path.exists('%s/%s/%s_symmTemplate_AI.nii.gz' % (COMP_DIR, patient_id, patient_id))


def create_aal_atlas(patient_id):
    """
    Creates the AAL atlas image in patient original PET image space
    """

    if not os.path.exists('%s/%s/%s_AAL.nii.gz' % (COMP_DIR, patient_id, patient_id)):
        # Change working directory to patient directory in COMP_DIR
        os.chdir('%s/%s/' % (COMP_DIR, patient_id))
        os.system('cp %s/*.nii.gz .' % pwd)
        # Register template with AAL segmentation to PET image space using Mutual Information similarity metric
        os.system(
            '$ANTSPATH/antsIntroduction.sh '
            '-d 3 '
            '-i T_template0.nii.gz '
            '-r %s/%s/%s_PET.nii.gz '
            '-s MI '
            '-t RA '
            '-o T_template0_to_%s_PET_' % (
                COMP_DIR, patient_id, patient_id, patient_id))
        # Warp AAL atlas using nearest neighbor interpolation to original PET image space
        os.system(
            '$ANTSPATH/WarpImageMultiTransform '
            '3 '
            'AAL_CortexLabels_NickOasisTemplate.nii.gz '
            '%s_AAL.nii.gz '
            '-R %s/%s/%s_PET.nii.gz '
            '%s/%s/T_template0_to_%s_PET_Affine.txt '
            '--use-NN' % (
                patient_id, COMP_DIR, patient_id, patient_id, COMP_DIR, patient_id, patient_id))
    if not os.path.exists('%s/%s/%s_BrainMask.nii.gz' % (COMP_DIR, patient_id, patient_id)):
        # Change working directory to patient directory in COMP_DIR
        os.chdir('%s/%s/' % (COMP_DIR, patient_id))
        os.system('cp %s/*.nii.gz .' % pwd)
        # Warp AAL atlas brain mask using nearest neighbor interpolation to original PET image space
        os.system(
            '$ANTSPATH/WarpImageMultiTransform '
            '3 '
            ' T_template0_BrainCerebellumMask.nii.gz '
            '%s_BrainMask.nii.gz '
            '-R %s/%s/%s_PET.nii.gz '
            '%s/%s/T_template0_to_%s_PET_Affine.txt '
            '--use-NN' % (
                patient_id, COMP_DIR, patient_id, patient_id, COMP_DIR, patient_id, patient_id))
    assert os.path.exists('%s/%s/%s_AAL.nii.gz' % (COMP_DIR, patient_id, patient_id))
    assert os.path.exists('%s/%s/%s_BrainMask.nii.gz' % (COMP_DIR, patient_id, patient_id))


def create_dataframe(patient_id_list, list_of_resection_localizations):
    """
    Returns the entire DataFrame for the study. The
    """

    # Initialize dictionary object for all patient features
    patient_idx = []
    patient_features = {}

    # Set Parameters
    erode_radius = 1

    # Set up cerebellum ROIs
    aal_cerebellum = np.arange(91, 117)

    # For each patient, collect features in different AAL regions, excluding cerebellar regions
    for patient_id_ii, patient_id in enumerate(patient_id_list):
        print "Running feature collection for patient %s" % patient_id

        # Load PET image
        try:
            pet = nib.load('%s/%s/%s_PET.nii.gz' % (COMP_DIR, patient_id, patient_id))
            pet = pet.get_data()
        except Exception:
            raise IOError('Patient image %s/%s/%s_PET.nii.gz does not exist' % (COMP_DIR, patient_id, patient_id))

        # Get AI
        ai = nib.load('%s/%s/%s_AI.nii.gz' % (COMP_DIR, patient_id, patient_id))
        ai = ai.get_data()
        ai[np.isnan(ai)] = 0

        # Get symmetric template based AI
        symm_ai = nib.load('%s/%s/%s_symmTemplate_AI.nii.gz' % (COMP_DIR, patient_id, patient_id))
        symm_ai = symm_ai.get_data()
        symm_ai[np.isnan(symm_ai)] = 0

        # Get AAL
        aal = nib.load('%s/%s/%s_AAL.nii.gz' % (COMP_DIR, patient_id, patient_id))
        aal = aal.get_data()

        # Erode AAL
        aal_mask = np.zeros(aal.shape)
        aal_mask[aal > 0] = 1
        for k in range(erode_radius):
            aal_mask = ndimage.binary_erosion(aal_mask)

        # Add back in any regions that were too small and eroded away
        aal_copy = np.copy(aal)
        aal_copy[aal_mask == 0] = 0
        old_labels = np.unique(aal[aal > 0])
        new_labels = np.unique(aal_copy[aal_copy > 0])
        for label in old_labels:
            if label not in new_labels:
                aal_mask[aal == label] = 1
        aal[aal_mask == 0] = 0

        # Get hemisphere of resection to determine ipsilateral and contralateral regions
        resection_hemisphere = list_of_resection_localizations[patient_id_ii]

        # Add to dataset
        patient_idx.append(patient_id)

        # Initialize features data structure for patient "pt"
        patient_features[patient_id] = {}

        # Voxel AI
        for roi_id in np.unique(aal[aal > 0]):
            # Ensure not in AAL cerebellum
            if roi_id in aal_cerebellum:
                continue

            # Ensure both hemispheres are present
            if roi_id % 2 == 1 and not ((aal[roi_id + 1 == aal]).any()):
                continue
            if (roi_id % 2 == 0) and not ((aal[aal == roi_id - 1]).any()):
                continue
            # Skip if ROI in AAL id is even because we are doing both hemispheres at the same time,
            # i.e. when roi_id =- 1
            if roi_id % 2 == 0:
                continue

            # Ipsilateral and Contralateral average asymmetry index in region "roi_id"
            if resection_hemisphere == 'R':
                feature_name = 'ipsi_voxel_ai_' + aal_labels[int(roi_id)]
                feature_value = np.mean(ai[aal == roi_id + 1])
                patient_features[patient_id][feature_name] = feature_value

                feature_name = 'contra_voxel_ai_' + aal_labels[int(roi_id)]
                feature_value = np.mean(ai[aal == roi_id])
                patient_features[patient_id][feature_name] = feature_value
            elif resection_hemisphere == 'L':
                feature_name = 'ipsi_voxel_ai_' + aal_labels[int(roi_id)]
                feature_value = np.mean(ai[aal == roi_id])
                patient_features[patient_id][feature_name] = feature_value

                feature_name = 'contra_voxel_ai_' + aal_labels[int(roi_id)]
                feature_value = np.mean(ai[aal == roi_id + 1])
                patient_features[patient_id][feature_name] = feature_value

        # AI
        for roi_id in np.unique(aal[aal > 0]):
            # Ensure not in AAL cerebellum
            if roi_id in aal_cerebellum:
                continue

            # Ensure both hemispheres are present
            if roi_id % 2 == 1 and not ((aal[aal == roi_id + 1]).any()):
                continue
            if roi_id % 2 == 0 and not ((aal[aal == roi_id - 1]).any()):
                continue
            # Skip if ROI in AAL id is even because we are doing both hemispheres at the same time,
            # i.e. when roi_id =- 1
            if roi_id % 2 == 0:
                continue

            # Ipsilateral and Contralateral asymmetry index of the PET average in region "roi_id"
            # Ipsilateral here refers to ipsilateral to the hemisphere of resection
            if resection_hemisphere == 'R':
                mean1 = np.mean(pet[aal == roi_id])
                mean2 = np.mean(pet[aal == roi_id + 1])

                feature_name = 'ipsi_pet_ai_' + aal_labels[int(roi_id)]
                feature_value = 2.0 * (mean2 - mean1) / (mean2 + mean1)
                patient_features[patient_id][feature_name] = feature_value

                feature_name = 'contra_pet_ai_' + aal_labels[int(roi_id)]
                feature_value = 2.0 * (mean1 - mean2) / (mean1 + mean2)
                patient_features[patient_id][feature_name] = feature_value
            elif resection_hemisphere == 'L':
                mean1 = np.mean(pet[aal == roi_id])
                mean2 = np.mean(pet[aal == roi_id + 1])

                feature_name = 'ipsi_pet_ai_' + aal_labels[int(roi_id)]
                feature_value = 2.0 * (mean1 - mean2) / (mean1 + mean2)
                patient_features[patient_id][feature_name] = feature_value

                feature_name = 'contra_pet_ai_' + aal_labels[int(roi_id)]
                feature_value = 2.0 * (mean2 - mean1) / (mean2 + mean1)
                patient_features[patient_id][feature_name] = feature_value

        # Symmetric AI
        for roi_id in np.unique(aal[aal > 0]):
            # Ensure not in AAL cerebellum
            if roi_id in aal_cerebellum:
                continue

            # Ensure both hemispheres are present
            if roi_id % 2 == 1 and (not ((aal[roi_id + 1 == aal]).any())):
                continue
            if (roi_id % 2 == 0) and not ((aal[aal == roi_id - 1]).any()):
                continue
            # Skip if ROI in AAL id is even because we are doing both hemispheres at the same time,
            # i.e. when roi_id =- 1
            if roi_id % 2 == 0:
                continue

            if resection_hemisphere == 'R':
                feature_name = 'ipsi_voxel_symm_ai_' + aal_labels[int(roi_id)]
                feature_value = np.mean(symm_ai[aal == roi_id + 1])
                patient_features[patient_id][feature_name] = feature_value

                feature_name = 'contra_voxel_symm_ai_' + aal_labels[int(roi_id)]
                feature_value = np.mean(symm_ai[aal == roi_id])
                patient_features[patient_id][feature_name] = feature_value
            elif resection_hemisphere == 'L':
                feature_name = 'ipsi_voxel_symm_ai_' + aal_labels[int(roi_id)]
                feature_value = np.mean(symm_ai[aal == roi_id])
                patient_features[patient_id][feature_name] = feature_value

                feature_name = 'contra_voxel_symm_ai_' + aal_labels[int(roi_id)]
                feature_value = np.mean(symm_ai[aal == roi_id + 1])
                patient_features[patient_id][feature_name] = feature_value

    # Check that all feature values filled
    feature_names = []
    for first_patient in [sorted(patient_features.keys())[0]]:
        for feature_name in sorted(patient_features[first_patient].keys()):
            feature_names.append(feature_name)
    for pt_number, pt_id in enumerate(sorted(patient_features.keys())[1:]):
        for feature_name in sorted(patient_features[pt_id].keys()):
            if feature_name not in feature_names:
                raise ValueError("Extra Feature %s in patient %s not found!!" % (feature_name, pt_id))
        for feature_name in feature_names:
            if feature_name not in sorted(patient_features[pt_id].keys()):
                raise ValueError("Feature %s not found in patient %s's feature list!!" % (feature_name, pt_id))

    # Convert to data matrix
    data = []
    for pt_number, pt_id in enumerate(sorted(patient_features.keys())):
        row_data = []
        for feature_name in feature_names:
            row_data.append(patient_features[pt_id][feature_name])
        data.append(row_data)

    # Initialize dataframe with clinical features
    df_clinical_features = pd.DataFrame({'id': patient_idx,
                                         })

    df_patient_idx_and_data_matrix = pd.DataFrame(
        np.hstack((np.reshape(np.array(patient_idx), [len(patient_idx), 1]), np.array(data))),
        columns=np.hstack((np.array(('id',)), np.array(feature_names))))

    # Merge patient ID with feature row
    df = pd.merge(df_clinical_features, df_patient_idx_and_data_matrix)

    return df


def create_html_output(patient_id_list, resection_hemispheres, feature_dataframe):
    """
    Creates output HTML files to view feature images and values for each patient and each region.
        Output HTML file out.html and patient specific HTML files are generated.
    """

    # Create output HTML text
    main_output_html_text = """
    <html>
    <h1> PET Analysis Result </h1>
    <ol>
    """
    for patient_id_ii, patient_id in enumerate(patient_id_list):
        # Add to main output HTML text
        main_output_html_text += """
            <li><a href="%s/%s.html">%s</li>
        """ % (patient_id, patient_id.replace(' ', ''), patient_id)

        # Create individual result HTML files
        print(patient_id)
        individual_output_html_text = """
            <html>
            <style>
            table {
              font-family: arial, sans-serif;
              border-collapse: collapse;
              width: 100%%;
            }

            td, th {
              border: 1px solid #dddddd;
              text-align: left;
              padding: 8px;
            }

            tr:nth-child(even) {
              background-color: #dddddd;
            }
            </style>
            </head>
            <body>
            <h1>Patient Identifier: %s</h1>
            <h2>Hemisphere of Resection: %s</h2>
            """ % (patient_id, resection_hemispheres[patient_id_ii])

        # Generate screenshot images of all images
        # Get brain mask
        mask = nib.load('%s/%s/%s_BrainMask.nii.gz' % (COMP_DIR, patient_id, patient_id))
        mask = mask.get_data()
        mask[mask != 0] = 1
        for image_type, colormap in [
            ('AAL', cv2.COLORMAP_JET),
            ('PET', cv2.COLORMAP_HOT),
            ('PET_flipped', cv2.COLORMAP_HOT),
            ('AI', cv2.COLORMAP_JET),
            ('symmTemplate_AI', cv2.COLORMAP_JET),
        ]:
            if image_type.endswith('_exp'):
                image = nib.load(
                    '%s/%s/%s_%s.nii.gz' % (COMP_DIR, patient_id, patient_id, image_type.replace('_exp', '')))
            else:
                image = nib.load('%s/%s/%s_%s.nii.gz' % (COMP_DIR, patient_id, patient_id, image_type))
            image = image.get_data()
            image[mask == 0] = 0
            nn_x, nn_y, nn_z = image.shape[0], image.shape[1], image.shape[2]
            image = image / (1.0 * np.max(image.flatten())) * 254.0
            image = image.astype(np.uint8)

            image_slice = image[nn_x / 2, :, :].squeeze()
            image_slice = np.rot90(image_slice)
            heatmap = cv2.applyColorMap(image_slice, colormap)
            cv2.imwrite('%s/%s/%s_%s_x.png' % (COMP_DIR, patient_id, patient_id, image_type), heatmap)

            image_slice = image[:, nn_y / 2, :].squeeze()
            image_slice = np.rot90(image_slice)
            heatmap = cv2.applyColorMap(image_slice, colormap)
            cv2.imwrite('%s/%s/%s_%s_y.png' % (COMP_DIR, patient_id, patient_id, image_type), heatmap)

            image_slice = image[:, :, nn_z / 2].squeeze()
            image_slice = np.rot90(image_slice)
            heatmap = cv2.applyColorMap(image_slice, colormap)
            cv2.imwrite('%s/%s/%s_%s_z.png' % (COMP_DIR, patient_id, patient_id, image_type), heatmap)

            # output: aal roi seg
            if image_type == 'AAL':
                headline = 'AAL Atlas Regions'
            elif image_type == 'PET':
                headline = 'Original PET Image'
            elif image_type == 'PET_flipped':
                headline = 'Mirror of PET Image'
            elif image_type == 'AI':
                headline = 'Asymmetry Index (Voxel-wise)'
            elif image_type == 'symmTemplate_AI':
                headline = 'Asymmetry Index based on Symmetric Template generation'
            else:
                raise ValueError("Variable headline is not defined")

            # Add feature name and image
            individual_output_html_text += """<h2>%s</h2>
            <br>
            <img src="%s_%s_x.png">
            <img src="%s_%s_y.png">
            <img src="%s_%s_z.png">
            <br>\n""" % (headline, patient_id, image_type, patient_id, image_type, patient_id, image_type)

        # Table of feature values
        individual_output_html_text += """
        <table>
          <tr>
            <th>Feature Name</th>
            <th>Feature Value</th>
          </tr>
        """
        for column in feature_dataframe:
            try:
                individual_output_html_text += """
                  <tr>
                    <td>%s</td>
                    <td>%.8f</td>
                  </tr>
                """ % (column, np.float32(feature_dataframe[column][0]))
            except ValueError:
                individual_output_html_text += """
                  <tr>
                    <td>%s</td>
                    <td>%s</td>
                  </tr>
                """ % (column, feature_dataframe[column][0])
        individual_output_html_text += """</table>"""

        # individual_output_html_text += """Probability of Poor Outcome: """
        individual_output_html_text += """
            <br>
            <br>
            <br>
            <br>
            </body>
            </html>
            """
        open('out/%s/%s.html' % (patient_id, patient_id.replace(' ', '')), 'w').write(individual_output_html_text)

    main_output_html_text += """
    </ol>
    </html>
    """

    open('out/out.html', 'w').write(main_output_html_text)
    return


if __name__ == '__main__':
    # Get current directory
    pwd = os.getcwd()

    # Check to see inputs are correct
    try:
        if not sys.argv[2].endswith('csv'):
            raise IOError('%s is not a properly formatted patient CSV file' % sys.argv[2])
        else:
            lines = open(sys.argv[2], 'r').readlines()
    except Exception:
        if os.path.exists('sample_data.csv'):
            lines = open('sample_data.csv', 'r').readlines()
        else:
            raise IOError('No CSV file has been provided as input.')

    # Set COMP_DIR
    try:
        COMP_DIR = sys.argv[3]
    except Exception:
        if not os.path.exists('out/'):
            os.mkdir('out/')
        COMP_DIR = os.path.join(pwd, 'out/')

    # Get num_cores
    try:
        num_cores = sys.argv[4]
    except Exception:
        num_cores = 2

    # Generate list of patient identifiers, images
    ptx = []
    patient_images = []
    hemispheres_of_resection = []
    for line in lines:
        patient_images.append(line.split(',')[0].rstrip('\n'))
        hemisphere_of_resection = line.split(',')[1].rstrip('\n').replace(' ', '')
        try:
            assert hemisphere_of_resection == 'R' or hemisphere_of_resection == 'L'
        except:
            raise ValueError('Hemisphere of resection in the following line is invalid:\n%s' % line)
        hemispheres_of_resection.append(hemisphere_of_resection)
    for pt in patient_images:
        ptx.append(os.path.basename(pt).split('.')[0])
    print('\nWorking with the following patients: ' + '\n'.join(ptx))
    print('\nWorking with the following patient image files: ' + '\n'.join(patient_images))

    # Copy PET images to output directory and standardize naming
    for ii, pt in enumerate(ptx):
        if not os.path.exists('%s/%s' % (COMP_DIR, pt)):
            os.mkdir('%s/%s' % (COMP_DIR, pt))
            os.system('cp %s %s/%s/%s_PET.nii.gz' % (patient_images[ii], COMP_DIR, pt, pt))

    # For each patient, create the AI image, Symmetric AI image, and AAL atlas
    for ii, pt in enumerate(ptx):
        # Create AI Image
        print('Creating AI image for patient: %s' % pt)
        create_ai_image(pt)

        # Create symmetric template based AI
        print('Creating Symmetric AI image for patient: %s' % pt)
        create_symmetric_ai_image(pt, num_cores)

        # AAL atlas
        print('Creating AAL atlas image for patient: %s' % pt)
        create_aal_atlas(pt)

    # Obtain imaging features for each region in AAL atlas
    features = create_dataframe(ptx, hemispheres_of_resection)

    create_html_output(ptx, hemispheres_of_resection, features)
