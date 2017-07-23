import warnings

# Import general modules
import os
import re
from itertools import cycle
from multiprocessing import Pool

# Import HTML/js libraries
from IPython.display import display, Javascript, HTML, Math, Latex
import jinja2

# Import data science modules
import numpy as np
import scipy
import scipy.stats
from scipy import interp, ndimage
import pandas as pd
import nibabel as nib
import seaborn as sns

# Import graphing modules
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

# Import machine learning modules
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.linear_model import *
from sklearn.model_selection import *
from sklearn.feature_selection import *
from sklearn.preprocessing import *
from sklearn.tree import *
from sklearn.externals import joblib
from sklearn.svm import *
from sklearn.decomposition import *
import sklearn.utils
from sklearn.utils import *

from collections import defaultdict
from collections import OrderedDict

# Import R - py module
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
pandas2ri.activate()
numpy2ri.activate()

##################### DEFINITIONS #####################

# Model parameters
NUM_BOOTSTRAPS = 10
max_depth = 2
RANDOM_STATE = 929
n_proc = 2
pool = Pool(n_proc)

global ssp_labels
ssp_labels = [u'*',u'Angular Gyrus',u'Anterior Cingulate',u'Caudate', u'Cerebellar Lingual',u'Cerebellar Tonsil',u'Cingulate Gyrus',u'Culmen', u'Culmen of Vermis',u'Cuneus',u'Declive',u'Declive of Vermis', u'Extra-Nuclear',u'Fusiform Gyrus',u'Inferior Frontal Gyrus', u'Inferior Occipital Gyrus',u'Inferior Parietal Lobule', u'Inferior Semi-Lunar Lobule',u'Inferior Temporal Gyrus',u'Lingual Gyrus', u'Medial Frontal Gyrus',u'Middle Frontal Gyrus',u'Middle Occipital Gyrus', u'Middle Temporal Gyrus',u'Nodule',u'Orbital Gyrus',u'Paracentral Lobule', u'Parahippocampal Gyrus',u'Postcentral Gyrus',u'Posterior Cingulate', u'Precentral Gyrus',u'Precuneus',u'Pyramis',u'Pyramis of Vermis', u'Rectal Gyrus',u'Subcallosal Gyrus',u'Superior Frontal Gyrus', u'Superior Occipital Gyrus',u'Superior Parietal Lobule', u'Superior Temporal Gyrus',u'Supramarginal Gyrus',u'Thalamus', u'Transverse Temporal Gyrus',u'Tuber',u'Tuber of Vermis',u'Uncus',u'Uvula', u'Uvula of Vermis']


##################### MISCELLAENOUS #####################

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def _helper(job):
    '''
    This helper function trains and predicts on bootstraps of the training data. This is used for parallel computation.
    '''
    job_iter,train,test,X1,y1,XN,yN,classifier1,classifierN = job
    X1_train = X1[train]
    y1_train = y1[train]
    X1_test = X1[test]
    y1_test = y1[test]

    XN_train = XN[train]
    yN_train = yN[train]
    XN_test = XN[test]
    yN_test = yN[test]

    probas_1 = classifier1.fit(X1_train, y1_train).predict_proba(X1_test)
    probas_N = classifierN.fit(XN_train, yN_train).predict_proba(XN_test)

    return [probas_1,y1_test,probas_N,yN_test,classifier1.oob_score_,classifierN.oob_score_]


##################### RESULTS #####################

def compute_table1():
    # Generate final form of Table 1
    columns=[
        'Demographic Feature',
        'Complete Seizure Freedom (Engel IA)',
        'Good Surgical Outcome (Engel IB-D)',
        'Poor Surgical Outcome (Engel II, III, IV)',
        'p-value (Developmental)',
        'Complete Seizure Freedom (Engel IA)',
        'Good Surgical Outcome (Engel IB-D)',
        'Poor Surgical Outcome (Engel II, III, IV)',
        'p-value (Validation)'
            ]
    results = pd.DataFrame(
        {
        'Demographic Feature':[],
        'Complete Seizure Freedom (Engel IA)':[],
        'Good Surgical Outcome (Engel IB-D)':[],
        'Poor Surgical Outcome (Engel II, III, IV)':[],
        'p-value (Developmental)':[],
        'Complete Seizure Freedom (Engel IA)':[],
        'Good Surgical Outcome (Engel IB-D)':[],
        'Poor Surgical Outcome (Engel II, III, IV)':[],
        'p-value (Validation)':[]
                 },
        columns=columns
    )

    # Load data
    df = pd.read_csv('qPET_feature_matrix_clinical_voxel_ai.csv')
    df_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_voxel_ai.csv')

    # Load clinical CSV file
    clinical = pd.read_csv('clinical.csv')

    idx = clinical['Anon ID'].isin(df.id)
    idx_test = clinical['Anon ID'].isin(df_test.id)

    ## Get indices of training and testing for the three different classifications
    idx1 = clinical[idx].Outcome_7.astype(int)==1
    idx2 = (clinical[idx].Outcome_7.astype(int)>1).astype(bool) & (clinical[idx].Outcome_7.astype(int)<5).astype(bool)
    idx3 = clinical[idx].Outcome_7.astype(int)>=5

    idx1_test = clinical[idx_test].Outcome_7.astype(int)==1
    idx2_test = (clinical[idx_test].Outcome_7.astype(int)>1).astype(bool) & (clinical[idx_test].Outcome_7.astype(int)<5).astype(bool)
    idx3_test = clinical[idx_test].Outcome_7.astype(int)>=5


    # Get age vectors
    age1 = robjects.FloatVector(clinical[idx][idx1].Age)
    age2 = robjects.FloatVector(clinical[idx][idx2].Age)
    age3 = robjects.FloatVector(clinical[idx][idx3].Age)
    age1_test = robjects.FloatVector(clinical[idx_test][idx1_test].Age)
    age2_test = robjects.FloatVector(clinical[idx_test][idx2_test].Age)
    age3_test = robjects.FloatVector(clinical[idx_test][idx3_test].Age)
    robjects.globalenv["age1"] = age1
    robjects.globalenv["age2"] = age2
    robjects.globalenv["age3"] = age3
    robjects.globalenv["age1_test"] = age1_test
    robjects.globalenv["age2_test"] = age2_test
    robjects.globalenv["age3_test"] = age3_test

    results = results.append(
        pd.DataFrame(
            [
                [
                    'Total number of subjects',
                    np.count_nonzero(idx1),
                    np.count_nonzero(idx2),
                    np.count_nonzero(idx3),
                    np.nan,
                    np.count_nonzero(idx1_test),
                    np.count_nonzero(idx2_test),
                    np.count_nonzero(idx3_test),
                    np.nan
                ],
                [
                    'Age at surgery',
                    '-',
                    '-',
                    '-',
                    np.nan,
                    '-',
                    '-',
                    '-',
                    np.nan
                ],
                [
                    '-',
                    '%0.2f p/m %0.2f'%(
                        robjects.r('mean(age1)')[0],
                        robjects.r('sd(age1)')[0]
                        ),
                    '%0.2f p/m %0.2f'%(
                        robjects.r('mean(age2)')[0],
                        robjects.r('sd(age2)')[0]
                        ),
                    '%0.2f p/m %0.2f'%(
                        robjects.r('mean(age3)')[0],
                        robjects.r('sd(age3)')[0]
                        ),
                    '',
                    '%0.2f p/m %0.2f'%(
                        robjects.r('mean(age1_test)')[0],
                        robjects.r('sd(age1_test)')[0]
                        ),
                    '%0.2f p/m %0.2f'%(
                        robjects.r('mean(age2_test)')[0],
                        robjects.r('sd(age2_test)')[0]
                        ),
                    '%0.2f p/m %0.2f'%(
                        robjects.r('mean(age3_test)')[0],
                        robjects.r('sd(age3_test)')[0]
                        ),
                    ''
                ]
            ],
            columns=columns
        ),
        ignore_index=True
    )

    # GENDER
    male1 = np.count_nonzero(robjects.BoolVector(clinical[idx][idx1].Sex == 'M'))
    male2 = np.count_nonzero(robjects.BoolVector(clinical[idx][idx2].Sex == 'M'))
    male3 = np.count_nonzero(robjects.BoolVector(clinical[idx][idx3].Sex == 'M'))
    male1_test = np.count_nonzero(robjects.BoolVector(clinical[idx_test][idx1_test].Sex == 'M'))
    male2_test = np.count_nonzero(robjects.BoolVector(clinical[idx_test][idx2_test].Sex == 'M'))
    male3_test = np.count_nonzero(robjects.BoolVector(clinical[idx_test][idx3_test].Sex == 'M'))
    female1 = np.count_nonzero(robjects.BoolVector(clinical[idx][idx1].Sex == 'F'))
    female2 = np.count_nonzero(robjects.BoolVector(clinical[idx][idx2].Sex == 'F'))
    female3 = np.count_nonzero(robjects.BoolVector(clinical[idx][idx3].Sex == 'F'))
    female1_test = np.count_nonzero(robjects.BoolVector(clinical[idx_test][idx1_test].Sex == 'F'))
    female2_test = np.count_nonzero(robjects.BoolVector(clinical[idx_test][idx2_test].Sex == 'F'))
    female3_test = np.count_nonzero(robjects.BoolVector(clinical[idx_test][idx3_test].Sex == 'F'))


    results = results.append(
        pd.DataFrame(
            [
                [
                    'Gender',
                    '-',
                    '-',
                    '-',
                    robjects.r('chisq.test(matrix(c(%i,%i,%i,%i,%i,%i),ncol=3,byrow=TRUE),correct=FALSE)'%(male1,male2,male3,female1,female2,female3)).rx2('p.value')[0],
                    '-',
                    '-',
                    '-',
                    robjects.r('chisq.test(matrix(c(%i,%i,%i,%i,%i,%i),ncol=3,byrow=TRUE),correct=FALSE)'%(male1_test,male2_test,male3_test,female1_test,female2_test,female3_test)).rx2('p.value')[0]
                ],
                [
                    'Male',
                    male1,
                    male2,
                    male3,
                    '-',
                    male1_test,
                    male2_test,
                    male3_test,
                    '-'
                ],
                [
                    'Female',
                    female1,
                    female2,
                    female3,
                    '-',
                    female1_test,
                    female2_test,
                    female3_test,
                    '-'
                ]
            ],
            columns=columns
        ),
        ignore_index=True
    )

    # Resected lobe
    clinical_hemilobe = clinical['Resected Hemi'] + clinical['Resected Lobe']
    ltl_1 = np.count_nonzero(clinical_hemilobe[idx][idx1]=='LTL')
    ltl_2 = np.count_nonzero(clinical_hemilobe[idx][idx2]=='LTL')
    ltl_3 = np.count_nonzero(clinical_hemilobe[idx][idx3]=='LTL')
    ltl_1_test = np.count_nonzero(clinical_hemilobe[idx_test][idx1_test]=='LTL')
    ltl_2_test = np.count_nonzero(clinical_hemilobe[idx_test][idx2_test]=='LTL')
    ltl_3_test = np.count_nonzero(clinical_hemilobe[idx_test][idx3_test]=='LTL')
    rtl_1 = np.count_nonzero(clinical_hemilobe[idx][idx1]=='RTL')
    rtl_2 = np.count_nonzero(clinical_hemilobe[idx][idx2]=='RTL')
    rtl_3 = np.count_nonzero(clinical_hemilobe[idx][idx3]=='RTL')
    rtl_1_test = np.count_nonzero(clinical_hemilobe[idx_test][idx1_test]=='RTL')
    rtl_2_test = np.count_nonzero(clinical_hemilobe[idx_test][idx2_test]=='RTL')
    rtl_3_test = np.count_nonzero(clinical_hemilobe[idx_test][idx3_test]=='RTL')
    lfl_1 = np.count_nonzero(clinical_hemilobe[idx][idx1]=='LFL')
    lfl_2 = np.count_nonzero(clinical_hemilobe[idx][idx2]=='LFL')
    lfl_3 = np.count_nonzero(clinical_hemilobe[idx][idx3]=='LFL')
    lfl_1_test = np.count_nonzero(clinical_hemilobe[idx_test][idx1_test]=='LFL')
    lfl_2_test = np.count_nonzero(clinical_hemilobe[idx_test][idx2_test]=='LFL')
    lfl_3_test = np.count_nonzero(clinical_hemilobe[idx_test][idx3_test]=='LFL')
    rfl_1 = np.count_nonzero(clinical_hemilobe[idx][idx1]=='RFL')
    rfl_2 = np.count_nonzero(clinical_hemilobe[idx][idx2]=='RFL')
    rfl_3 = np.count_nonzero(clinical_hemilobe[idx][idx3]=='RFL')
    rfl_1_test = np.count_nonzero(clinical_hemilobe[idx_test][idx1_test]=='RFL')
    rfl_2_test = np.count_nonzero(clinical_hemilobe[idx_test][idx2_test]=='RFL')
    rfl_3_test = np.count_nonzero(clinical_hemilobe[idx_test][idx3_test]=='RFL')
    lpl_1 = np.count_nonzero(clinical_hemilobe[idx][idx1]=='LPL')
    lpl_2 = np.count_nonzero(clinical_hemilobe[idx][idx2]=='LPL')
    lpl_3 = np.count_nonzero(clinical_hemilobe[idx][idx3]=='LPL')
    lpl_1_test = np.count_nonzero(clinical_hemilobe[idx_test][idx1_test]=='LPL')
    lpl_2_test = np.count_nonzero(clinical_hemilobe[idx_test][idx2_test]=='LPL')
    lpl_3_test = np.count_nonzero(clinical_hemilobe[idx_test][idx3_test]=='LPL')
    rpl_1 = np.count_nonzero(clinical_hemilobe[idx][idx1]=='RPL')
    rpl_2 = np.count_nonzero(clinical_hemilobe[idx][idx2]=='RPL')
    rpl_3 = np.count_nonzero(clinical_hemilobe[idx][idx3]=='RPL')
    rpl_1_test = np.count_nonzero(clinical_hemilobe[idx_test][idx1_test]=='RPL')
    rpl_2_test = np.count_nonzero(clinical_hemilobe[idx_test][idx2_test]=='RPL')
    rpl_3_test = np.count_nonzero(clinical_hemilobe[idx_test][idx3_test]=='RPL')
    results = results.append(
        pd.DataFrame(
            [
                [
                    'Resected Regions',
                    '-',
                    '-',
                    '-',
                    robjects.r('chisq.test(matrix(c(%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i),ncol=3,byrow=TRUE),correct=FALSE)'%(ltl_1,ltl_2,ltl_3,rtl_1,rtl_2,rtl_3,lfl_1+rfl_1,lfl_2+rfl_2,lfl_3+rfl_3,lpl_1+rpl_1,lpl_2+rpl_2,lpl_3+rpl_3)).rx2('p.value')[0],
                    '-',
                    '-',
                    '-',
                    robjects.r('chisq.test(matrix(c(%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i),ncol=3,byrow=TRUE),correct=FALSE)'%(ltl_1_test,ltl_2_test,ltl_3_test,rtl_1_test,rtl_2_test,rtl_3_test,lfl_1_test+rfl_1_test,lfl_2_test+rfl_2_test,lfl_3_test+rfl_3_test,lpl_1_test+rpl_1_test,lpl_2_test+rpl_2_test,lpl_3_test+rpl_3_test)).rx2('p.value')[0]
                ],
                [
                    'LTL',
                    ltl_1,
                    ltl_2,
                    ltl_3,
                    '-',
                    ltl_1_test,
                    ltl_2_test,
                    ltl_3_test,
                    '-'
                ],
                [
                    'RTL',
                    rtl_1,
                    rtl_2,
                    rtl_3,
                    '-',
                    rtl_1_test,
                    rtl_2_test,
                    rtl_3_test,
                    '-'
                ],
                [
                    'LFL/RFL',
                    lfl_1+rfl_1,
                    lfl_2+rfl_2,
                    lfl_3+rfl_3,
                    '-',
                    lfl_1_test+rfl_1_test,
                    lfl_2_test+rfl_2_test,
                    lfl_3_test+rfl_3_test,
                    '-'
                ],
                [
                    'LPL/RPL',
                    lpl_1+rpl_1,
                    lpl_2+rpl_2,
                    lpl_3+rpl_3,
                    '-',
                    lpl_1_test+rpl_1_test,
                    lpl_2_test+rpl_2_test,
                    lpl_3_test+rpl_3_test,
                    '-'
                ]
            ],
            columns=columns
        ),
        ignore_index=True
    )


    # Lesional
    lesional1 = np.count_nonzero(robjects.BoolVector(clinical[idx][idx1]['Lesional? (L;NL)'] == 'L'))
    lesional2 = np.count_nonzero(robjects.BoolVector(clinical[idx][idx2]['Lesional? (L;NL)'] == 'L'))
    lesional3 = np.count_nonzero(robjects.BoolVector(clinical[idx][idx3]['Lesional? (L;NL)'] == 'L'))
    lesional1_test = np.count_nonzero(robjects.BoolVector(clinical[idx_test][idx1_test]['Lesional? (L;NL)'] == 'L'))
    lesional2_test = np.count_nonzero(robjects.BoolVector(clinical[idx_test][idx2_test]['Lesional? (L;NL)'] == 'L'))
    lesional3_test = np.count_nonzero(robjects.BoolVector(clinical[idx_test][idx3_test]['Lesional? (L;NL)'] == 'L'))
    non_lesional1 = np.count_nonzero(robjects.BoolVector(clinical[idx][idx1]['Lesional? (L;NL)'] == 'NL'))
    non_lesional2 = np.count_nonzero(robjects.BoolVector(clinical[idx][idx2]['Lesional? (L;NL)'] == 'NL'))
    non_lesional3 = np.count_nonzero(robjects.BoolVector(clinical[idx][idx3]['Lesional? (L;NL)'] == 'NL'))
    non_lesional1_test = np.count_nonzero(robjects.BoolVector(clinical[idx_test][idx1_test]['Lesional? (L;NL)'] == 'NL'))
    non_lesional2_test = np.count_nonzero(robjects.BoolVector(clinical[idx_test][idx2_test]['Lesional? (L;NL)'] == 'NL'))
    non_lesional3_test = np.count_nonzero(robjects.BoolVector(clinical[idx_test][idx3_test]['Lesional? (L;NL)'] == 'NL'))


    results = results.append(
        pd.DataFrame(
            [
                [
                    'MRI Findings',
                    '-',
                    '-',
                    '-',
                    robjects.r('chisq.test(matrix(c(%i,%i,%i,%i,%i,%i),ncol=3,byrow=TRUE),correct=FALSE)'%(lesional1,lesional2,lesional3,non_lesional1,non_lesional2,non_lesional3)).rx2('p.value')[0],
                    '-',
                    '-',
                    '-',
                    robjects.r('chisq.test(matrix(c(%i,%i,%i,%i,%i,%i),ncol=3,byrow=TRUE),correct=FALSE)'%(lesional1_test,lesional2_test,lesional3_test,non_lesional1_test,non_lesional2_test,non_lesional3_test)).rx2('p.value')[0]
                ],
                [
                    'Lesional',
                    lesional1,
                    lesional2,
                    lesional3,
                    '-',
                    lesional1_test,
                    lesional2_test,
                    lesional3_test,
                    '-'
                ],
                [
                    'Non-Lesional',
                    non_lesional1,
                    non_lesional2,
                    non_lesional3,
                    '-',
                    non_lesional1_test,
                    non_lesional2_test,
                    non_lesional3_test,
                    '-'
                ]
            ],
            columns=columns
        ),
        ignore_index=True
    )




    # Pathology
    hsmts_1 = np.count_nonzero(clinical[idx][idx1]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==1)
    hsmts_2 = np.count_nonzero(clinical[idx][idx2]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==1)
    hsmts_3 = np.count_nonzero(clinical[idx][idx3]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==1)
    hsmts_1_test = np.count_nonzero(clinical[idx_test][idx1_test]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==1)
    hsmts_2_test = np.count_nonzero(clinical[idx_test][idx2_test]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==1)
    hsmts_3_test = np.count_nonzero(clinical[idx_test][idx3_test]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==1)
    gliosis_1 = np.count_nonzero(clinical[idx][idx1]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==2)
    gliosis_2 = np.count_nonzero(clinical[idx][idx2]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==2)
    gliosis_3 = np.count_nonzero(clinical[idx][idx3]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==2)
    gliosis_1_test = np.count_nonzero(clinical[idx_test][idx1_test]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==2)
    gliosis_2_test = np.count_nonzero(clinical[idx_test][idx2_test]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==2)
    gliosis_3_test = np.count_nonzero(clinical[idx_test][idx3_test]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==2)
    mcd_1 = np.count_nonzero(clinical[idx][idx1]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==3)
    mcd_2 = np.count_nonzero(clinical[idx][idx2]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==3)
    mcd_3 = np.count_nonzero(clinical[idx][idx3]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==3)
    mcd_1_test = np.count_nonzero(clinical[idx_test][idx1_test]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==3)
    mcd_2_test = np.count_nonzero(clinical[idx_test][idx2_test]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==3)
    mcd_3_test = np.count_nonzero(clinical[idx_test][idx3_test]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==3)
    tumorvascular_1 = np.count_nonzero(clinical[idx][idx1]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==4)
    tumorvascular_2 = np.count_nonzero(clinical[idx][idx2]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==4)
    tumorvascular_3 = np.count_nonzero(clinical[idx][idx3]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==4)
    tumorvascular_1_test = np.count_nonzero(clinical[idx_test][idx1_test]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==4)
    tumorvascular_2_test = np.count_nonzero(clinical[idx_test][idx2_test]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==4)
    tumorvascular_3_test = np.count_nonzero(clinical[idx_test][idx3_test]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==4)
    dual_1 = np.count_nonzero(clinical[idx][idx1]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==5)
    dual_2 = np.count_nonzero(clinical[idx][idx2]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==5)
    dual_3 = np.count_nonzero(clinical[idx][idx3]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==5)
    dual_1_test = np.count_nonzero(clinical[idx_test][idx1_test]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==5)
    dual_2_test = np.count_nonzero(clinical[idx_test][idx2_test]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==5)
    dual_3_test = np.count_nonzero(clinical[idx_test][idx3_test]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==5)
    normal_1 = np.count_nonzero(clinical[idx][idx1]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==6)+np.count_nonzero(clinical[idx][idx1]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==-1)
    normal_2 = np.count_nonzero(clinical[idx][idx2]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==6)+np.count_nonzero(clinical[idx][idx2]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==-1)
    normal_3 = np.count_nonzero(clinical[idx][idx3]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==6)+np.count_nonzero(clinical[idx][idx3]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==-1)
    normal_1_test = np.count_nonzero(clinical[idx_test][idx1_test]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==6)+np.count_nonzero(clinical[idx_test][idx1_test]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==-1)
    normal_2_test = np.count_nonzero(clinical[idx_test][idx2_test]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==6)+np.count_nonzero(clinical[idx_test][idx2_test]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==-1)
    normal_3_test = np.count_nonzero(clinical[idx_test][idx3_test]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==6)+np.count_nonzero(clinical[idx_test][idx3_test]['PathologyCode (1=MTS/HS;2=Gliosis;3=MCD;4=Tumor;5=Dual;6=Normal)']==-1)

    results = results.append(
        pd.DataFrame(
            [
                [
                    'Pathology',
                    '-',
                    '-',
                    '-',
                    robjects.r('chisq.test(matrix(c(%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i),ncol=3,byrow=TRUE),correct=FALSE)'%(hsmts_1,hsmts_2,hsmts_3,gliosis_1,gliosis_2,gliosis_3,mcd_1,mcd_2,mcd_3,tumorvascular_1,tumorvascular_2,tumorvascular_3,dual_1,dual_2,dual_3,normal_1,normal_2,normal_3)).rx2('p.value')[0],
                    '-',
                    '-',
                    '-',
                    robjects.r('chisq.test(matrix(c(%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i),ncol=3,byrow=TRUE),correct=FALSE)'%(hsmts_1_test,hsmts_2_test,hsmts_3_test,gliosis_1_test,gliosis_2_test,gliosis_3_test,mcd_1_test,mcd_2_test,mcd_3_test,tumorvascular_1_test,tumorvascular_2_test,tumorvascular_3_test,dual_1_test,dual_2_test,dual_3_test,normal_1_test,normal_2_test,normal_3_test)).rx2('p.value')[0]
                ],
                [
                    'HS/MTS',
                    hsmts_1,
                    hsmts_2,
                    hsmts_3,
                    '-',
                    hsmts_1_test,
                    hsmts_2_test,
                    hsmts_3_test,
                    '-'
                ],
                [
                    'Gliosis',
                    gliosis_1,
                    gliosis_2,
                    gliosis_3,
                    '-',
                    gliosis_1_test,
                    gliosis_2_test,
                    gliosis_3_test,
                    '-'
                ],
                [
                    'MCD',
                    mcd_1,
                    mcd_2,
                    mcd_3,
                    '-',
                    mcd_1_test,
                    mcd_2_test,
                    mcd_3_test,
                    '-'
                ],
                [
                    'Tumor/Vascular',
                    tumorvascular_1,
                    tumorvascular_2,
                    tumorvascular_3,
                    '-',
                    tumorvascular_1_test,
                    tumorvascular_2_test,
                    tumorvascular_3_test,
                    '-'
                ],
                [
                    'Dual Pathology',
                    dual_1,
                    dual_2,
                    dual_3,
                    '-',
                    dual_1_test,
                    dual_2_test,
                    dual_3_test,
                    '-'
                ],
                [
                    'Normal/Not Available',
                    normal_1,
                    normal_2,
                    normal_3,
                    '-',
                    normal_1_test,
                    normal_2_test,
                    normal_3_test,
                    '-'
                ]
            ],
            columns=columns
        ),
        ignore_index=True
    )

    # PET Read
    restricted_1 = np.count_nonzero(clinical[idx][idx1]['PET Class (R;S;M;D;Not Available;Normal)']=='R')
    restricted_2 = np.count_nonzero(clinical[idx][idx2]['PET Class (R;S;M;D;Not Available;Normal)']=='R')
    restricted_3 = np.count_nonzero(clinical[idx][idx3]['PET Class (R;S;M;D;Not Available;Normal)']=='R')
    restricted_1_test = np.count_nonzero(clinical[idx_test][idx1_test]['PET Class (R;S;M;D;Not Available;Normal)']=='R')
    restricted_2_test = np.count_nonzero(clinical[idx_test][idx2_test]['PET Class (R;S;M;D;Not Available;Normal)']=='R')
    restricted_3_test = np.count_nonzero(clinical[idx_test][idx3_test]['PET Class (R;S;M;D;Not Available;Normal)']=='R')
    subtle_1 = np.count_nonzero(clinical[idx][idx1]['PET Class (R;S;M;D;Not Available;Normal)']=='S')
    subtle_2 = np.count_nonzero(clinical[idx][idx2]['PET Class (R;S;M;D;Not Available;Normal)']=='S')
    subtle_3 = np.count_nonzero(clinical[idx][idx3]['PET Class (R;S;M;D;Not Available;Normal)']=='S')
    subtle_1_test = np.count_nonzero(clinical[idx_test][idx1_test]['PET Class (R;S;M;D;Not Available;Normal)']=='S')
    subtle_2_test = np.count_nonzero(clinical[idx_test][idx2_test]['PET Class (R;S;M;D;Not Available;Normal)']=='S')
    subtle_3_test = np.count_nonzero(clinical[idx_test][idx3_test]['PET Class (R;S;M;D;Not Available;Normal)']=='S')
    diffuse_multifocal_1 = np.count_nonzero(clinical[idx][idx1]['PET Class (R;S;M;D;Not Available;Normal)']=='D')+np.count_nonzero(clinical[idx][idx1]['PET Class (R;S;M;D;Not Available;Normal)']=='M')
    diffuse_multifocal_2 = np.count_nonzero(clinical[idx][idx2]['PET Class (R;S;M;D;Not Available;Normal)']=='D')+np.count_nonzero(clinical[idx][idx2]['PET Class (R;S;M;D;Not Available;Normal)']=='M')
    diffuse_multifocal_3 = np.count_nonzero(clinical[idx][idx3]['PET Class (R;S;M;D;Not Available;Normal)']=='D')+np.count_nonzero(clinical[idx][idx3]['PET Class (R;S;M;D;Not Available;Normal)']=='M')
    diffuse_multifocal_1_test = np.count_nonzero(clinical[idx_test][idx1_test]['PET Class (R;S;M;D;Not Available;Normal)']=='D')+np.count_nonzero(clinical[idx_test][idx1_test]['PET Class (R;S;M;D;Not Available;Normal)']=='M')
    diffuse_multifocal_2_test = np.count_nonzero(clinical[idx_test][idx2_test]['PET Class (R;S;M;D;Not Available;Normal)']=='D')+np.count_nonzero(clinical[idx_test][idx2_test]['PET Class (R;S;M;D;Not Available;Normal)']=='M')
    diffuse_multifocal_3_test = np.count_nonzero(clinical[idx_test][idx3_test]['PET Class (R;S;M;D;Not Available;Normal)']=='D')+np.count_nonzero(clinical[idx_test][idx3_test]['PET Class (R;S;M;D;Not Available;Normal)']=='M')
    normal_1 = np.count_nonzero(clinical[idx][idx1]['PET Class (R;S;M;D;Not Available;Normal)']=='Normal')
    normal_2 = np.count_nonzero(clinical[idx][idx2]['PET Class (R;S;M;D;Not Available;Normal)']=='Normal')+np.count_nonzero(clinical[idx][idx2]['PET Class (R;S;M;D;Not Available;Normal)']=='Not Available')
    normal_3 = np.count_nonzero(clinical[idx][idx3]['PET Class (R;S;M;D;Not Available;Normal)']=='Normal')+np.count_nonzero(clinical[idx][idx3]['PET Class (R;S;M;D;Not Available;Normal)']=='Not Available')
    normal_1_test = np.count_nonzero(clinical[idx_test][idx1_test]['PET Class (R;S;M;D;Not Available;Normal)']=='Normal')+np.count_nonzero(clinical[idx_test][idx1_test]['PET Class (R;S;M;D;Not Available;Normal)']=='Not Available')
    normal_2_test = np.count_nonzero(clinical[idx_test][idx2_test]['PET Class (R;S;M;D;Not Available;Normal)']=='Normal')+np.count_nonzero(clinical[idx_test][idx2_test]['PET Class (R;S;M;D;Not Available;Normal)']=='Not Available')
    normal_3_test = np.count_nonzero(clinical[idx_test][idx3_test]['PET Class (R;S;M;D;Not Available;Normal)']=='Normal')+np.count_nonzero(clinical[idx_test][idx3_test]['PET Class (R;S;M;D;Not Available;Normal)']=='Not Available')

    results = results.append(
        pd.DataFrame(
            [
                [
                    'PET Read',
                    '-',
                    '-',
                    '-',
                    robjects.r('chisq.test(matrix(c(%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i),ncol=3,byrow=TRUE),correct=FALSE)'%(restricted_1,restricted_2,restricted_3,subtle_1,subtle_2,subtle_3,diffuse_multifocal_1,diffuse_multifocal_2,diffuse_multifocal_3,normal_1,normal_2,normal_3)).rx2('p.value')[0],
                    '-',
                    '-',
                    '-',
                    robjects.r('chisq.test(matrix(c(%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i),ncol=3,byrow=TRUE),correct=FALSE)'%(restricted_1_test,restricted_2_test,restricted_3_test,subtle_1_test,subtle_2_test,subtle_3_test,diffuse_multifocal_1_test,diffuse_multifocal_2_test,diffuse_multifocal_3_test,normal_1_test,normal_2_test,normal_3_test)).rx2('p.value')[0]
                ],
                [
                    'Restricted',
                    restricted_1,
                    restricted_2,
                    restricted_3,
                    '-',
                    restricted_1_test,
                    restricted_2_test,
                    restricted_3_test,
                    '-'
                ],
                [
                    'Subtle',
                    subtle_1,
                    subtle_2,
                    subtle_3,
                    '-',
                    subtle_1_test,
                    subtle_2_test,
                    subtle_3_test,
                    '-'
                ],
                [
                    'Diffuse or Multifocal',
                    diffuse_multifocal_1,
                    diffuse_multifocal_2,
                    diffuse_multifocal_3,
                    '-',
                    diffuse_multifocal_1_test,
                    diffuse_multifocal_2_test,
                    diffuse_multifocal_3_test,
                    '-'
                ],
                [
                    'Normal/Not Available',
                    normal_1,
                    normal_2,
                    normal_3,
                    '-',
                    normal_1_test,
                    normal_2_test,
                    normal_3_test,
                    '-'
                ]
            ],
            columns=columns
        ),
        ignore_index=True
    )


    return results


def classification1_feature_selection(rerun = False):
    '''
    This script runs the two step feature selection on outcome coded using Classification 1. First, it runs a feature selection using the F test (over 100 bootstraps of the training data). Second, a Random Forest based Cross-Validation feature elimination is used separately to get another set of features. Finally, the two sets are collated and only the features present in both sets are kept. This is done for the SSP-based feature set and Asymmetry-based feature set separately.
    '''
    if rerun:
        #Feature Selection for Model B

        # Parameters for feature selection
        n_repeats = 100
        num_cv = 5

        # Load data
        df = pd.read_csv('qPET_feature_matrix_clinical_3dssp.csv')
        df_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_3dssp.csv')

        # Generate feature matrix and target vectors
        X = np.array(df[df.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
        X_labels = np.array(df.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))

        # Generate output labels
        y = np.array(df.outcomeLatest)
        y[y<2] = 0
        y[y>=2] = 1

        # Run first set of selection feature using F test over bootstraps
        all_features = []
        for iter_id in range(n_repeats):
            X_resample, y_resample = sklearn.utils.resample(X,y,n_samples=int(0.7*X.shape[0]),replace=False, random_state=RANDOM_STATE)
            fdr = SelectKBest(f_classif,k=int(n_repeats/2))
            fdr.fit(X_resample, y_resample)
            for index in fdr.get_support(indices=True):
                all_features.append(index)
        all_features =  np.sort(all_features)

        ft_counts = {}
        for ft in all_features:
            try:
                ft_counts[ft] += 1
            except KeyError:
                ft_counts[ft] = 1
        SSP_best_features1 = []
        for ft,ft_count in ft_counts.items():
            if ft_count > n_repeats/2:
                SSP_best_features1.append(ft)

        # Run second set of selection feature using recursive feature elimination
        estimator = RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=RANDOM_STATE)
        selector = RFECV(estimator, step=1, cv=num_cv)
        selector = selector.fit(X,y)
        SSP_best_features2 = np.where(selector.support_)[0]

        SSP_features = list(set(SSP_best_features1) & set(SSP_best_features2))
        print sorted(SSP_features)

        #Feature Selection for Model C

        # Load data
        df = pd.read_csv('qPET_feature_matrix_clinical_voxel_ai.csv')
        df_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_voxel_ai.csv')

        # Generate feature matrix and target vectors
        X = np.array(df[df.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
        X_labels = np.array(df.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
        # Generate output labels
        y = np.array(df.outcomeLatest)
        y[y<2] = 0
        y[y>=2] = 1

        # Run first set of selection feature using F test over bootstraps
        all_features = []
        for iter_id in range(n_repeats):
            X_resample, y_resample = sklearn.utils.resample(X,y,n_samples=int(0.7*X.shape[0]),replace=False, random_state=RANDOM_STATE)
            fdr = SelectKBest(f_classif,k=int(n_repeats/2))
            fdr.fit(X_resample, y_resample)
            for index in fdr.get_support(indices=True):
                all_features.append(index)
        all_features =  np.sort(all_features)

        ft_counts = {}
        for ft in all_features:
            try:
                ft_counts[ft] += 1
            except KeyError:
                ft_counts[ft] = 1
        ASYMM_best_features1 = []
        for ft,ft_count in ft_counts.items():
            if ft_count > n_repeats/2:
                ASYMM_best_features1.append(ft)

        # Run second set of selection feature using recursive feature elimination
        estimator = RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=RANDOM_STATE)
        selector = RFECV(estimator, step=1, cv=num_cv)
        selector = selector.fit(X,y)
        ASYMM_best_features2 = np.where(selector.support_)[0]

        ASYMM_features = list(set(ASYMM_best_features1) & set(ASYMM_best_features2))
        print sorted(ASYMM_features)
    else:
        # The feature selection step is hard-coded in order to avoid long runtime. With RANDOM_STATE, the following features were selected after combination of both sets.
        SSP_features = [1, 5, 32, 33, 41, 44, 50, 51, 59, 83, 108, 109, 112, 117, 126, 127, 135, 136]
        ASYMM_features = [27, 42, 44, 49, 65, 73, 144, 217, 234, 236, 339, 341, 352]
    return SSP_features, ASYMM_features

def classification1_num_estimators(SSP_features, ASYMM_features):
    '''
    This script determines the optimal number of trees to use for each random forest classifier (Models A-C) based on outcome coded using Classification 1.
    '''
    # Load data for CLINICAL ONLY
    dfA = pd.read_csv('qPET_feature_matrix_clinical_only.csv')
    dfA_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_only.csv')

    # Generate feature matrix for training and testing data
    # Training
    X1 = np.array(dfA[dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X1_labels = np.array(dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))

    # Load data for SSP ONLY
    dfB = pd.read_csv('qPET_feature_matrix_clinical_3dssp.csv')
    dfB_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_3dssp.csv')

    # Generate feature matrix for training and testing data
    # Training
    X2 = np.array(dfB[dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X2_labels = np.array(dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X2 = X2[:,SSP_features]
    X2_labels = X2_labels[SSP_features]

    # Load data for ASYMMETRY ONLY
    dfC = pd.read_csv('qPET_feature_matrix_clinical_voxel_ai.csv')
    dfC_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_voxel_ai.csv')

    # Generate feature matrix for training testing data
    # Training
    X3 = np.array(dfC[dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X3_labels = np.array(dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X3 = X3[:,ASYMM_features]
    X3_labels = X3_labels[ASYMM_features]

    # Generate outcome variable
    outcome_threshold = 2 # i.e. Engel 1B
    y = np.array(dfA.outcomeLatest)
    y_test = np.array(dfA_test.outcomeLatest)
    y[y<outcome_threshold] = 0
    y[y>=outcome_threshold] = 1
    y_test[y_test<outcome_threshold] = 0
    y_test[y_test>=outcome_threshold] = 1

    # Generate feature matrices combine with clinical variables
    XA = X1
    XA_labels = X1_labels
    yA = y

    XB = np.hstack((X1,X2))
    XB_labels = np.hstack((X1_labels,X2_labels))
    yB = y

    yC = y
    XC = np.hstack((X1,X3))
    XC_labels = np.hstack((X1_labels,X3_labels))

    # Plot OOB score versus number of estimators
    min_estimators = 15
    max_estimators = 1000

    clfA = RandomForestClassifier(warm_start=True, max_features=None,
                                 oob_score=True, max_depth=max_depth,
                                 random_state=RANDOM_STATE)
    error_rate = []
    for i in range(min_estimators, max_estimators+1):
        clfA.set_params(n_estimators=i)
        clfA.fit(XA,yA)
        oob_error = 1-clfA.oob_score_
        error_rate.append((i, oob_error))
    error_rate = np.array(error_rate)
    plt.plot(error_rate[:,0], error_rate[:,1])
    plt.xlim(min_estimators, max_estimators)
    plt.title('OOB Error versus Number of Estimators for Clinical Only')
    plt.xlabel('n_estimators')
    plt.ylabel('OOB error rate')
    plt.show()
    nest = error_rate[:,0]
    err = error_rate[:,1]
    n_estimatorsA = int(nest[np.where(err == np.min(err))[0][0]])

    clfB = RandomForestClassifier(warm_start=True, max_features=None,
                                 oob_score=True, max_depth=max_depth,
                                 random_state=RANDOM_STATE)
    error_rate = []
    for i in range(min_estimators, max_estimators+1):
        clfB.set_params(n_estimators=i)
        clfB.fit(XB,yB)
        oob_error = 1-clfB.oob_score_
        error_rate.append((i, oob_error))
    error_rate = np.array(error_rate)
    plt.plot(error_rate[:,0], error_rate[:,1])
    plt.xlim(min_estimators, max_estimators)
    plt.title('OOB Error versus Number of Estimators for SSP and Clinical variables')
    plt.xlabel('n_estimators')
    plt.ylabel('OOB error rate')
    plt.show()
    nest = error_rate[:,0]
    err = error_rate[:,1]
    n_estimatorsB = int(nest[np.where(err == np.min(err))[0][0]])

    clfC = RandomForestClassifier(warm_start=True, max_features=None,
                                 oob_score=True, max_depth=max_depth,
                                 random_state=RANDOM_STATE)
    error_rate = []
    for i in range(min_estimators, max_estimators+1):
        clfC.set_params(n_estimators=i)
        clfC.fit(XC,yC)
        oob_error = 1-clfC.oob_score_
        error_rate.append((i, oob_error))
    error_rate = np.array(error_rate)
    plt.plot(error_rate[:,0], error_rate[:,1])
    plt.xlim(min_estimators, max_estimators)
    plt.title('OOB Error versus Number of Estimators for Asymmetry and Clinical variables')
    plt.xlabel('n_estimators')
    plt.ylabel('OOB error rate')
    plt.show()
    nest = error_rate[:,0]
    err = error_rate[:,1]
    n_estimatorsC = int(nest[np.where(err == np.min(err))[0][0]])

    return n_estimatorsA, n_estimatorsB, n_estimatorsC

def classification1_feature_importances(SSP_features, ASYMM_features):
    '''
    This script computes feature importance and performs ordinal logistic regression to 7-point outcome.
    '''
    # Create feature matrix for training and testing
    # Load data for CLINICAL ONLY
    dfA = pd.read_csv('qPET_feature_matrix_clinical_only.csv')
    dfA_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_only.csv')

    # Generate feature matrix for training and testing data
    # Training
    X1 = np.array(dfA[dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X1_labels = np.array(dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    # Testing
    X1_test = np.array(dfA_test[dfA_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X1_test_labels = np.array(dfA_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))

    # Load data for SSP ONLY
    dfB = pd.read_csv('qPET_feature_matrix_clinical_3dssp.csv')
    dfB_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_3dssp.csv')

    # Generate feature matrix for training and testing data
    # Training
    X2 = np.array(dfB[dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X2_labels = np.array(dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X2 = X2[:,SSP_features]
    X2_labels = X2_labels[SSP_features]
    # Testing
    X2_test = np.array(dfB_test[dfB_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X2_test_labels = np.array(dfB_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X2_test = X2_test[:,SSP_features]
    X2_test_labels = X2_test_labels[SSP_features]

    # Load data for ASYMMETRY ONLY
    dfC = pd.read_csv('qPET_feature_matrix_clinical_voxel_ai.csv')
    dfC_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_voxel_ai.csv')

    # Generate feature matrix for training testing data
    # Training
    X3 = np.array(dfC[dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X3_labels = np.array(dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X3 = X3[:,ASYMM_features]
    X3_labels = X3_labels[ASYMM_features]
    # Testing
    X3_test = np.array(dfC_test[dfC_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X3_test_labels = np.array(dfC_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X3_test = X3_test[:,ASYMM_features]
    X3_test_labels = X3_test_labels[ASYMM_features]

    # Generate outcome variable
    # outcome_threshold = 2 # i.e. Engel 1B
    y = np.array(dfA.outcomeLatest)
    y_test = np.array(dfA_test.outcomeLatest)
    # y[y<outcome_threshold] = 0
    # y[y>=outcome_threshold] = 1
    # y_test[y_test<outcome_threshold] = 0
    # y_test[y_test>=outcome_threshold] = 1

    # Run predictions
    # Generate feature matrix for CLINICAL ONLY
    XA = X1
    XA_labels = X1_labels
    XA_test = X1_test
    XA_test_labels = X1_test_labels
    yA = y

    # Generate feature matrix for SSP and CLINICAL
    XB = np.hstack((X1,X2))
    XB_labels = np.hstack((X1_labels,X2_labels))
    XB_test = np.hstack((X1_test,X2_test))
    XB_test_labels = np.hstack((X1_test_labels,X2_test_labels))
    yB = y

    # Generate feature matrix for SSP and CLINICAL
    XC = np.hstack((X1,X3))
    XC_labels = np.hstack((X1_labels,X3_labels))
    XC_test = np.hstack((X1_test,X3_test))
    XC_test_labels = np.hstack((X1_test_labels,X3_test_labels))
    yC = y


    robjects.globalenv['RANDOM_STATE'] = RANDOM_STATE

    # Generate final form of Table 2
    columns=[
        'Feature',
        'Feature Importance',
        'Univariate Odds Ratio per Unit\n Increase in Seizure Recurrence [95% CI, p]'
            ]
    results = pd.DataFrame(
        {'Feature':[],
         'Feature Importance':[],
         'Univariate Odds Ratio per Unit\n Increase in Seizure Recurrence [95% CI, p]':[]
                 },
        columns=columns
    )

    for X,y, X_labels in [(XA,yA,XA_labels), (XB,yB,XB_labels), (XC,yC,XC_labels)]:
        # Run feature importance (IncMSE) for random forests
        robjects.globalenv['X'] = X
        robjects.globalenv['y'] = y
        summary = robjects.r['summary']

        robjects.r('''
            set.seed(RANDOM_STATE)
            library(randomForest)
            rf <- randomForest(y ~ ., data=X, importance=TRUE)
            X_importances <- importance(rf)
        ''')
        X_importances = robjects.r['X_importances']
        feature_order = np.argsort(-X_importances[:,0])

        # Run univariate odds ratio computation
        for feature in feature_order:
            robjects.globalenv['feature'] = X[:,feature]
            robjects.r('''
                set.seed(RANDOM_STATE)
                require(MASS)
                myfit <- polr(as.factor(y) ~ feature, Hess=TRUE)
                (ctable <- coef(summary(myfit)))
                p <- pnorm(abs(ctable[,"t value"]), lower.tail=FALSE)*2
                (ctable <- cbind(ctable, "p value" = p))
                (ci <- confint(myfit))
                myfit_ci <-exp(cbind(OR = coef(myfit), ci))
            ''')
            myfit = robjects.r['myfit']
            myfit_ci = robjects.r['myfit_ci']
            pval = robjects.r['p'][0]
            label = X_labels[feature]
            try:
                label = label.replace(label.split('_')[-1],ssp_labels[int(label.split('_')[-1])])
            except Exception:
                pass

            if X_importances[feature,0] <= 0.0:
                continue

            if pval < 0.001:
                pval = '%0.3f ***'%pval
            elif pval < 0.01:
                pval = '%0.3f **'%pval
            elif pval < 0.05:
                pval = '%0.3f *'%pval
            elif pval < 0.1:
                pval = '%0.3f .'%pval
            else:
                pval = '%0.2f'%pval

            results = results.append(
                pd.DataFrame(
                    [
                        [
                            label,
                            '%0.2f'%X_importances[feature,0],
                            '%0.2f [%0.2f - %0.2f, %s]'%(
                                myfit_ci[0,0],
                                myfit_ci[0,1], myfit_ci[1,1],
                                pval
                            )
                        ]
                    ],
                    columns=columns
                ),
                ignore_index=True
            )
        results = results.append(
            pd.DataFrame(
                [
                    [
                        '',
                        '',
                        ''
                    ]
                ],
                columns=columns
            ),
            ignore_index=True
        )
    return results

def classification1_cross_validation(SSP_features, ASYMM_features, n_estimatorsA, n_estimatorsB, n_estimatorsC):
    '''
    This script performs cross validation and measures out-of-bag score on outcome coded using Classification 1.
    '''
    # Preprocessing
    classifierA = RandomForestClassifier(n_estimators=n_estimatorsA, max_depth=max_depth, random_state=RANDOM_STATE,oob_score=True)
    classifierB = RandomForestClassifier(n_estimators=n_estimatorsB, max_depth=max_depth, random_state=RANDOM_STATE, oob_score=True)
    classifierC = RandomForestClassifier(n_estimators=n_estimatorsC, max_depth=max_depth, random_state=RANDOM_STATE, oob_score=True)

    sss = StratifiedShuffleSplit(n_splits=NUM_BOOTSTRAPS, test_size=0.2, random_state=RANDOM_STATE)

    print 'Generating feature matrices ...'
    # Load data for CLINICAL ONLY
    dfA = pd.read_csv('qPET_feature_matrix_clinical_only.csv')
    dfA_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_only.csv')

    # Generate feature matrix for training and testing data
    # Training
    X1 = np.array(dfA[dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X1_labels = np.array(dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    # Testing
    X1_test = np.array(dfA_test[dfA_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X1_test_labels = np.array(dfA_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))

    # Load data for SSP ONLY
    dfB = pd.read_csv('qPET_feature_matrix_clinical_3dssp.csv')
    dfB_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_3dssp.csv')

    # Generate feature matrix for training and testing data
    # Training
    X2 = np.array(dfB[dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X2_labels = np.array(dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X2 = X2[:,SSP_features]
    X2_labels = X2_labels[SSP_features]
    # Testing
    X2_test = np.array(dfB_test[dfB_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X2_test_labels = np.array(dfB_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X2_test = X2_test[:,SSP_features]
    X2_test_labels = X2_test_labels[SSP_features]

    # Load data for ASYMMETRY ONLY
    dfC = pd.read_csv('qPET_feature_matrix_clinical_voxel_ai.csv')
    dfC_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_voxel_ai.csv')

    # Generate feature matrix for training testing data
    # Training
    X3 = np.array(dfC[dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X3_labels = np.array(dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X3 = X3[:,ASYMM_features]
    X3_labels = X3_labels[ASYMM_features]
    # Testing
    X3_test = np.array(dfC_test[dfC_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X3_test_labels = np.array(dfC_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X3_test = X3_test[:,ASYMM_features]
    X3_test_labels = X3_test_labels[ASYMM_features]

    # Generate outcome variable
    outcome_threshold = 2 # i.e. Engel 1B
    y = np.array(dfA.outcomeLatest)
    y_test = np.array(dfA_test.outcomeLatest)
    y[y<outcome_threshold] = 0
    y[y>=outcome_threshold] = 1
    y_test[y_test<outcome_threshold] = 0
    y_test[y_test>=outcome_threshold] = 1


    columns=['Clinical Variable(s)','CV 5-Fold AUC_1','Out Of Bag (OOB1) Error',
             'Quantitative Variables','CV 5-Fold AUC_2','Out Of Bag (OOB2) Error',
             'AUC Difference 95% C.I.','OOB Difference 95% C.I.'
            ]
    results = pd.DataFrame(
        {'Clinical Variable(s)':[],
         'CV 5-Fold AUC_1':[],
         'Out Of Bag (OOB1) Error':[],
         'Quantitative Variables':[],
         'CV 5-Fold AUC_2':[],
         'Out Of Bag (OOB2) Error':[],
         'AUC Difference 95% C.I.':[],
         'OOB Difference 95% C.I.':[]
                 },
        columns=columns
    )

    OPTIONS = {
        1:[np.arange(20),'EEG+MRI+PET'],
        2:[np.arange(12),'EEG'],
        3:[np.arange(12,14),'MRI'],
        4:[np.arange(13,20),'PET'],
    }

    for OPTION,(clinical_variable_idx,variables_list) in OPTIONS.items():
        # Generate feature matrix for CLINICAL ONLY
        XA = X1[:,clinical_variable_idx]
        XA_labels = X1_labels[clinical_variable_idx]
        yA = y

        # Generate feature matrix for SSP alone
        XB = X2
        XB_labels = X2_labels
        XB_test = X2_test
        XB_test_labels = X2_test_labels
        yB = y

        # Generate bootstrap jobs
        jobs = []
        out = []
        for job_iter,(train, test) in enumerate(sss.split(XA,yA)):
            jobs.append((job_iter,train,test,XA,yA,XB,yB,classifierA,classifierB))
        # Run all jobs
        return_list = pool.map(_helper,jobs)
        # Compute OOB/AUC for bootstraps
        for res in return_list:
            probas_A,yA_test,probas_B,yB_test,classifierA_oob_score_,classifierB_oob_score_ = res
            fprA, tprA, thresholds = roc_curve(yA_test, probas_A[:, 1])
            fprB, tprB, thresholds = roc_curve(yB_test, probas_B[:, 1])
            roc_aucA = auc(fprA, tprA)
            roc_aucB = auc(fprB, tprB)
            out.append([roc_aucA,roc_aucB,classifierA_oob_score_,classifierB_oob_score_])
        out21 = np.array(out)

        # Generate feature matrix for SSP and CLINICAL
        XB = np.hstack((X1,X2))
        XB_labels = np.hstack((X1_labels,X2_labels))
        XB_test = np.hstack((X1_test,X2_test))
        XB_test_labels = np.hstack((X1_test_labels,X2_test_labels))

        # Generate bootstrap jobs
        jobs = []
        out = []
        for job_iter,(train, test) in enumerate(sss.split(XA,yA)):
            jobs.append((job_iter,train,test,XA,yA,XB,yB,classifierA,classifierB))
        # Run all jobs
        return_list = pool.map(_helper,jobs)
        # Compute OOB/AUC for bootstraps
        for res in return_list:
            probas_A,yA_test,probas_B,yB_test,classifierA_oob_score_,classifierB_oob_score_ = res
            fprA, tprA, thresholds = roc_curve(yA_test, probas_A[:, 1])
            fprB, tprB, thresholds = roc_curve(yB_test, probas_B[:, 1])
            roc_aucA = auc(fprA, tprA)
            roc_aucB = auc(fprB, tprB)
            out.append([roc_aucA,roc_aucB,classifierA_oob_score_,classifierB_oob_score_])
        out31 = np.array(out)

        # Compute statistics
        AUC1 = np.mean(out21[:,0])
        OOB1 = np.mean(out21[:,2])
        AUC2 = np.mean(out21[:,1])
        OOB2 = np.mean(out21[:,3])
        AUC3 = np.mean(out31[:,1])
        OOB3 = np.mean(out31[:,3])
        CI_AUC2_minus_AUC1 = tuple(map(lambda x: np.percentile(out21[:,1] - out21[:,0],x), [2.5, 97.5]))
        CI_AUC3_minus_AUC1 = tuple(map(lambda x: np.percentile(out31[:,1] - out31[:,0],x), [2.5, 97.5]))
        CI_OOB2_minus_OOB1 = tuple(map(lambda x: np.percentile(out21[:,3] - out21[:,2],x), [2.5, 97.5]))
        CI_OOB3_minus_OOB1 = tuple(map(lambda x: np.percentile(out31[:,3] - out31[:,2],x), [2.5, 97.5]))


        results = results.append(
            pd.DataFrame(
                [
                    [variables_list,AUC1,OOB1,'%s+SSP'%(variables_list), AUC2, OOB2, CI_AUC2_minus_AUC1, CI_OOB2_minus_OOB1],
                    [variables_list,AUC1,OOB1,'SSP only', AUC3, OOB3, CI_AUC3_minus_AUC1, CI_OOB3_minus_OOB1]
                ],
                columns=columns
            ),
            ignore_index=True
        )
    resultsSSP = results

    columns=['Clinical Variable(s)','CV 5-Fold AUC_1','Out Of Bag (OOB1) Error',
             'Quantitative Variables','CV 5-Fold AUC_2','Out Of Bag (OOB2) Error',
             'AUC Difference 95% C.I.','OOB Difference 95% C.I.'
            ]
    results = pd.DataFrame(
        {'Clinical Variable(s)':[],
         'CV 5-Fold AUC_1':[],
         'Out Of Bag (OOB1) Error':[],
         'Quantitative Variables':[],
         'CV 5-Fold AUC_2':[],
         'Out Of Bag (OOB2) Error':[],
         'AUC Difference 95% C.I.':[],
         'OOB Difference 95% C.I.':[]
                 },
        columns=columns
    )

    OPTIONS = {
        1:[np.arange(20),'EEG+MRI+PET'],
        2:[np.arange(12),'EEG'],
        3:[np.arange(12,14),'MRI'],
        4:[np.arange(13,20),'PET'],
    }

    for OPTION,(clinical_variable_idx,variables_list) in OPTIONS.items():
        # Generate feature matrix for CLINICAL ONLY
        XA = X1[:,clinical_variable_idx]
        XA_labels = X1_labels[clinical_variable_idx]
        yA = y

        # Generate feature matrix for SSP alone
        XC = X3
        XC_labels = X3_labels
        XC_test = X3_test
        XC_test_labels = X3_test_labels
        yC = y

        # Generate bootstrap jobs
        jobs = []
        out = []
        for job_iter,(train, test) in enumerate(sss.split(XA,yA)):
            jobs.append((job_iter,train,test,XA,yA,XC,yC,classifierA,classifierC))
        # Run all jobs
        return_list = pool.map(_helper,jobs)
        # Compute OOB/AUC for bootstraps
        for res in return_list:
            probas_A,yA_test,probas_C,yC_test,classifierA_oob_score_,classifierC_oob_score_ = res
            fprA, tprA, thresholds = roc_curve(yA_test, probas_A[:, 1])
            fprC, tprC, thresholds = roc_curve(yC_test, probas_C[:, 1])
            roc_aucA = auc(fprA, tprA)
            roc_aucC = auc(fprC, tprC)
            out.append([roc_aucA,roc_aucC,classifierA_oob_score_,classifierC_oob_score_])
        out21 = np.array(out)

        # Generate feature matrix for SSP and CLINICAL
        XC = np.hstack((X1,X3))
        XC_labels = np.hstack((X1_labels,X3_labels))
        XC_test = np.hstack((X1_test,X3_test))
        XC_test_labels = np.hstack((X1_test_labels,X3_test_labels))

        # Generate bootstrap jobs
        jobs = []
        out = []
        for job_iter,(train, test) in enumerate(sss.split(XA,yA)):
            jobs.append((job_iter,train,test,XA,yA,XC,yB,classifierA,classifierB))
        # Run all jobs
        return_list = pool.map(_helper,jobs)
        # Compute OOB/AUC for bootstraps
        for res in return_list:
            probas_A,yA_test,probas_C,yC_test,classifierA_oob_score_,classifierC_oob_score_ = res
            fprA, tprA, thresholds = roc_curve(yA_test, probas_A[:, 1])
            fprC, tprC, thresholds = roc_curve(yC_test, probas_C[:, 1])
            roc_aucA = auc(fprA, tprA)
            roc_aucC = auc(fprC, tprC)
            out.append([roc_aucA,roc_aucC,classifierA_oob_score_,classifierC_oob_score_])
        out31 = np.array(out)

        # Compute statistics
        AUC1 = np.mean(out21[:,0])
        OOB1 = np.mean(out21[:,2])
        AUC2 = np.mean(out21[:,1])
        OOB2 = np.mean(out21[:,3])
        AUC3 = np.mean(out31[:,1])
        OOB3 = np.mean(out31[:,3])
        CI_AUC2_minus_AUC1 = tuple(map(lambda x: np.percentile(out21[:,1] - out21[:,0],x), [2.5, 97.5]))
        CI_AUC3_minus_AUC1 = tuple(map(lambda x: np.percentile(out31[:,1] - out31[:,0],x), [2.5, 97.5]))
        CI_OOB2_minus_OOB1 = tuple(map(lambda x: np.percentile(out21[:,3] - out21[:,2],x), [2.5, 97.5]))
        CI_OOB3_minus_OOB1 = tuple(map(lambda x: np.percentile(out31[:,3] - out31[:,2],x), [2.5, 97.5]))


        results = results.append(
            pd.DataFrame(
                [
                    [variables_list,AUC1,OOB1,'%s+Asymmetry'%(variables_list), AUC2, OOB2, CI_AUC2_minus_AUC1, CI_OOB2_minus_OOB1],
                    [variables_list,AUC1,OOB1,'Asymmetry only', AUC3, OOB3, CI_AUC3_minus_AUC1, CI_OOB3_minus_OOB1]
                ],
                columns=columns
            ),
            ignore_index=True
        )
    resultsASYMM = results
    return resultsSSP, resultsASYMM

def classification1_validation(SSP_features, ASYMM_features, n_estimatorsA, n_estimatorsB, n_estimatorsC):
    '''
    This script measures accuracy on the validation cohort using Classification 1.
    '''
    # Create feature matrix for training and testing
    # Load data for CLINICAL ONLY
    dfA = pd.read_csv('qPET_feature_matrix_clinical_only.csv')
    dfA_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_only.csv')

    # Generate feature matrix for training and testing data
    # Training
    X1 = np.array(dfA[dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X1_labels = np.array(dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    # Testing
    X1_test = np.array(dfA_test[dfA_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X1_test_labels = np.array(dfA_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))

    # Load data for SSP ONLY
    dfB = pd.read_csv('qPET_feature_matrix_clinical_3dssp.csv')
    dfB_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_3dssp.csv')

    # Generate feature matrix for training and testing data
    # Training
    X2 = np.array(dfB[dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X2_labels = np.array(dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X2 = X2[:,SSP_features]
    X2_labels = X2_labels[SSP_features]
    # Testing
    X2_test = np.array(dfB_test[dfB_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X2_test_labels = np.array(dfB_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X2_test = X2_test[:,SSP_features]
    X2_test_labels = X2_test_labels[SSP_features]

    # Load data for ASYMMETRY ONLY
    dfC = pd.read_csv('qPET_feature_matrix_clinical_voxel_ai.csv')
    dfC_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_voxel_ai.csv')

    # Generate feature matrix for training testing data
    # Training
    X3 = np.array(dfC[dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X3_labels = np.array(dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X3 = X3[:,ASYMM_features]
    X3_labels = X3_labels[ASYMM_features]
    # Testing
    X3_test = np.array(dfC_test[dfC_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X3_test_labels = np.array(dfC_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X3_test = X3_test[:,ASYMM_features]
    X3_test_labels = X3_test_labels[ASYMM_features]

    # Generate outcome variable
    outcome_threshold = 2 # i.e. Engel 1B
    y = np.array(dfA.outcomeLatest)
    y_test = np.array(dfA_test.outcomeLatest)
    y[y<outcome_threshold] = 0
    y[y>=outcome_threshold] = 1
    y_test[y_test<outcome_threshold] = 0
    y_test[y_test>=outcome_threshold] = 1

    # Run predictions
    # Generate feature matrix for CLINICAL ONLY
    XA = X1
    XA_labels = X1_labels
    XA_test = X1_test
    XA_test_labels = X1_test_labels

    # Generate feature matrix for SSP and CLINICAL
    XB = np.hstack((X1,X2))
    XB_labels = np.hstack((X1_labels,X2_labels))
    XB_test = np.hstack((X1_test,X2_test))
    XB_test_labels = np.hstack((X1_test_labels,X2_test_labels))

    # Generate feature matrix for SSP and CLINICAL
    XC = np.hstack((X1,X3))
    XC_labels = np.hstack((X1_labels,X3_labels))
    XC_test = np.hstack((X1_test,X3_test))
    XC_test_labels = np.hstack((X1_test_labels,X3_test_labels))

    # Load library in R
    pROC = importr('pROC')

    # Train classifier and apply to validation test set
    classifierA = RandomForestClassifier(
        n_estimators=n_estimatorsA,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        oob_score=True)
    classifierB = RandomForestClassifier(
        n_estimators=n_estimatorsB,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        oob_score=True)
    classifierC = RandomForestClassifier(
        n_estimators=n_estimatorsC,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        oob_score=True)

    probas = classifierA.fit(XA, y).predict_proba(XA_test)
    predsA = robjects.FloatVector(probas[:,1])
    labelsA = robjects.IntVector(y_test)
    probas = classifierB.fit(XB, y).predict_proba(XB_test)
    predsB = robjects.FloatVector(probas[:,1])
    labelsB = robjects.IntVector(y_test)
    probas = classifierC.fit(XC, y).predict_proba(XC_test)
    predsC = robjects.FloatVector(probas[:,1])
    labelsC = robjects.IntVector(y_test)


    # Copy from python workspace to R workspace
    robjects.globalenv["predsA"] = predsA
    robjects.globalenv["labelsA"] = labelsA
    robjects.globalenv["predsB"] = predsB
    robjects.globalenv["labelsB"] = labelsB
    robjects.globalenv["predsC"] = predsC
    robjects.globalenv["labelsC"] = labelsC

    # Run pROC.roc and pROC.roc.test in R
    robjects.r('''
        predsA<-as.numeric(unlist(predsA))
        labelsA<-as.numeric(unlist(labelsA))
        predsB<-as.numeric(unlist(predsB))
        labelsB<-as.numeric(unlist(labelsB))
        predsC<-as.numeric(unlist(predsC))
        labelsC<-as.numeric(unlist(labelsC))

        library(pROC)

        roc1 <- roc(labelsA, predsA, percent=FALSE,
        smooth=TRUE, ci=TRUE, boot.n=100, ci.alpha=0.95,
        stratified=TRUE,plot=FALSE,grid=FALSE,print.auc=FALSE, show.thres=TRUE, col='red')
        roc2 <- roc(labelsB, predsB, percent=FALSE,
        smooth=TRUE, ci=TRUE, boot.n=100, ci.alpha=0.95,
        stratified=TRUE,plot=FALSE,grid=FALSE,print.auc=FALSE, show.thres=TRUE, col='red')
        roc3 <- roc(labelsC, predsC, percent=FALSE,
        smooth=TRUE, ci=TRUE, boot.n=100, ci.alpha=0.95,
        stratified=TRUE,plot=FALSE,grid=FALSE,print.auc=FALSE, show.thres=TRUE, col='red')
        roc_test12<-roc.test(roc1,roc2)
        roc_test13<-roc.test(roc1,roc3)
        test_accuracy1<-coords(roc1, "best", ret=c("specificity","sensitivity","accuracy"), best.method=c("youden","closest.topleft"));
        test_accuracy2<-coords(roc2, "best", ret=c("specificity","sensitivity","accuracy"), best.method=c("youden","closest.topleft"));
        test_accuracy3<-coords(roc3, "best", ret=c("specificity","sensitivity","accuracy"), best.method=c("youden","closest.topleft"));
    ''')

    # Generate final form of Table 4
    columns=['Model','AUC','Accuracy','Sensitivity','Specificity','p-value'
            ]
    results = pd.DataFrame(
        {'Model':[],
         'AUC':[],
         'Accuracy':[],
         'Sensitivity':[],
         'Specificity':[],
         'p-value':[]
                 },
        columns=columns
    )

    results = results.append(
        pd.DataFrame(
            [
                [
                    'Model 1',
                    robjects.r["roc1"].rx2('auc')[0],
                    robjects.r["test_accuracy1"][2],
                    robjects.r["test_accuracy1"][1],
                    robjects.r["test_accuracy1"][0],
                    np.nan
                ],
                [
                    'Model 2',
                    robjects.r["roc2"].rx2('auc')[0],
                    robjects.r["test_accuracy2"][2],
                    robjects.r["test_accuracy2"][1],
                    robjects.r["test_accuracy2"][0],
                    robjects.r["roc_test12"].rx2('p.value')[0]
                ],
                [
                    'Model 3',
                    robjects.r["roc3"].rx2('auc')[0],
                    robjects.r["test_accuracy3"][2],
                    robjects.r["test_accuracy3"][1],
                    robjects.r["test_accuracy3"][0],
                    robjects.r["roc_test13"].rx2('p.value')[0]
                ]
            ],
            columns=columns
        ),
        ignore_index=True
    )
    return results


###
def classification2_feature_selection(rerun = False):
    '''
    This script runs the two step feature selection on outcome coded using Classification 2. First, it runs a feature selection using the F test (over 100 bootstraps of the training data). Second, a Random Forest based Cross-Validation feature elimination is used separately to get another set of features. Finally, the two sets are collated and only the features present in both sets are kept. This is done for the SSP-based feature set and Asymmetry-based feature set separately.
    '''
    if rerun:
        #Feature Selection for Model B

        # Parameters for feature selection
        n_repeats = 100
        num_cv = 5

        # Load data
        df = pd.read_csv('qPET_feature_matrix_clinical_3dssp.csv')
        df_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_3dssp.csv')

        # Generate feature matrix and target vectors
        X = np.array(df[df.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
        X_labels = np.array(df.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))

        # Generate output labels; Note Engel II - IV is considered poor surgical outcome
        y = np.array(df.outcomeLatest)
        y[y<5] = 0
        y[y>=5] = 1

        # Run first set of selection feature using F test over bootstraps
        all_features = []
        for iter_id in range(n_repeats):
            X_resample, y_resample = sklearn.utils.resample(X,y,n_samples=int(0.7*X.shape[0]),replace=False, random_state=RANDOM_STATE)
            fdr = SelectKBest(f_classif,k=int(n_repeats/2))
            fdr.fit(X_resample, y_resample)
            for index in fdr.get_support(indices=True):
                all_features.append(index)
        all_features =  np.sort(all_features)

        ft_counts = {}
        for ft in all_features:
            try:
                ft_counts[ft] += 1
            except KeyError:
                ft_counts[ft] = 1
        SSP_best_features1 = []
        for ft,ft_count in ft_counts.items():
            if ft_count > n_repeats/2:
                SSP_best_features1.append(ft)

        # Run second set of selection feature using recursive feature elimination
        estimator = RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=RANDOM_STATE)
        selector = RFECV(estimator, step=1, cv=num_cv)
        selector = selector.fit(X,y)
        SSP_best_features2 = np.where(selector.support_)[0]

        SSP_features = list(set(SSP_best_features1) & set(SSP_best_features2))
        print sorted(SSP_features)


        #Feature Selection for Model C

        # Load data
        df = pd.read_csv('qPET_feature_matrix_clinical_voxel_ai.csv')
        df_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_voxel_ai.csv')

        # Generate feature matrix and target vectors
        X = np.array(df[df.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
        X_labels = np.array(df.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
        # Generate output labels
        y = np.array(df.outcomeLatest)
        y[y<5] = 0
        y[y>=5] = 1

        # Run first set of selection feature using F test over bootstraps
        all_features = []
        for iter_id in range(n_repeats):
            X_resample, y_resample = sklearn.utils.resample(X,y,n_samples=int(0.7*X.shape[0]),replace=False, random_state=RANDOM_STATE)
            fdr = SelectKBest(f_classif,k=int(n_repeats/2))
            fdr.fit(X_resample, y_resample)
            for index in fdr.get_support(indices=True):
                all_features.append(index)
        all_features =  np.sort(all_features)

        ft_counts = {}
        for ft in all_features:
            try:
                ft_counts[ft] += 1
            except KeyError:
                ft_counts[ft] = 1
        ASYMM_best_features1 = []
        for ft,ft_count in ft_counts.items():
            if ft_count > n_repeats/2:
                ASYMM_best_features1.append(ft)

        # Run second set of selection feature using recursive feature elimination
        estimator = RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=RANDOM_STATE)
        selector = RFECV(estimator, step=1, cv=num_cv)
        selector = selector.fit(X,y)
        ASYMM_best_features2 = np.where(selector.support_)[0]

        ASYMM_features = list(set(ASYMM_best_features1) & set(ASYMM_best_features2))
        print sorted(ASYMM_features)
    else:
        # The feature selection step is hard-coded in order to avoid long runtime. With RANDOM_STATE, the following features were selected after combination of both sets.
        SSP_features = [0, 4, 6, 14, 15, 16, 18, 19, 20, 21, 22, 28, 29, 32, 36, 37, 43, 51, 52, 59, 60, 62, 78, 81, 82, 84, 85, 96, 97, 98, 105, 108, 112, 113, 119, 127, 128, 135, 136, 138, 140, 141, 142, 143]
        ASYMM_features = [43, 235]
    return SSP_features, ASYMM_features

def classification2_num_estimators(SSP_features, ASYMM_features):
    '''
    This script determines the optimal number of trees to use for each random forest classifier (Models A-C) based on outcome coded using Classification 2.
    '''
    # Load data for CLINICAL ONLY
    dfA = pd.read_csv('qPET_feature_matrix_clinical_only.csv')
    dfA_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_only.csv')

    # Generate feature matrix for training and testing data
    # Training
    X1 = np.array(dfA[dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X1_labels = np.array(dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))

    # Load data for SSP ONLY
    dfB = pd.read_csv('qPET_feature_matrix_clinical_3dssp.csv')
    dfB_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_3dssp.csv')

    # Generate feature matrix for training and testing data
    # Training
    X2 = np.array(dfB[dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X2_labels = np.array(dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X2 = X2[:,SSP_features]
    X2_labels = X2_labels[SSP_features]

    # Load data for ASYMMETRY ONLY
    dfC = pd.read_csv('qPET_feature_matrix_clinical_voxel_ai.csv')
    dfC_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_voxel_ai.csv')

    # Generate feature matrix for training testing data
    # Training
    X3 = np.array(dfC[dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X3_labels = np.array(dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X3 = X3[:,ASYMM_features]
    X3_labels = X3_labels[ASYMM_features]

    # Generate outcome variable
    outcome_threshold = 5 # i.e. Engel 1B
    y = np.array(dfA.outcomeLatest)
    y_test = np.array(dfA_test.outcomeLatest)
    y[y<outcome_threshold] = 0
    y[y>=outcome_threshold] = 1
    y_test[y_test<outcome_threshold] = 0
    y_test[y_test>=outcome_threshold] = 1

    # Generate feature matrices combined with clinical variables
    XA = X1
    XA_labels = X1_labels
    yA = y

    XB = np.hstack((X1,X2))
    XB_labels = np.hstack((X1_labels,X2_labels))
    yB = y

    yC = y
    XC = np.hstack((X1,X3))
    XC_labels = np.hstack((X1_labels,X3_labels))

    # Plot OOB score versus number of estimators
    min_estimators = 15
    max_estimators = 1000

    clfA = RandomForestClassifier(warm_start=True, max_features=None,
                                 oob_score=True, max_depth=max_depth,
                                 random_state=RANDOM_STATE)
    error_rate = []
    for i in range(min_estimators, max_estimators+1):
        clfA.set_params(n_estimators=i)
        clfA.fit(XA,yA)
        oob_error = 1-clfA.oob_score_
        error_rate.append((i, oob_error))
    error_rate = np.array(error_rate)
    plt.plot(error_rate[:,0], error_rate[:,1])
    plt.xlim(min_estimators, max_estimators)
    plt.title('OOB Error versus Number of Estimators for Clinical Only')
    plt.xlabel('n_estimators')
    plt.ylabel('OOB error rate')
    plt.show()
    nest = error_rate[:,0]
    err = error_rate[:,1]
    n_estimatorsA = int(nest[np.where(err == np.min(err))[0][0]])

    clfB = RandomForestClassifier(warm_start=True, max_features=None,
                                 oob_score=True, max_depth=max_depth,
                                 random_state=RANDOM_STATE)
    error_rate = []
    for i in range(min_estimators, max_estimators+1):
        clfB.set_params(n_estimators=i)
        clfB.fit(XB,yB)
        oob_error = 1-clfB.oob_score_
        error_rate.append((i, oob_error))
    error_rate = np.array(error_rate)
    plt.plot(error_rate[:,0], error_rate[:,1])
    plt.xlim(min_estimators, max_estimators)
    plt.title('OOB Error versus Number of Estimators for SSP and Clinical variables')
    plt.xlabel('n_estimators')
    plt.ylabel('OOB error rate')
    plt.show()
    nest = error_rate[:,0]
    err = error_rate[:,1]
    n_estimatorsB = int(nest[np.where(err == np.min(err))[0][0]])

    clfC = RandomForestClassifier(warm_start=True, max_features=None,
                                 oob_score=True, max_depth=max_depth,
                                 random_state=RANDOM_STATE)
    error_rate = []
    for i in range(min_estimators, max_estimators+1):
        clfC.set_params(n_estimators=i)
        clfC.fit(XC,yC)
        oob_error = 1-clfC.oob_score_
        error_rate.append((i, oob_error))
    error_rate = np.array(error_rate)
    plt.plot(error_rate[:,0], error_rate[:,1])
    plt.xlim(min_estimators, max_estimators)
    plt.title('OOB Error versus Number of Estimators for Asymmetry and Clinical variables')
    plt.xlabel('n_estimators')
    plt.ylabel('OOB error rate')
    plt.show()
    nest = error_rate[:,0]
    err = error_rate[:,1]
    n_estimatorsC = int(nest[np.where(err == np.min(err))[0][0]])

    return n_estimatorsA, n_estimatorsB, n_estimatorsC

def classification2_feature_importances(SSP_features, ASYMM_features):
    '''
    This script computes feature importance and performs ordinal logistic regression to 7-point outcome.
    '''
    # Create feature matrix for training and testing
    # Load data for CLINICAL ONLY
    dfA = pd.read_csv('qPET_feature_matrix_clinical_only.csv')
    dfA_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_only.csv')

    # Generate feature matrix for training and testing data
    # Training
    X1 = np.array(dfA[dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X1_labels = np.array(dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    # Testing
    X1_test = np.array(dfA_test[dfA_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X1_test_labels = np.array(dfA_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))

    # Load data for SSP ONLY
    dfB = pd.read_csv('qPET_feature_matrix_clinical_3dssp.csv')
    dfB_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_3dssp.csv')

    # Generate feature matrix for training and testing data
    # Training
    X2 = np.array(dfB[dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X2_labels = np.array(dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X2 = X2[:,SSP_features]
    X2_labels = X2_labels[SSP_features]
    # Testing
    X2_test = np.array(dfB_test[dfB_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X2_test_labels = np.array(dfB_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X2_test = X2_test[:,SSP_features]
    X2_test_labels = X2_test_labels[SSP_features]

    # Load data for ASYMMETRY ONLY
    dfC = pd.read_csv('qPET_feature_matrix_clinical_voxel_ai.csv')
    dfC_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_voxel_ai.csv')

    # Generate feature matrix for training testing data
    # Training
    X3 = np.array(dfC[dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X3_labels = np.array(dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X3 = X3[:,ASYMM_features]
    X3_labels = X3_labels[ASYMM_features]
    # Testing
    X3_test = np.array(dfC_test[dfC_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X3_test_labels = np.array(dfC_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X3_test = X3_test[:,ASYMM_features]
    X3_test_labels = X3_test_labels[ASYMM_features]

    # Generate outcome variable
    outcome_threshold = 5 # i.e. Engel 1B
    y = np.array(dfA.outcomeLatest)
    y_test = np.array(dfA_test.outcomeLatest)
    # y[y<outcome_threshold] = 0
    # y[y>=outcome_threshold] = 1
    # y_test[y_test<outcome_threshold] = 0
    # y_test[y_test>=outcome_threshold] = 1

    # Run predictions
    # Generate feature matrix for CLINICAL ONLY
    XA = X1
    XA_labels = X1_labels
    XA_test = X1_test
    XA_test_labels = X1_test_labels
    yA = y

    # Generate feature matrix for SSP and CLINICAL
    XB = np.hstack((X1,X2))
    XB_labels = np.hstack((X1_labels,X2_labels))
    XB_test = np.hstack((X1_test,X2_test))
    XB_test_labels = np.hstack((X1_test_labels,X2_test_labels))
    yB = y

    # Generate feature matrix for SSP and CLINICAL
    XC = np.hstack((X1,X3))
    XC_labels = np.hstack((X1_labels,X3_labels))
    XC_test = np.hstack((X1_test,X3_test))
    XC_test_labels = np.hstack((X1_test_labels,X3_test_labels))
    yC = y

    robjects.globalenv['RANDOM_STATE'] = RANDOM_STATE

    # Generate final form of Table 2
    columns=[
        'Feature',
        'Feature Importance',
        'Univariate Odds Ratio per Unit\n Increase in Seizure Recurrence [95% CI, p]'
            ]
    results = pd.DataFrame(
        {'Feature':[],
         'Feature Importance':[],
         'Univariate Odds Ratio per Unit\n Increase in Seizure Recurrence [95% CI, p]':[]
                 },
        columns=columns
    )

    for X,y, X_labels in [(XA,yA,XA_labels), (XB,yB,XB_labels), (XC,yC,XC_labels)]:
        # Run feature importance (IncMSE) for random forests
        robjects.globalenv['X'] = X
        robjects.globalenv['y'] = y
        summary = robjects.r['summary']

        robjects.r('''
            set.seed(RANDOM_STATE)
            library(randomForest)
            rf <- randomForest(y ~ ., data=X, importance=TRUE)
            X_importances <- importance(rf)
        ''')
        X_importances = robjects.r['X_importances']
        feature_order = np.argsort(-X_importances[:,0])

        # Run univariate odds ratio computation
        for feature in feature_order:
            robjects.globalenv['feature'] = X[:,feature]
            robjects.r('''
                set.seed(RANDOM_STATE)
                require(MASS)
                myfit <- polr(as.factor(y) ~ feature, Hess=TRUE)
                (ctable <- coef(summary(myfit)))
                p <- pnorm(abs(ctable[,"t value"]), lower.tail=FALSE)*2
                (ctable <- cbind(ctable, "p value" = p))
                (ci <- confint(myfit))
                myfit_ci <-exp(cbind(OR = coef(myfit), ci))
            ''')
            myfit = robjects.r['myfit']
            myfit_ci = robjects.r['myfit_ci']
            pval = robjects.r['p'][0]
            label = X_labels[feature]
            try:
                label = label.replace(label.split('_')[-1],ssp_labels[int(label.split('_')[-1])])
            except Exception:
                pass

            if X_importances[feature,0] <= 0.0:
                continue

            if pval < 0.001:
                pval = '%0.3f ***'%pval
            elif pval < 0.01:
                pval = '%0.3f **'%pval
            elif pval < 0.05:
                pval = '%0.3f *'%pval
            elif pval < 0.1:
                pval = '%0.3f .'%pval
            else:
                pval = '%0.2f'%pval

            results = results.append(
                pd.DataFrame(
                    [
                        [
                            label,
                            '%0.2f'%X_importances[feature,0],
                            '%0.2f [%0.2f - %0.2f, %s]'%(
                                myfit_ci[0,0],
                                myfit_ci[0,1], myfit_ci[1,1],
                                pval
                            )
                        ]
                    ],
                    columns=columns
                ),
                ignore_index=True
            )
        results = results.append(
            pd.DataFrame(
                [
                    [
                        '',
                        '',
                        ''
                    ]
                ],
                columns=columns
            ),
            ignore_index=True
        )
    return results

def classification2_cross_validation(SSP_features, ASYMM_features, n_estimatorsA, n_estimatorsB, n_estimatorsC):
    '''
    This script performs cross validation and measures out-of-bag score on outcome coded using Classification 2.
    '''
    # Preprocessing
    classifierA = RandomForestClassifier(n_estimators=n_estimatorsA, max_depth=max_depth, random_state=RANDOM_STATE,oob_score=True)
    classifierB = RandomForestClassifier(n_estimators=n_estimatorsB, max_depth=max_depth, random_state=RANDOM_STATE, oob_score=True)
    classifierC = RandomForestClassifier(n_estimators=n_estimatorsC, max_depth=max_depth, random_state=RANDOM_STATE, oob_score=True)

    sss = StratifiedShuffleSplit(n_splits=NUM_BOOTSTRAPS, test_size=0.2, random_state=RANDOM_STATE)

    print 'Generating feature matrices ...'
    # Load data for CLINICAL ONLY
    dfA = pd.read_csv('qPET_feature_matrix_clinical_only.csv')
    dfA_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_only.csv')

    # Generate feature matrix for training and testing data
    # Training
    X1 = np.array(dfA[dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X1_labels = np.array(dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    # Testing
    X1_test = np.array(dfA_test[dfA_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X1_test_labels = np.array(dfA_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))

    # Load data for SSP ONLY
    dfB = pd.read_csv('qPET_feature_matrix_clinical_3dssp.csv')
    dfB_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_3dssp.csv')

    # Generate feature matrix for training and testing data
    # Training
    X2 = np.array(dfB[dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X2_labels = np.array(dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X2 = X2[:,SSP_features]
    X2_labels = X2_labels[SSP_features]
    # Testing
    X2_test = np.array(dfB_test[dfB_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X2_test_labels = np.array(dfB_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X2_test = X2_test[:,SSP_features]
    X2_test_labels = X2_test_labels[SSP_features]

    # Load data for ASYMMETRY ONLY
    dfC = pd.read_csv('qPET_feature_matrix_clinical_voxel_ai.csv')
    dfC_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_voxel_ai.csv')

    # Generate feature matrix for training testing data
    # Training
    X3 = np.array(dfC[dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X3_labels = np.array(dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X3 = X3[:,ASYMM_features]
    X3_labels = X3_labels[ASYMM_features]
    # Testing
    X3_test = np.array(dfC_test[dfC_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X3_test_labels = np.array(dfC_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X3_test = X3_test[:,ASYMM_features]
    X3_test_labels = X3_test_labels[ASYMM_features]

    # Generate outcome variable
    outcome_threshold = 5 # i.e. Engel 1B
    y = np.array(dfA.outcomeLatest)
    y_test = np.array(dfA_test.outcomeLatest)
    y[y<outcome_threshold] = 0
    y[y>=outcome_threshold] = 1
    y_test[y_test<outcome_threshold] = 0
    y_test[y_test>=outcome_threshold] = 1


    columns=['Clinical Variable(s)','CV 5-Fold AUC_1','Out Of Bag (OOB1) Error',
             'Quantitative Variables','CV 5-Fold AUC_2','Out Of Bag (OOB2) Error',
             'AUC Difference 95% C.I.','OOB Difference 95% C.I.'
            ]
    results = pd.DataFrame(
        {'Clinical Variable(s)':[],
         'CV 5-Fold AUC_1':[],
         'Out Of Bag (OOB1) Error':[],
         'Quantitative Variables':[],
         'CV 5-Fold AUC_2':[],
         'Out Of Bag (OOB2) Error':[],
         'AUC Difference 95% C.I.':[],
         'OOB Difference 95% C.I.':[]
                 },
        columns=columns
    )

    OPTIONS = {
        1:[np.arange(20),'EEG+MRI+PET'],
        2:[np.arange(12),'EEG'],
        3:[np.arange(12,14),'MRI'],
        4:[np.arange(13,20),'PET'],
    }

    for OPTION,(clinical_variable_idx,variables_list) in OPTIONS.items():
        # Generate feature matrix for CLINICAL ONLY
        XA = X1[:,clinical_variable_idx]
        XA_labels = X1_labels[clinical_variable_idx]
        yA = y

        # Generate feature matrix for SSP alone
        XB = X2
        XB_labels = X2_labels
        XB_test = X2_test
        XB_test_labels = X2_test_labels
        yB = y

        # Generate bootstrap jobs
        jobs = []
        out = []
        for job_iter,(train, test) in enumerate(sss.split(XA,yA)):
            jobs.append((job_iter,train,test,XA,yA,XB,yB,classifierA,classifierB))
        # Run all jobs
        return_list = pool.map(_helper,jobs)
        # Compute OOB/AUC for bootstraps
        for res in return_list:
            probas_A,yA_test,probas_B,yB_test,classifierA_oob_score_,classifierB_oob_score_ = res
            fprA, tprA, thresholds = roc_curve(yA_test, probas_A[:, 1])
            fprB, tprB, thresholds = roc_curve(yB_test, probas_B[:, 1])
            roc_aucA = auc(fprA, tprA)
            roc_aucB = auc(fprB, tprB)
            out.append([roc_aucA,roc_aucB,classifierA_oob_score_,classifierB_oob_score_])
        out21 = np.array(out)

        # Generate feature matrix for SSP and CLINICAL
        XB = np.hstack((X1,X2))
        XB_labels = np.hstack((X1_labels,X2_labels))
        XB_test = np.hstack((X1_test,X2_test))
        XB_test_labels = np.hstack((X1_test_labels,X2_test_labels))

        # Generate bootstrap jobs
        jobs = []
        out = []
        for job_iter,(train, test) in enumerate(sss.split(XA,yA)):
            jobs.append((job_iter,train,test,XA,yA,XB,yB,classifierA,classifierB))
        # Run all jobs
        return_list = pool.map(_helper,jobs)
        # Compute OOB/AUC for bootstraps
        for res in return_list:
            probas_A,yA_test,probas_B,yB_test,classifierA_oob_score_,classifierB_oob_score_ = res
            fprA, tprA, thresholds = roc_curve(yA_test, probas_A[:, 1])
            fprB, tprB, thresholds = roc_curve(yB_test, probas_B[:, 1])
            roc_aucA = auc(fprA, tprA)
            roc_aucB = auc(fprB, tprB)
            out.append([roc_aucA,roc_aucB,classifierA_oob_score_,classifierB_oob_score_])
        out31 = np.array(out)

        # Compute statistics
        AUC1 = np.mean(out21[:,0])
        OOB1 = np.mean(out21[:,2])
        AUC2 = np.mean(out21[:,1])
        OOB2 = np.mean(out21[:,3])
        AUC3 = np.mean(out31[:,1])
        OOB3 = np.mean(out31[:,3])
        CI_AUC2_minus_AUC1 = tuple(map(lambda x: np.percentile(out21[:,1] - out21[:,0],x), [2.5, 97.5]))
        CI_AUC3_minus_AUC1 = tuple(map(lambda x: np.percentile(out31[:,1] - out31[:,0],x), [2.5, 97.5]))
        CI_OOB2_minus_OOB1 = tuple(map(lambda x: np.percentile(out21[:,3] - out21[:,2],x), [2.5, 97.5]))
        CI_OOB3_minus_OOB1 = tuple(map(lambda x: np.percentile(out31[:,3] - out31[:,2],x), [2.5, 97.5]))


        results = results.append(
            pd.DataFrame(
                [
                    [variables_list,AUC1,OOB1,'%s+SSP'%(variables_list), AUC2, OOB2, CI_AUC2_minus_AUC1, CI_OOB2_minus_OOB1],
                    [variables_list,AUC1,OOB1,'SSP only', AUC3, OOB3, CI_AUC3_minus_AUC1, CI_OOB3_minus_OOB1]
                ],
                columns=columns
            ),
            ignore_index=True
        )
        resultsSSP = results

    columns=['Clinical Variable(s)','CV 5-Fold AUC_1','Out Of Bag (OOB1) Error',
             'Quantitative Variables','CV 5-Fold AUC_2','Out Of Bag (OOB2) Error',
             'AUC Difference 95% C.I.','OOB Difference 95% C.I.'
            ]
    results = pd.DataFrame(
        {'Clinical Variable(s)':[],
         'CV 5-Fold AUC_1':[],
         'Out Of Bag (OOB1) Error':[],
         'Quantitative Variables':[],
         'CV 5-Fold AUC_2':[],
         'Out Of Bag (OOB2) Error':[],
         'AUC Difference 95% C.I.':[],
         'OOB Difference 95% C.I.':[]
                 },
        columns=columns
    )

    OPTIONS = {
        1:[np.arange(20),'EEG+MRI+PET'],
        2:[np.arange(12),'EEG'],
        3:[np.arange(12,14),'MRI'],
        4:[np.arange(13,20),'PET'],
    }

    for OPTION,(clinical_variable_idx,variables_list) in OPTIONS.items():
        # Generate feature matrix for CLINICAL ONLY
        XA = X1[:,clinical_variable_idx]
        XA_labels = X1_labels[clinical_variable_idx]
        yA = y

        # Generate feature matrix for SSP alone
        XC = X3
        XC_labels = X3_labels
        XC_test = X3_test
        XC_test_labels = X3_test_labels
        yC = y

        # Generate bootstrap jobs
        jobs = []
        out = []
        for job_iter,(train, test) in enumerate(sss.split(XA,yA)):
            jobs.append((job_iter,train,test,XA,yA,XC,yC,classifierA,classifierC))
        # Run all jobs
        return_list = pool.map(_helper,jobs)
        # Compute OOB/AUC for bootstraps
        for res in return_list:
            probas_A,yA_test,probas_C,yC_test,classifierA_oob_score_,classifierC_oob_score_ = res
            fprA, tprA, thresholds = roc_curve(yA_test, probas_A[:, 1])
            fprC, tprC, thresholds = roc_curve(yC_test, probas_C[:, 1])
            roc_aucA = auc(fprA, tprA)
            roc_aucC = auc(fprC, tprC)
            out.append([roc_aucA,roc_aucC,classifierA_oob_score_,classifierC_oob_score_])
        out21 = np.array(out)

        # Generate feature matrix for SSP and CLINICAL
        XC = np.hstack((X1,X3))
        XC_labels = np.hstack((X1_labels,X3_labels))
        XC_test = np.hstack((X1_test,X3_test))
        XC_test_labels = np.hstack((X1_test_labels,X3_test_labels))

        # Generate bootstrap jobs
        jobs = []
        out = []
        for job_iter,(train, test) in enumerate(sss.split(XA,yA)):
            jobs.append((job_iter,train,test,XA,yA,XC,yB,classifierA,classifierB))
        # Run all jobs
        return_list = pool.map(_helper,jobs)
        # Compute OOB/AUC for bootstraps
        for res in return_list:
            probas_A,yA_test,probas_C,yC_test,classifierA_oob_score_,classifierC_oob_score_ = res
            fprA, tprA, thresholds = roc_curve(yA_test, probas_A[:, 1])
            fprC, tprC, thresholds = roc_curve(yC_test, probas_C[:, 1])
            roc_aucA = auc(fprA, tprA)
            roc_aucC = auc(fprC, tprC)
            out.append([roc_aucA,roc_aucC,classifierA_oob_score_,classifierC_oob_score_])
        out31 = np.array(out)

        # Compute statistics
        AUC1 = np.mean(out21[:,0])
        OOB1 = np.mean(out21[:,2])
        AUC2 = np.mean(out21[:,1])
        OOB2 = np.mean(out21[:,3])
        AUC3 = np.mean(out31[:,1])
        OOB3 = np.mean(out31[:,3])
        CI_AUC2_minus_AUC1 = tuple(map(lambda x: np.percentile(out21[:,1] - out21[:,0],x), [2.5, 97.5]))
        CI_AUC3_minus_AUC1 = tuple(map(lambda x: np.percentile(out31[:,1] - out31[:,0],x), [2.5, 97.5]))
        CI_OOB2_minus_OOB1 = tuple(map(lambda x: np.percentile(out21[:,3] - out21[:,2],x), [2.5, 97.5]))
        CI_OOB3_minus_OOB1 = tuple(map(lambda x: np.percentile(out31[:,3] - out31[:,2],x), [2.5, 97.5]))


        results = results.append(
            pd.DataFrame(
                [
                    [variables_list,AUC1,OOB1,'%s+Asymmetry'%(variables_list), AUC2, OOB2, CI_AUC2_minus_AUC1, CI_OOB2_minus_OOB1],
                    [variables_list,AUC1,OOB1,'Asymmetry only', AUC3, OOB3, CI_AUC3_minus_AUC1, CI_OOB3_minus_OOB1]
                ],
                columns=columns
            ),
            ignore_index=True
        )
    resultsASYMM = results
    return resultsSSP, resultsASYMM

def classification2_validation(SSP_features, ASYMM_features, n_estimatorsA, n_estimatorsB, n_estimatorsC):
    '''
    This script measures accuracy on the validation cohort using Classification 2.
    '''
    # Create feature matrix for training and testing
    # Load data for CLINICAL ONLY
    dfA = pd.read_csv('qPET_feature_matrix_clinical_only.csv')
    dfA_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_only.csv')

    # Generate feature matrix for training and testing data
    # Training
    X1 = np.array(dfA[dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X1_labels = np.array(dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    # Testing
    X1_test = np.array(dfA_test[dfA_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X1_test_labels = np.array(dfA_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))

    # Load data for SSP ONLY
    dfB = pd.read_csv('qPET_feature_matrix_clinical_3dssp.csv')
    dfB_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_3dssp.csv')

    # Generate feature matrix for training and testing data
    # Training
    X2 = np.array(dfB[dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X2_labels = np.array(dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X2 = X2[:,SSP_features]
    X2_labels = X2_labels[SSP_features]
    # Testing
    X2_test = np.array(dfB_test[dfB_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X2_test_labels = np.array(dfB_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X2_test = X2_test[:,SSP_features]
    X2_test_labels = X2_test_labels[SSP_features]

    # Load data for ASYMMETRY ONLY
    dfC = pd.read_csv('qPET_feature_matrix_clinical_voxel_ai.csv')
    dfC_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_voxel_ai.csv')

    # Generate feature matrix for training testing data
    # Training
    X3 = np.array(dfC[dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X3_labels = np.array(dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X3 = X3[:,ASYMM_features]
    X3_labels = X3_labels[ASYMM_features]
    # Testing
    X3_test = np.array(dfC_test[dfC_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X3_test_labels = np.array(dfC_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X3_test = X3_test[:,ASYMM_features]
    X3_test_labels = X3_test_labels[ASYMM_features]

    # Generate outcome variable
    outcome_threshold = 5 # i.e. Engel 1B
    y = np.array(dfA.outcomeLatest)
    y_test = np.array(dfA_test.outcomeLatest)
    y[y<outcome_threshold] = 0
    y[y>=outcome_threshold] = 1
    y_test[y_test<outcome_threshold] = 0
    y_test[y_test>=outcome_threshold] = 1

    # Run predictions
    # Generate feature matrix for CLINICAL ONLY
    XA = X1
    XA_labels = X1_labels
    XA_test = X1_test
    XA_test_labels = X1_test_labels

    # Generate feature matrix for SSP and CLINICAL
    XB = np.hstack((X1,X2))
    XB_labels = np.hstack((X1_labels,X2_labels))
    XB_test = np.hstack((X1_test,X2_test))
    XB_test_labels = np.hstack((X1_test_labels,X2_test_labels))

    # Generate feature matrix for SSP and CLINICAL
    XC = np.hstack((X1,X3))
    XC_labels = np.hstack((X1_labels,X3_labels))
    XC_test = np.hstack((X1_test,X3_test))
    XC_test_labels = np.hstack((X1_test_labels,X3_test_labels))

    # Load library in R
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    pROC = importr('pROC')

    # Train classifier and apply to validation test set
    classifierA = RandomForestClassifier(
        n_estimators=n_estimatorsA,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        oob_score=True)
    classifierB = RandomForestClassifier(
        n_estimators=n_estimatorsB,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        oob_score=True)
    classifierC = RandomForestClassifier(
        n_estimators=n_estimatorsC,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        oob_score=True)

    probas = classifierA.fit(XA, y).predict_proba(XA_test)
    predsA = robjects.FloatVector(probas[:,1])
    labelsA = robjects.IntVector(y_test)
    probas = classifierB.fit(XB, y).predict_proba(XB_test)
    predsB = robjects.FloatVector(probas[:,1])
    labelsB = robjects.IntVector(y_test)
    probas = classifierC.fit(XC, y).predict_proba(XC_test)
    predsC = robjects.FloatVector(probas[:,1])
    labelsC = robjects.IntVector(y_test)


    # Copy from python workspace to R workspace
    robjects.globalenv["predsA"] = predsA
    robjects.globalenv["labelsA"] = labelsA
    robjects.globalenv["predsB"] = predsB
    robjects.globalenv["labelsB"] = labelsB
    robjects.globalenv["predsC"] = predsC
    robjects.globalenv["labelsC"] = labelsC

    # Run pROC.roc and pROC.roc.test in R
    robjects.r('''
        predsA<-as.numeric(unlist(predsA))
        labelsA<-as.numeric(unlist(labelsA))
        predsB<-as.numeric(unlist(predsB))
        labelsB<-as.numeric(unlist(labelsB))
        predsC<-as.numeric(unlist(predsC))
        labelsC<-as.numeric(unlist(labelsC))

        library(pROC)

        roc1 <- roc(labelsA, predsA, percent=FALSE,
        smooth=TRUE, ci=TRUE, boot.n=100, ci.alpha=0.95,
        stratified=TRUE,plot=FALSE,grid=FALSE,print.auc=FALSE, show.thres=TRUE, col='red')
        roc2 <- roc(labelsB, predsB, percent=FALSE,
        smooth=TRUE, ci=TRUE, boot.n=100, ci.alpha=0.95,
        stratified=TRUE,plot=FALSE,grid=FALSE,print.auc=FALSE, show.thres=TRUE, col='red')
        roc3 <- roc(labelsC, predsC, percent=FALSE,
        smooth=TRUE, ci=TRUE, boot.n=100, ci.alpha=0.95,
        stratified=TRUE,plot=FALSE,grid=FALSE,print.auc=FALSE, show.thres=TRUE, col='red')
        roc_test12<-roc.test(roc1,roc2)
        roc_test13<-roc.test(roc1,roc3)
        test_accuracy1<-coords(roc1, "best", ret=c("specificity","sensitivity","accuracy"), best.method=c("youden","closest.topleft"));
        test_accuracy2<-coords(roc2, "best", ret=c("specificity","sensitivity","accuracy"), best.method=c("youden","closest.topleft"));
        test_accuracy3<-coords(roc3, "best", ret=c("specificity","sensitivity","accuracy"), best.method=c("youden","closest.topleft"));
    ''')

    # Generate final form of Table 4
    columns=['Model','AUC','Accuracy','Sensitivity','Specificity','p-value'
            ]
    results = pd.DataFrame(
        {'Model':[],
         'AUC':[],
         'Accuracy':[],
         'Sensitivity':[],
         'Specificity':[],
         'p-value':[]
                 },
        columns=columns
    )

    results = results.append(
        pd.DataFrame(
            [
                [
                    'Model 1',
                    robjects.r["roc1"].rx2('auc')[0],
                    robjects.r["test_accuracy1"][2],
                    robjects.r["test_accuracy1"][1],
                    robjects.r["test_accuracy1"][0],
                    np.nan
                ],
                [
                    'Model 2',
                    robjects.r["roc2"].rx2('auc')[0],
                    robjects.r["test_accuracy2"][2],
                    robjects.r["test_accuracy2"][1],
                    robjects.r["test_accuracy2"][0],
                    robjects.r["roc_test12"].rx2('p.value')[0]
                ],
                [
                    'Model 3',
                    robjects.r["roc3"].rx2('auc')[0],
                    robjects.r["test_accuracy3"][2],
                    robjects.r["test_accuracy3"][1],
                    robjects.r["test_accuracy3"][0],
                    robjects.r["roc_test13"].rx2('p.value')[0]
                ]
            ],
            columns=columns
        ),
        ignore_index=True
    )

    return results

###
def classification3_feature_selection(rerun = False):
    '''
    This script runs the two step feature selection on outcome coded using Classification 1. First, it runs a feature selection using the F test (over 100 bootstraps of the training data). Second, a Random Forest based Cross-Validation feature elimination is used separately to get another set of features. Finally, the two sets are collated and only the features present in both sets are kept. This is done for the SSP-based feature set and Asymmetry-based feature set separately.
    '''
    if rerun:
        #Feature Selection for Model B

        # Parameters for feature selection
        n_repeats = 100
        num_cv = 5

        # Load data
        df = pd.read_csv('qPET_feature_matrix_clinical_3dssp.csv')
        df_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_3dssp.csv')
        drop_idx = np.where(np.array(df.outcomeLatest) > 4)[0]
        df = df.drop(df.index[drop_idx])


        # Generate feature matrix and target vectors
        X = np.array(df[df.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
        X_labels = np.array(df.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))

        # Generate output labels
        y = np.array(df.outcomeLatest)
        y[y<2] = 0
        y[y>=2] = 1


        all_features = []
        for iter_id in range(n_repeats):
            X_resample, y_resample = sklearn.utils.resample(X,y,n_samples=int(0.7*X.shape[0]),replace=False, random_state=RANDOM_STATE)
            fdr = SelectKBest(f_classif,k=int(n_repeats/2))
            fdr.fit(X_resample, y_resample)
            for index in fdr.get_support(indices=True):
                all_features.append(index)
        all_features =  np.sort(all_features)

        ft_counts = {}
        for ft in all_features:
            try:
                ft_counts[ft] += 1
            except KeyError:
                ft_counts[ft] = 1
        SSP_best_features1 = []
        for ft,ft_count in ft_counts.items():
            if ft_count > n_repeats/2:
                SSP_best_features1.append(ft)

        estimator = RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=RANDOM_STATE)
        selector = RFECV(estimator, step=1, cv=num_cv)
        selector = selector.fit(X,y)
        SSP_best_features2 = np.where(selector.support_)[0]

        SSP_features = list(set(SSP_best_features1) & set(SSP_best_features2))
        print sorted(SSP_features)



        #Feature Selection for Model C

        # Load data
        df = pd.read_csv('qPET_feature_matrix_clinical_voxel_ai.csv')
        df_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_voxel_ai.csv')
        drop_idx = np.where(np.array(df.outcomeLatest) > 4)[0]
        df = df.drop(df.index[drop_idx])

        # Generate feature matrix and target vectors
        X = np.array(df[df.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
        X_labels = np.array(df.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
        # Generate output labels
        y = np.array(df.outcomeLatest)
        y[y<2] = 0
        y[y>=2] = 1

        all_features = []
        for iter_id in range(n_repeats):
            X_resample, y_resample = sklearn.utils.resample(X,y,n_samples=int(0.7*X.shape[0]),replace=False, random_state=RANDOM_STATE)
            fdr = SelectKBest(f_classif,k=int(n_repeats/2))
            fdr.fit(X_resample, y_resample)
            for index in fdr.get_support(indices=True):
                all_features.append(index)
        all_features =  np.sort(all_features)

        ft_counts = {}
        for ft in all_features:
            try:
                ft_counts[ft] += 1
            except KeyError:
                ft_counts[ft] = 1
        ASYMM_best_features1 = []
        for ft,ft_count in ft_counts.items():
            if ft_count > n_repeats/2:
                ASYMM_best_features1.append(ft)

        estimator = RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=RANDOM_STATE)
        selector = RFECV(estimator, step=1, cv=num_cv)
        selector = selector.fit(X,y)
        ASYMM_best_features2 = np.where(selector.support_)[0]

        ASYMM_features = list(set(ASYMM_best_features1) & set(ASYMM_best_features2))
        print sorted(ASYMM_features)
    else:
        SSP_features = [1, 2, 5, 8, 9, 12, 19, 22, 29, 30, 32, 33, 37, 38, 39, 40, 41, 51, 52, 58, 60, 64, 68, 78, 80, 84, 85, 88, 91, 94, 98, 102, 104, 106, 108, 109, 113, 114, 115, 116, 117, 118, 127, 128, 134, 136, 140, 141]
        ASYMM_features = [12, 25, 27, 34, 42, 54, 73, 75, 77, 79, 88, 217, 219, 226, 234, 246, 250, 263, 265, 267, 269, 271, 280, 299, 339, 352]
    return SSP_features, ASYMM_features

def classification3_num_estimators(SSP_features, ASYMM_features):
    '''
    This script determines the optimal number of trees to use for each random forest classifier (Models A-C) based on outcome coded using Classification 3.
    '''
    # Load data for CLINICAL ONLY
    dfA = pd.read_csv('qPET_feature_matrix_clinical_only.csv')
    dfA_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_only.csv')
    drop_idx = np.where(np.array(dfA.outcomeLatest) > 4)[0]
    drop_test_idx = np.where(np.array(dfA_test.outcomeLatest) > 4)[0]
    dfA = dfA.drop(dfA.index[drop_idx])
    dfA_test = dfA_test.drop(dfA_test.index[drop_test_idx])

    # Generate feature matrix for training and testing data
    # Training
    X1 = np.array(dfA[dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X1_labels = np.array(dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))

    # Load data for SSP ONLY
    dfB = pd.read_csv('qPET_feature_matrix_clinical_3dssp.csv')
    dfB_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_3dssp.csv')
    drop_idx = np.where(np.array(dfB.outcomeLatest) > 4)[0]
    drop_test_idx = np.where(np.array(dfB_test.outcomeLatest) > 4)[0]
    dfB = dfB.drop(dfB.index[drop_idx])
    dfB_test = dfB_test.drop(dfB_test.index[drop_test_idx])

    # Generate feature matrix for training and testing data
    # Training
    X2 = np.array(dfB[dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X2_labels = np.array(dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X2 = X2[:,SSP_features]
    X2_labels = X2_labels[SSP_features]

    # Load data for ASYMMETRY ONLY
    dfC = pd.read_csv('qPET_feature_matrix_clinical_voxel_ai.csv')
    dfC_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_voxel_ai.csv')
    drop_idx = np.where(np.array(dfC.outcomeLatest) > 4)[0]
    drop_test_idx = np.where(np.array(dfC_test.outcomeLatest) > 4)[0]
    dfC = dfC.drop(dfC.index[drop_idx])
    dfC_test = dfC_test.drop(dfC_test.index[drop_test_idx])

    # Generate feature matrix for training testing data
    # Training
    X3 = np.array(dfC[dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X3_labels = np.array(dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X3 = X3[:,ASYMM_features]
    X3_labels = X3_labels[ASYMM_features]

    # Generate outcome variable
    outcome_threshold = 2 # i.e. Engel 1B
    y = np.array(dfA.outcomeLatest)
    y_test = np.array(dfA_test.outcomeLatest)
    y[y<outcome_threshold] = 0
    y[y>=outcome_threshold] = 1
    y_test[y_test<outcome_threshold] = 0
    y_test[y_test>=outcome_threshold] = 1

    # Generate feature matrices combined with clinical variables
    XA = X1
    XA_labels = X1_labels
    yA = y

    XB = np.hstack((X1,X2))
    XB_labels = np.hstack((X1_labels,X2_labels))
    yB = y

    yC = y
    XC = np.hstack((X1,X3))
    XC_labels = np.hstack((X1_labels,X3_labels))

    # Plot OOB score versus number of estimators
    min_estimators = 15
    max_estimators = 1000

    clfA = RandomForestClassifier(warm_start=True, max_features=None,
                                 oob_score=True, max_depth=max_depth,
                                 random_state=RANDOM_STATE)
    error_rate = []
    for i in range(min_estimators, max_estimators+1):
        clfA.set_params(n_estimators=i)
        clfA.fit(XA,yA)
        oob_error = 1-clfA.oob_score_
        error_rate.append((i, oob_error))
    error_rate = np.array(error_rate)
    plt.plot(error_rate[:,0], error_rate[:,1])
    plt.xlim(min_estimators, max_estimators)
    plt.title('OOB Error versus Number of Estimators for Clinical Only')
    plt.xlabel('n_estimators')
    plt.ylabel('OOB error rate')
    plt.show()
    nest = error_rate[:,0]
    err = error_rate[:,1]
    n_estimatorsA = int(nest[np.where(err == np.min(err))[0][0]])

    clfB = RandomForestClassifier(warm_start=True, max_features=None,
                                 oob_score=True, max_depth=max_depth,
                                 random_state=RANDOM_STATE)
    error_rate = []
    for i in range(min_estimators, max_estimators+1):
        clfB.set_params(n_estimators=i)
        clfB.fit(XB,yB)
        oob_error = 1-clfB.oob_score_
        error_rate.append((i, oob_error))
    error_rate = np.array(error_rate)
    plt.plot(error_rate[:,0], error_rate[:,1])
    plt.xlim(min_estimators, max_estimators)
    plt.title('OOB Error versus Number of Estimators for SSP and Clinical variables')
    plt.xlabel('n_estimators')
    plt.ylabel('OOB error rate')
    plt.show()
    nest = error_rate[:,0]
    err = error_rate[:,1]
    n_estimatorsB = int(nest[np.where(err == np.min(err))[0][0]])

    clfC = RandomForestClassifier(warm_start=True, max_features=None,
                                 oob_score=True, max_depth=max_depth,
                                 random_state=RANDOM_STATE)
    error_rate = []
    for i in range(min_estimators, max_estimators+1):
        clfC.set_params(n_estimators=i)
        clfC.fit(XC,yC)
        oob_error = 1-clfC.oob_score_
        error_rate.append((i, oob_error))
    error_rate = np.array(error_rate)
    plt.plot(error_rate[:,0], error_rate[:,1])
    plt.xlim(min_estimators, max_estimators)
    plt.title('OOB Error versus Number of Estimators for Asymmetry and Clinical variables')
    plt.xlabel('n_estimators')
    plt.ylabel('OOB error rate')
    plt.show()
    nest = error_rate[:,0]
    err = error_rate[:,1]
    n_estimatorsC = int(nest[np.where(err == np.min(err))[0][0]])

    return n_estimatorsA, n_estimatorsB, n_estimatorsC

def classification3_feature_importances(SSP_features, ASYMM_features):
    '''
    This script computes feature importance and performs ordinal logistic regression to 7-point outcome.
    '''
    # Load data for CLINICAL ONLY
    dfA = pd.read_csv('qPET_feature_matrix_clinical_only.csv')
    dfA_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_only.csv')
    drop_idx = np.where(np.array(dfA.outcomeLatest) > 4)[0]
    drop_test_idx = np.where(np.array(dfA_test.outcomeLatest) > 4)[0]
    dfA = dfA.drop(dfA.index[drop_idx])
    dfA_test = dfA_test.drop(dfA_test.index[drop_test_idx])

    # Generate feature matrix for training and testing data
    # Training
    X1 = np.array(dfA[dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X1_labels = np.array(dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    # Testing
    X1_test = np.array(dfA_test[dfA_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X1_test_labels = np.array(dfA_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))

    # Load data for SSP ONLY
    dfB = pd.read_csv('qPET_feature_matrix_clinical_3dssp.csv')
    dfB_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_3dssp.csv')
    drop_idx = np.where(np.array(dfB.outcomeLatest) > 4)[0]
    drop_test_idx = np.where(np.array(dfB_test.outcomeLatest) > 4)[0]
    dfB = dfB.drop(dfB.index[drop_idx])
    dfB_test = dfB_test.drop(dfB_test.index[drop_test_idx])

    # Generate feature matrix for training and testing data
    # Training
    X2 = np.array(dfB[dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X2_labels = np.array(dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X2 = X2[:,SSP_features]
    X2_labels = X2_labels[SSP_features]
    # Testing
    X2_test = np.array(dfB_test[dfB_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X2_test_labels = np.array(dfB_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X2_test = X2_test[:,SSP_features]
    X2_test_labels = X2_test_labels[SSP_features]

    # Load data for ASYMMETRY ONLY
    dfC = pd.read_csv('qPET_feature_matrix_clinical_voxel_ai.csv')
    dfC_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_voxel_ai.csv')
    drop_idx = np.where(np.array(dfC.outcomeLatest) > 4)[0]
    drop_test_idx = np.where(np.array(dfC_test.outcomeLatest) > 4)[0]
    dfC = dfC.drop(dfC.index[drop_idx])
    dfC_test = dfC_test.drop(dfC_test.index[drop_test_idx])

    # Generate feature matrix for training testing data
    # Training
    X3 = np.array(dfC[dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X3_labels = np.array(dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X3 = X3[:,ASYMM_features]
    X3_labels = X3_labels[ASYMM_features]
    # Testing
    X3_test = np.array(dfC_test[dfC_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X3_test_labels = np.array(dfC_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X3_test = X3_test[:,ASYMM_features]
    X3_test_labels = X3_test_labels[ASYMM_features]

    # Generate outcome variable
    outcome_threshold = 2 # i.e. Engel 1B
    y = np.array(dfA.outcomeLatest)
    y_test = np.array(dfA_test.outcomeLatest)
    # y[y<outcome_threshold] = 0
    # y[y>=outcome_threshold] = 1
    # y_test[y_test<outcome_threshold] = 0
    # y_test[y_test>=outcome_threshold] = 1

    # Run predictions
    # Generate feature matrix for CLINICAL ONLY
    XA = X1
    XA_labels = X1_labels
    XA_test = X1_test
    XA_test_labels = X1_test_labels
    yA = y

    # Generate feature matrix for SSP and CLINICAL
    XB = np.hstack((X1,X2))
    XB_labels = np.hstack((X1_labels,X2_labels))
    XB_test = np.hstack((X1_test,X2_test))
    XB_test_labels = np.hstack((X1_test_labels,X2_test_labels))
    yB = y

    # Generate feature matrix for SSP and CLINICAL
    XC = np.hstack((X1,X3))
    XC_labels = np.hstack((X1_labels,X3_labels))
    XC_test = np.hstack((X1_test,X3_test))
    XC_test_labels = np.hstack((X1_test_labels,X3_test_labels))
    yC = y


    robjects.globalenv['RANDOM_STATE'] = RANDOM_STATE

    # Generate final form of Table 2
    columns=[
        'Feature',
        'Feature Importance',
        'Univariate Odds Ratio per Unit\n Increase in Seizure Recurrence [95% CI, p]'
            ]
    results = pd.DataFrame(
        {'Feature':[],
         'Feature Importance':[],
         'Univariate Odds Ratio per Unit\n Increase in Seizure Recurrence [95% CI, p]':[]
                 },
        columns=columns
    )

    for X,y, X_labels in [(XA,yA,XA_labels), (XB,yB,XB_labels), (XC,yC,XC_labels)]:
        # Run feature importance (IncMSE) for random forests
        robjects.globalenv['X'] = X
        robjects.globalenv['y'] = y
        summary = robjects.r['summary']

        robjects.r('''
            set.seed(RANDOM_STATE)
            library(randomForest)
            rf <- randomForest(y ~ ., data=X, importance=TRUE)
            X_importances <- importance(rf)
        ''')
        X_importances = robjects.r['X_importances']
        feature_order = np.argsort(-X_importances[:,0])

        # Run univariate odds ratio computation
        for feature in feature_order:
            try:
                robjects.globalenv['feature'] = X[:,feature]
                robjects.r('''
                    set.seed(RANDOM_STATE)
                    require(MASS)
                    myfit <- polr(as.factor(y) ~ feature, Hess=TRUE)
                    (ctable <- coef(summary(myfit)))
                    p <- pnorm(abs(ctable[,"t value"]), lower.tail=FALSE)*2
                    (ctable <- cbind(ctable, "p value" = p))
                    (ci <- confint(myfit))
                    myfit_ci <-exp(cbind(OR = coef(myfit), ci))
                ''')
            except Exception:
                continue
            myfit = robjects.r['myfit']
            myfit_ci = robjects.r['myfit_ci']
            pval = robjects.r['p'][0]
            label = X_labels[feature]
            try:
                label = label.replace(label.split('_')[-1],ssp_labels[int(label.split('_')[-1])])
            except Exception:
                pass

            if X_importances[feature,0] <= 0.0:
                continue

            if pval < 0.001:
                pval = '%0.3f ***'%pval
            elif pval < 0.01:
                pval = '%0.3f **'%pval
            elif pval < 0.05:
                pval = '%0.3f *'%pval
            elif pval < 0.1:
                pval = '%0.3f .'%pval
            else:
                pval = '%0.2f'%pval

            results = results.append(
                pd.DataFrame(
                    [
                        [
                            label,
                            '%0.2f'%X_importances[feature,0],
                            '%0.2f [%0.2f - %0.2f, %s]'%(
                                myfit_ci[0,0],
                                myfit_ci[0,1], myfit_ci[1,1],
                                pval
                            )
                        ]
                    ],
                    columns=columns
                ),
                ignore_index=True
            )
        results = results.append(
            pd.DataFrame(
                [
                    [
                        '',
                        '',
                        ''
                    ]
                ],
                columns=columns
            ),
            ignore_index=True
        )
    return results

def classification3_cross_validation(SSP_features, ASYMM_features, n_estimatorsA, n_estimatorsB, n_estimatorsC):
    '''
    This script performs cross validation and measures out-of-bag score on outcome coded using Classification 3.
    '''
    # Preprocessing
    classifierA = RandomForestClassifier(n_estimators=n_estimatorsA, max_depth=max_depth, random_state=RANDOM_STATE,oob_score=True)
    classifierB = RandomForestClassifier(n_estimators=n_estimatorsB, max_depth=max_depth, random_state=RANDOM_STATE, oob_score=True)
    classifierC = RandomForestClassifier(n_estimators=n_estimatorsC, max_depth=max_depth, random_state=RANDOM_STATE, oob_score=True)

    sss = StratifiedShuffleSplit(n_splits=NUM_BOOTSTRAPS, test_size=0.2, random_state=RANDOM_STATE)

    print 'Generating feature matrices ...'
    # Load data for CLINICAL ONLY
    dfA = pd.read_csv('qPET_feature_matrix_clinical_only.csv')
    dfA_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_only.csv')
    drop_idx = np.where(np.array(dfA.outcomeLatest) > 4)[0]
    drop_test_idx = np.where(np.array(dfA_test.outcomeLatest) > 4)[0]
    dfA = dfA.drop(dfA.index[drop_idx])
    dfA_test = dfA_test.drop(dfA_test.index[drop_test_idx])

    # Generate feature matrix for training and testing data
    # Training
    X1 = np.array(dfA[dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X1_labels = np.array(dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    # Testing
    X1_test = np.array(dfA_test[dfA_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X1_test_labels = np.array(dfA_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))

    # Load data for SSP ONLY
    dfB = pd.read_csv('qPET_feature_matrix_clinical_3dssp.csv')
    dfB_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_3dssp.csv')
    drop_idx = np.where(np.array(dfB.outcomeLatest) > 4)[0]
    drop_test_idx = np.where(np.array(dfB_test.outcomeLatest) > 4)[0]
    dfB = dfB.drop(dfB.index[drop_idx])
    dfB_test = dfB_test.drop(dfB_test.index[drop_test_idx])

    # Generate feature matrix for training and testing data
    # Training
    X2 = np.array(dfB[dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X2_labels = np.array(dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X2 = X2[:,SSP_features]
    X2_labels = X2_labels[SSP_features]
    # Testing
    X2_test = np.array(dfB_test[dfB_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X2_test_labels = np.array(dfB_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X2_test = X2_test[:,SSP_features]
    X2_test_labels = X2_test_labels[SSP_features]

    # Load data for ASYMMETRY ONLY
    dfC = pd.read_csv('qPET_feature_matrix_clinical_voxel_ai.csv')
    dfC_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_voxel_ai.csv')
    drop_idx = np.where(np.array(dfC.outcomeLatest) > 4)[0]
    drop_test_idx = np.where(np.array(dfC_test.outcomeLatest) > 4)[0]
    dfC = dfC.drop(dfC.index[drop_idx])
    dfC_test = dfC_test.drop(dfC_test.index[drop_test_idx])

    # Generate feature matrix for training testing data
    # Training
    X3 = np.array(dfC[dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X3_labels = np.array(dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X3 = X3[:,ASYMM_features]
    X3_labels = X3_labels[ASYMM_features]
    # Testing
    X3_test = np.array(dfC_test[dfC_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X3_test_labels = np.array(dfC_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X3_test = X3_test[:,ASYMM_features]
    X3_test_labels = X3_test_labels[ASYMM_features]

    # Generate outcome variable
    outcome_threshold = 2 # i.e. Engel 1B
    y = np.array(dfA.outcomeLatest)
    y_test = np.array(dfA_test.outcomeLatest)
    y[y<outcome_threshold] = 0
    y[y>=outcome_threshold] = 1
    y_test[y_test<outcome_threshold] = 0
    y_test[y_test>=outcome_threshold] = 1


    columns=['Clinical Variable(s)','CV 5-Fold AUC_1','Out Of Bag (OOB1) Error',
             'Quantitative Variables','CV 5-Fold AUC_2','Out Of Bag (OOB2) Error',
             'AUC Difference 95% C.I.','OOB Difference 95% C.I.'
            ]
    results = pd.DataFrame(
        {'Clinical Variable(s)':[],
         'CV 5-Fold AUC_1':[],
         'Out Of Bag (OOB1) Error':[],
         'Quantitative Variables':[],
         'CV 5-Fold AUC_2':[],
         'Out Of Bag (OOB2) Error':[],
         'AUC Difference 95% C.I.':[],
         'OOB Difference 95% C.I.':[]
                 },
        columns=columns
    )

    OPTIONS = {
        1:[np.arange(20),'EEG+MRI+PET'],
        2:[np.arange(12),'EEG'],
        3:[np.arange(12,14),'MRI'],
        4:[np.arange(13,20),'PET'],
    }

    for OPTION,(clinical_variable_idx,variables_list) in OPTIONS.items():
        # Generate feature matrix for CLINICAL ONLY
        XA = X1[:,clinical_variable_idx]
        XA_labels = X1_labels[clinical_variable_idx]
        yA = y

        # Generate feature matrix for SSP alone
        XB = X2
        XB_labels = X2_labels
        XB_test = X2_test
        XB_test_labels = X2_test_labels
        yB = y

        # Generate bootstrap jobs
        jobs = []
        out = []
        for job_iter,(train, test) in enumerate(sss.split(XA,yA)):
            jobs.append((job_iter,train,test,XA,yA,XB,yB,classifierA,classifierB))
        # Run all jobs
        return_list = pool.map(_helper,jobs)
        # Compute OOB/AUC for bootstraps
        for res in return_list:
            probas_A,yA_test,probas_B,yB_test,classifierA_oob_score_,classifierB_oob_score_ = res
            fprA, tprA, thresholds = roc_curve(yA_test, probas_A[:, 1])
            fprB, tprB, thresholds = roc_curve(yB_test, probas_B[:, 1])
            roc_aucA = auc(fprA, tprA)
            roc_aucB = auc(fprB, tprB)
            out.append([roc_aucA,roc_aucB,classifierA_oob_score_,classifierB_oob_score_])
        out21 = np.array(out)

        # Generate feature matrix for SSP and CLINICAL
        XB = np.hstack((X1,X2))
        XB_labels = np.hstack((X1_labels,X2_labels))
        XB_test = np.hstack((X1_test,X2_test))
        XB_test_labels = np.hstack((X1_test_labels,X2_test_labels))

        # Generate bootstrap jobs
        jobs = []
        out = []
        for job_iter,(train, test) in enumerate(sss.split(XA,yA)):
            jobs.append((job_iter,train,test,XA,yA,XB,yB,classifierA,classifierB))
        # Run all jobs
        return_list = pool.map(_helper,jobs)
        # Compute OOB/AUC for bootstraps
        for res in return_list:
            probas_A,yA_test,probas_B,yB_test,classifierA_oob_score_,classifierB_oob_score_ = res
            fprA, tprA, thresholds = roc_curve(yA_test, probas_A[:, 1])
            fprB, tprB, thresholds = roc_curve(yB_test, probas_B[:, 1])
            roc_aucA = auc(fprA, tprA)
            roc_aucB = auc(fprB, tprB)
            out.append([roc_aucA,roc_aucB,classifierA_oob_score_,classifierB_oob_score_])
        out31 = np.array(out)

        # Compute statistics
        AUC1 = np.mean(out21[:,0])
        OOB1 = np.mean(out21[:,2])
        AUC2 = np.mean(out21[:,1])
        OOB2 = np.mean(out21[:,3])
        AUC3 = np.mean(out31[:,1])
        OOB3 = np.mean(out31[:,3])
        CI_AUC2_minus_AUC1 = tuple(map(lambda x: np.percentile(out21[:,1] - out21[:,0],x), [2.5, 97.5]))
        CI_AUC3_minus_AUC1 = tuple(map(lambda x: np.percentile(out31[:,1] - out31[:,0],x), [2.5, 97.5]))
        CI_OOB2_minus_OOB1 = tuple(map(lambda x: np.percentile(out21[:,3] - out21[:,2],x), [2.5, 97.5]))
        CI_OOB3_minus_OOB1 = tuple(map(lambda x: np.percentile(out31[:,3] - out31[:,2],x), [2.5, 97.5]))


        results = results.append(
            pd.DataFrame(
                [
                    [variables_list,AUC1,OOB1,'%s+SSP'%(variables_list), AUC2, OOB2, CI_AUC2_minus_AUC1, CI_OOB2_minus_OOB1],
                    [variables_list,AUC1,OOB1,'SSP only', AUC3, OOB3, CI_AUC3_minus_AUC1, CI_OOB3_minus_OOB1]
                ],
                columns=columns
            ),
            ignore_index=True
        )

    resultsSSP = results

    columns=['Clinical Variable(s)','CV 5-Fold AUC_1','Out Of Bag (OOB1) Error',
             'Quantitative Variables','CV 5-Fold AUC_2','Out Of Bag (OOB2) Error',
             'AUC Difference 95% C.I.','OOB Difference 95% C.I.'
            ]
    results = pd.DataFrame(
        {'Clinical Variable(s)':[],
         'CV 5-Fold AUC_1':[],
         'Out Of Bag (OOB1) Error':[],
         'Quantitative Variables':[],
         'CV 5-Fold AUC_2':[],
         'Out Of Bag (OOB2) Error':[],
         'AUC Difference 95% C.I.':[],
         'OOB Difference 95% C.I.':[]
                 },
        columns=columns
    )

    OPTIONS = {
        1:[np.arange(20),'EEG+MRI+PET'],
        2:[np.arange(12),'EEG'],
        3:[np.arange(12,14),'MRI'],
        4:[np.arange(13,20),'PET'],
    }

    for OPTION,(clinical_variable_idx,variables_list) in OPTIONS.items():
        # Generate feature matrix for CLINICAL ONLY
        XA = X1[:,clinical_variable_idx]
        XA_labels = X1_labels[clinical_variable_idx]
        yA = y

        # Generate feature matrix for SSP alone
        XC = X3
        XC_labels = X3_labels
        XC_test = X3_test
        XC_test_labels = X3_test_labels
        yC = y

        # Generate bootstrap jobs
        jobs = []
        out = []
        for job_iter,(train, test) in enumerate(sss.split(XA,yA)):
            jobs.append((job_iter,train,test,XA,yA,XC,yC,classifierA,classifierC))
        # Run all jobs
        return_list = pool.map(_helper,jobs)
        # Compute OOB/AUC for bootstraps
        for res in return_list:
            probas_A,yA_test,probas_C,yC_test,classifierA_oob_score_,classifierC_oob_score_ = res
            fprA, tprA, thresholds = roc_curve(yA_test, probas_A[:, 1])
            fprC, tprC, thresholds = roc_curve(yC_test, probas_C[:, 1])
            roc_aucA = auc(fprA, tprA)
            roc_aucC = auc(fprC, tprC)
            out.append([roc_aucA,roc_aucC,classifierA_oob_score_,classifierC_oob_score_])
        out21 = np.array(out)

        # Generate feature matrix for SSP and CLINICAL
        XC = np.hstack((X1,X3))
        XC_labels = np.hstack((X1_labels,X3_labels))
        XC_test = np.hstack((X1_test,X3_test))
        XC_test_labels = np.hstack((X1_test_labels,X3_test_labels))

        # Generate bootstrap jobs
        jobs = []
        out = []
        for job_iter,(train, test) in enumerate(sss.split(XA,yA)):
            jobs.append((job_iter,train,test,XA,yA,XC,yB,classifierA,classifierB))
        # Run all jobs
        return_list = pool.map(_helper,jobs)
        # Compute OOB/AUC for bootstraps
        for res in return_list:
            probas_A,yA_test,probas_C,yC_test,classifierA_oob_score_,classifierC_oob_score_ = res
            fprA, tprA, thresholds = roc_curve(yA_test, probas_A[:, 1])
            fprC, tprC, thresholds = roc_curve(yC_test, probas_C[:, 1])
            roc_aucA = auc(fprA, tprA)
            roc_aucC = auc(fprC, tprC)
            out.append([roc_aucA,roc_aucC,classifierA_oob_score_,classifierC_oob_score_])
        out31 = np.array(out)

        # Compute statistics
        AUC1 = np.mean(out21[:,0])
        OOB1 = np.mean(out21[:,2])
        AUC2 = np.mean(out21[:,1])
        OOB2 = np.mean(out21[:,3])
        AUC3 = np.mean(out31[:,1])
        OOB3 = np.mean(out31[:,3])
        CI_AUC2_minus_AUC1 = tuple(map(lambda x: np.percentile(out21[:,1] - out21[:,0],x), [2.5, 97.5]))
        CI_AUC3_minus_AUC1 = tuple(map(lambda x: np.percentile(out31[:,1] - out31[:,0],x), [2.5, 97.5]))
        CI_OOB2_minus_OOB1 = tuple(map(lambda x: np.percentile(out21[:,3] - out21[:,2],x), [2.5, 97.5]))
        CI_OOB3_minus_OOB1 = tuple(map(lambda x: np.percentile(out31[:,3] - out31[:,2],x), [2.5, 97.5]))


        results = results.append(
            pd.DataFrame(
                [
                    [variables_list,AUC1,OOB1,'%s+Asymmetry'%(variables_list), AUC2, OOB2, CI_AUC2_minus_AUC1, CI_OOB2_minus_OOB1],
                    [variables_list,AUC1,OOB1,'Asymmetry only', AUC3, OOB3, CI_AUC3_minus_AUC1, CI_OOB3_minus_OOB1]
                ],
                columns=columns
            ),
            ignore_index=True
        )

    resultsASYMM = results
    return resultsSSP, resultsASYMM

def classification3_validation(SSP_features, ASYMM_features, n_estimatorsA, n_estimatorsB, n_estimatorsC):
    '''
    This script measures accuracy on the validation cohort using Classification 3.
    '''
    # Create feature matrix for training and testing
    # Load data for CLINICAL ONLY
    dfA = pd.read_csv('qPET_feature_matrix_clinical_only.csv')
    dfA_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_only.csv')
    drop_idx = np.where(np.array(dfA.outcomeLatest) > 4)[0]
    drop_test_idx = np.where(np.array(dfA_test.outcomeLatest) > 4)[0]
    dfA = dfA.drop(dfA.index[drop_idx])
    dfA_test = dfA_test.drop(dfA_test.index[drop_test_idx])

    # Generate feature matrix for training and testing data
    # Training
    X1 = np.array(dfA[dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X1_labels = np.array(dfA.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    # Testing
    X1_test = np.array(dfA_test[dfA_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X1_test_labels = np.array(dfA_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))

    # Load data for SSP ONLY
    dfB = pd.read_csv('qPET_feature_matrix_clinical_3dssp.csv')
    dfB_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_3dssp.csv')
    drop_idx = np.where(np.array(dfB.outcomeLatest) > 4)[0]
    drop_test_idx = np.where(np.array(dfB_test.outcomeLatest) > 4)[0]
    dfB = dfB.drop(dfB.index[drop_idx])
    dfB_test = dfB_test.drop(dfB_test.index[drop_test_idx])

    # Generate feature matrix for training and testing data
    # Training
    X2 = np.array(dfB[dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X2_labels = np.array(dfB.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X2 = X2[:,SSP_features]
    X2_labels = X2_labels[SSP_features]
    # Testing
    X2_test = np.array(dfB_test[dfB_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X2_test_labels = np.array(dfB_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X2_test = X2_test[:,SSP_features]
    X2_test_labels = X2_test_labels[SSP_features]

    # Load data for ASYMMETRY ONLY
    dfC = pd.read_csv('qPET_feature_matrix_clinical_voxel_ai.csv')
    dfC_test = pd.read_csv('qPET_Validation_feature_matrix_clinical_voxel_ai.csv')
    drop_idx = np.where(np.array(dfC.outcomeLatest) > 4)[0]
    drop_test_idx = np.where(np.array(dfC_test.outcomeLatest) > 4)[0]
    dfC = dfC.drop(dfC.index[drop_idx])
    dfC_test = dfC_test.drop(dfC_test.index[drop_test_idx])

    # Generate feature matrix for training testing data
    # Training
    X3 = np.array(dfC[dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X3_labels = np.array(dfC.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X3 = X3[:,ASYMM_features]
    X3_labels = X3_labels[ASYMM_features]
    # Testing
    X3_test = np.array(dfC_test[dfC_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest'])])
    X3_test_labels = np.array(dfC_test.columns.difference(['Unnamed: 0','id','outcome12','outcome24','outcome6','outcomeLatest']))
    X3_test = X3_test[:,ASYMM_features]
    X3_test_labels = X3_test_labels[ASYMM_features]

    # Generate outcome variable
    outcome_threshold = 2 # i.e. Engel 1B
    y = np.array(dfA.outcomeLatest)
    y_test = np.array(dfA_test.outcomeLatest)
    y[y<outcome_threshold] = 0
    y[y>=outcome_threshold] = 1
    y_test[y_test<outcome_threshold] = 0
    y_test[y_test>=outcome_threshold] = 1

    # Run predictions
    # Generate feature matrix for CLINICAL ONLY
    XA = X1
    XA_labels = X1_labels
    XA_test = X1_test
    XA_test_labels = X1_test_labels

    # Generate feature matrix for SSP and CLINICAL
    XB = np.hstack((X1,X2))
    XB_labels = np.hstack((X1_labels,X2_labels))
    XB_test = np.hstack((X1_test,X2_test))
    XB_test_labels = np.hstack((X1_test_labels,X2_test_labels))

    # Generate feature matrix for SSP and CLINICAL
    XC = np.hstack((X1,X3))
    XC_labels = np.hstack((X1_labels,X3_labels))
    XC_test = np.hstack((X1_test,X3_test))
    XC_test_labels = np.hstack((X1_test_labels,X3_test_labels))

    # Load library in R
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    pROC = importr('pROC')

    # Train classifier and apply to validation test set
    classifierA = RandomForestClassifier(
        n_estimators=n_estimatorsA,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        oob_score=True)
    classifierB = RandomForestClassifier(
        n_estimators=n_estimatorsB,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        oob_score=True)
    classifierC = RandomForestClassifier(
        n_estimators=n_estimatorsC,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        oob_score=True)

    probas = classifierA.fit(XA, y).predict_proba(XA_test)
    predsA = robjects.FloatVector(probas[:,1])
    labelsA = robjects.IntVector(y_test)
    probas = classifierB.fit(XB, y).predict_proba(XB_test)
    predsB = robjects.FloatVector(probas[:,1])
    labelsB = robjects.IntVector(y_test)
    probas = classifierC.fit(XC, y).predict_proba(XC_test)
    predsC = robjects.FloatVector(probas[:,1])
    labelsC = robjects.IntVector(y_test)


    # Copy from python workspace to R workspace
    robjects.globalenv["predsA"] = predsA
    robjects.globalenv["labelsA"] = labelsA
    robjects.globalenv["predsB"] = predsB
    robjects.globalenv["labelsB"] = labelsB
    robjects.globalenv["predsC"] = predsC
    robjects.globalenv["labelsC"] = labelsC

    # Run pROC.roc and pROC.roc.test in R
    robjects.r('''
        predsA<-as.numeric(unlist(predsA))
        labelsA<-as.numeric(unlist(labelsA))
        predsB<-as.numeric(unlist(predsB))
        labelsB<-as.numeric(unlist(labelsB))
        predsC<-as.numeric(unlist(predsC))
        labelsC<-as.numeric(unlist(labelsC))

        library(pROC)

        roc1 <- roc(labelsA, predsA, percent=FALSE,
        smooth=TRUE, ci=TRUE, boot.n=100, ci.alpha=0.95,
        stratified=TRUE,plot=FALSE,grid=FALSE,print.auc=FALSE, show.thres=TRUE, col='red')
        roc2 <- roc(labelsB, predsB, percent=FALSE,
        smooth=TRUE, ci=TRUE, boot.n=100, ci.alpha=0.95,
        stratified=TRUE,plot=FALSE,grid=FALSE,print.auc=FALSE, show.thres=TRUE, col='red')
        roc3 <- roc(labelsC, predsC, percent=FALSE,
        smooth=TRUE, ci=TRUE, boot.n=100, ci.alpha=0.95,
        stratified=TRUE,plot=FALSE,grid=FALSE,print.auc=FALSE, show.thres=TRUE, col='red')
        roc_test12<-roc.test(roc1,roc2)
        roc_test13<-roc.test(roc1,roc3)
        test_accuracy1<-coords(roc1, "best", ret=c("specificity","sensitivity","accuracy"), best.method=c("youden","closest.topleft"));
        test_accuracy2<-coords(roc2, "best", ret=c("specificity","sensitivity","accuracy"), best.method=c("youden","closest.topleft"));
        test_accuracy3<-coords(roc3, "best", ret=c("specificity","sensitivity","accuracy"), best.method=c("youden","closest.topleft"));
    ''')

    # Generate final form of Table 4
    columns=['Model','AUC','Accuracy','Sensitivity','Specificity','p-value'
            ]
    results = pd.DataFrame(
        {'Model':[],
         'AUC':[],
         'Accuracy':[],
         'Sensitivity':[],
         'Specificity':[],
         'p-value':[]
                 },
        columns=columns
    )

    results = results.append(
        pd.DataFrame(
            [
                [
                    'Model 1',
                    robjects.r["roc1"].rx2('auc')[0],
                    robjects.r["test_accuracy1"][2],
                    robjects.r["test_accuracy1"][1],
                    robjects.r["test_accuracy1"][0],
                    np.nan
                ],
                [
                    'Model 2',
                    robjects.r["roc2"].rx2('auc')[0],
                    robjects.r["test_accuracy2"][2],
                    robjects.r["test_accuracy2"][1],
                    robjects.r["test_accuracy2"][0],
                    robjects.r["roc_test12"].rx2('p.value')[0]
                ],
                [
                    'Model 3',
                    robjects.r["roc3"].rx2('auc')[0],
                    robjects.r["test_accuracy3"][2],
                    robjects.r["test_accuracy3"][1],
                    robjects.r["test_accuracy3"][0],
                    robjects.r["roc_test13"].rx2('p.value')[0]
                ]
            ],
            columns=columns
        ),
        ignore_index=True
    )
    return results
