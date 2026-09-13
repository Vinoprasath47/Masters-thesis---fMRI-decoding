#!/usr/bin/env python
import os
import sys
import glob
import json
import time
import pickle
import argparse
import warnings

import numpy as np
import pandas as pd
import nibabel as nib

from pathlib import Path
from collections import defaultdict

from nilearn.image import (load_img, concat_imgs, mean_img, new_img_like, resample_to_img, index_img, math_img)
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import MultiNiftiLabelsMasker, NiftiLabelsMasker, NiftiMasker  # CHANGED: added NiftiMasker
from nilearn.decoding import Decoder
from scipy.ndimage import binary_dilation

from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC                 
from sklearn.metrics import roc_auc_score         
from sklearn.preprocessing import LabelBinarizer  #needed for roc_auc_ovr_scorer
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed

print("=" * 60, flush=True)
print("SETUP SCRIPT STARTED", flush=True)
print("=" * 60, flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_jobs", type=int, default=48, help="Number of parallel jobs (set to --cpus-per-task)")
args = parser.parse_args()
n_jobs = args.n_jobs
print(f"Using n_jobs = {n_jobs}", flush=True)


def suppress_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

suppress_warnings()
Parallel(n_jobs=n_jobs)(delayed(suppress_warnings)() for _ in range(2 * n_jobs))

#creating dirctories
base_dir = os.getcwd()                                  # renamed from 'dir' (shadows builtin)
output_dir = os.path.join(base_dir, "Masks")
os.makedirs(output_dir, exist_ok=True)


betamaps = os.path.join(base_dir, "Folder", "sub-**", "**_run-**_**.nii.gz")
betafiles = glob.glob(betamaps)
print(f" no of betafiles: {len(betafiles)}", flush=True)

t0 = time.time()
x_files = concat_imgs(betafiles)
print(f"concat_imgs done in {time.time()-t0:.1f}s | shape: {x_files.shape}", flush=True)

m_img = mean_img(x_files, n_jobs=n_jobs)
print(f" mean image shape: {m_img.shape}", flush=True)

#mask files from contrast and conjunction
print("Loading mask files", flush=True)
mask_file_contrast = os.path.join(base_dir, "cluster_mask_zstat3.nii.gz")
mask_file_conjunction = os.path.join(base_dir, "cluster_mask_grot.nii.gz")

mask_contrast = load_img(mask_file_contrast)
mask_conjunction = load_img(mask_file_conjunction)
mask_combined = math_img("img > 0", img=mask_contrast)   # 1275 voxels


print("maps from schaefer atlas", flush=True)  
schaefer = fetch_atlas_schaefer_2018(yeo_networks=17, n_rois=400, resolution_mm=1)
schaefer_labels = schaefer["labels"][1:401]                  # skip background (index 0)
schaefer_maps = load_img(schaefer["maps"])


print("Neurosynth association maps", flush=True) 
episodic_img = load_img(os.path.join(base_dir, "episodic_association-test_z_FDR_0.01.nii.gz"))
semantic_img = load_img(os.path.join(base_dir, "semantic memory_association-test_z_FDR_0.01.nii.gz"))

atlas_mask = MultiNiftiLabelsMasker(schaefer_maps,background_label="background", resampling_target="data", n_jobs=n_jobs)

ep_raw = atlas_mask.fit_transform(episodic_img)
sm_raw = atlas_mask.fit_transform(semantic_img)

# threshold at 0 and round
ep_img = np.round(np.where(ep_raw > 0, ep_raw, 0), decimals=2)
sm_img = np.round(np.where(sm_raw > 0, sm_raw, 0), decimals=2)

conjunction_vals = np.round(atlas_mask.fit_transform(mask_conjunction), 2)
contrast_vals = np.round(atlas_mask.fit_transform(mask_combined),    2)

#ROI selection
# conjunction: overlap with semantic map  #contrast: overlap with episodic map
common_ep_conjunction = np.logical_and(sm_img, conjunction_vals)
common_ep_conjunction = np.array(np.delete(common_ep_conjunction, 0))  # remove background

common_ep_contrast = np.logical_and(ep_img, contrast_vals)
common_ep_contrast = np.array(np.delete(common_ep_contrast, 0))

maps_conjunction = np.array(schaefer_labels)[common_ep_conjunction].astype(str)
maps_contrast = np.array(schaefer_labels)[common_ep_contrast].astype(str)

print(f"Conjunction ROIs selected : {len(maps_conjunction)}", flush=True)
print(f"Contrast ROIs selected    : {len(maps_contrast)}",    flush=True)

#creating masks from schaefer atlas for selected ROIs
def build_masks(maps_list, schaefer_labels, schaefer_maps, m_img, label_prefix_len=11):

    masks_imgs = {}
    for label in maps_list:
        match_id = (np.char.find(schaefer_labels, label) >= 0)
        index = np.where(match_id)[0] + 1                           # +1: atlas indices are 1-based
        name = label[label_prefix_len:]
        index_mask = np.isin(schaefer_maps.get_fdata(), index)
        index_map = resample_to_img(
            new_img_like(schaefer_maps, binary_dilation(index_mask)),
            m_img,
            interpolation="nearest",
            copy_header=True,
            force_resample=True
        )
        masks_imgs[name] = index_map
    names = list(masks_imgs.keys())
    imgs = list(masks_imgs.values())
    return masks_imgs, names, imgs

print("creating conjunction masks", flush=True)
masks_maps_conjunction, list_masks_names_conjunction, list_masks_conjunction = \
    build_masks(maps_conjunction, schaefer_labels, schaefer_maps, m_img)

print("creating contrast masks", flush=True)
masks_maps_contrast, list_masks_names_contrast, list_masks_contrast = \
    build_masks(maps_contrast, schaefer_labels, schaefer_maps, m_img)


print(f"old conj list length is {len(list_masks_conjunction)}")
print(f"old cont list length is {len(list_masks_contrast)}")

contrast = np.load(os.path.join(base_dir, "contrast_rois.npy"), allow_pickle=True)
conjunction = np.load(os.path.join(base_dir, "conjunction_rois.npy"), allow_pickle=True)

contrast_masks = []
contrast_masks_names = []

conjunction_masks = []
conjunction_masks_names = []

for i, j in enumerate(list_masks_names_contrast):
    for mask in contrast:
        if j == mask:
            contrast_masks.append(list_masks_contrast[i])
            contrast_masks_names.append(list_masks_names_contrast[i])
list_masks_names_contrast = contrast_masks_names
list_masks_contrast = contrast_masks        
            
for i, j in enumerate(list_masks_names_conjunction):
    for mask in conjunction:
        if j == mask:
            conjunction_masks.append(list_masks_conjunction[i])
            conjunction_masks_names.append(list_masks_names_conjunction[i])
list_masks_names_conjunction = conjunction_masks_names
list_masks_conjunction = conjunction_masks

print(f"new conj list length is {len(list_masks_conjunction)}")
print(f"new cont list length is {len(list_masks_contrast)}")



#save NIfTI masks and JSON index (conjunction only, matching original)
masks_maps_json = {}
for key, img in masks_maps_conjunction.items():
    fname = key + ".nii.gz"
    fpath = os.path.join(output_dir, fname)
    nib.save(img, fpath)
    masks_maps_json[key] = fpath

with open(os.path.join(output_dir, "masks_dictionary_conjunction.json"), "w") as f:
    json.dump(masks_maps_json, f)

common_strings = list(set(list_masks_names_contrast) & set(list_masks_names_conjunction))
print(f"ROIs common to both mask sets: {len(common_strings)}", flush=True)


group_labels_list = []
category_labels = []
subject_labels_list = []

for f in betafiles:
    filename = os.path.basename(f)
    filename_no_ext = filename.replace(".nii.gz", "")
    parts = filename_no_ext.split("_")

    group_labels_list.append(parts[0])                                  # "AB" or "PD"
    category_labels.append(parts[2])                                    # "intemo" etc.
    subject_labels_list.append(os.path.basename(os.path.dirname(f)))   # "sub-13"

group_arr = np.array(group_labels_list)
category_arr = np.array(category_labels)
subject_arr = np.array(subject_labels_list)

selected_conditions = ["extsem", "intact", "intemo", "intobj", "intplc", "intprc", "intprs", "inttim"]

cond_mask_AB = group_arr == "AB"
y_mask = np.isin(category_arr, selected_conditions)

x_f_AB = index_img(x_files, np.logical_and(cond_mask_AB, y_mask))
x_f_PD = index_img(x_files, np.logical_and(np.logical_not(cond_mask_AB), y_mask))
y_AB = category_arr[np.logical_and(cond_mask_AB, y_mask)]
y_PD = category_arr[np.logical_and(np.logical_not(cond_mask_AB), y_mask)]
groups_AB = subject_arr[np.logical_and(cond_mask_AB, y_mask)]
groups_PD = subject_arr[np.logical_and(np.logical_not(cond_mask_AB), y_mask)]

print(f"AB images: {x_f_AB.shape[-1]}, subjects: {len(np.unique(groups_AB))}", flush=True)
print(f"PD images: {x_f_PD.shape[-1]}, subjects: {len(np.unique(groups_PD))}", flush=True)


all_classes = np.array(sorted(np.unique(np.concatenate([y_AB, y_PD]))))

def roc_auc_ovr_scorer(estimator, X, y):

    decision_scores = estimator.decision_function(X)   
    lb = LabelBinarizer()
    lb.fit(all_classes)                                
    y_bin = lb.transform(y)                            
    return roc_auc_score(y_bin, decision_scores, multi_class="ovr", average=None)


def subsample_fair(all_indices, groups, target_n, rng):
    
    unique_subjs = np.unique(groups[all_indices])
    n_subjs = len(unique_subjs)
    base = target_n // n_subjs
    extra = target_n % n_subjs    

    shuffled_subjs = rng.permutation(unique_subjs)   

    selected = []
    for i, subj in enumerate(shuffled_subjs):
        quota = base + (1 if i < extra else 0)
        subj_idx = all_indices[groups[all_indices] == subj]
        n_pick = min(quota, len(subj_idx))          # never exceed what subject has
        chosen = rng.choice(subj_idx, n_pick, replace=False)
        selected.extend(chosen.tolist())

    return np.sort(np.array(selected, dtype=int))

def fold_has_all_classes(indices, y, all_classes):

    present = np.unique(y[indices])
    return len(present) == len(all_classes)


#within group classification

group_names = ["AB", "PD"]
analysisfile   = [(x_f_AB, y_AB, groups_AB), (x_f_PD, y_PD, groups_PD)]
all_masks   = [list_masks_contrast,        list_masks_conjunction]
all_masks_names = [list_masks_names_contrast, list_masks_names_conjunction]
mask_set_names  = ["contrast", "conjunction"]

cv = GroupKFold(n_splits=10)

for n, (mask_list, mask_names) in enumerate(zip(all_masks, all_masks_names)):
    print(f"\n  Mask set {n+1}/2: {mask_set_names[n]} ({len(mask_list)} masks)", flush=True)
    results_dict = defaultdict(dict)

    for j, mask_img in enumerate(mask_list):
        print(f"    Mask {j+1}/{len(mask_list)}: {mask_names[j]}", flush=True)

        decoder = Decoder(
            estimator="svc_l2",
            mask=mask_img,
            standardize="zscore_sample",
            screening_percentile=100,
            scoring="roc_auc",
            cv=cv,                       # ← GroupKFold instead of cv=5
            n_jobs=n_jobs
        )

        for i, (key, labels, groups) in enumerate(analysisfile):
            t_start = time.time()
            decoder.fit(key, labels, groups=groups)   # ← groups passed here
            y_predict = decoder.predict(key)
            score  = decoder.cv_scores_

            temp = {}
            for cat, cv_list in score.items():
                mean_auc = np.mean(cv_list)
                print(f"{group_names[i]} | {cat}: roc_auc_ovr={mean_auc:.4f}", flush=True)
                temp[cat] = mean_auc
            temp["y_predict"] = y_predict
            temp["cv_scores"] = score

            results_dict[mask_names[j]][group_names[i]] = temp
            print(f"{group_names[i]} | {cat}: roc_auc_ovr={mean_auc:.4f}", flush=True)

    dfs = [pd.DataFrame(results_dict[roi]).stack() for roi in results_dict]
    df  = pd.concat(dfs, keys=results_dict.keys())
    df.index.names = [f"{mask_set_names[n]} ROI", "Category", "Condition"]
    df.name = "roc_auc_ovr"
    df_results = df.reset_index()

    excel_path = os.path.join(base_dir, f"decoder_results_{mask_set_names[n]}.xlsx")
    df_results.to_excel(excel_path, index=False)
    print(f"saved: {excel_path}", flush=True)

    pkl_path = os.path.join(output_dir, f"results_{mask_set_names[n]}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"saved: {pkl_path}", flush=True)




#cross task classification
N_SPLITS_CROSS = 10   # same as within-group CV
rng_setup = np.random.default_rng(seed=42)
cross_task_folds = []

unique_subjects  = np.unique(groups_AB)   # all 40 subject IDs
N_TEST_SUBJECTS = 4
N_TRAIN_SUBJECTS = len(unique_subjects) - N_TEST_SUBJECTS   # 36
MAX_ATTEMPTS = 500   # safety limit per fold

for fold_idx in range(10):
    accepted = False
    attempt  = 0

    while not accepted:
        if attempt >= MAX_ATTEMPTS:
            raise RuntimeError(
                f"Fold {fold_idx}: could not find a valid split after "
                f"{MAX_ATTEMPTS} attempts. Check class distribution."
            )

        # Randomly draw 4 test subjects; remaining 36 are training
        test_subjects  = rng_setup.choice(unique_subjects, N_TEST_SUBJECTS, replace=False)
        train_subjects = unique_subjects[~np.isin(unique_subjects, test_subjects)]

        # Get indices for both groups
        ab_train_all_idx = np.where(np.isin(groups_AB, train_subjects))[0]
        ab_test_idx = np.where(np.isin(groups_AB, test_subjects))[0]
        pd_train_idx = np.where(np.isin(groups_PD, train_subjects))[0]
        pd_test_idx = np.where(np.isin(groups_PD, test_subjects))[0]

        # Validate: all 8 classes must appear in every split
        splits_valid = (
            fold_has_all_classes(ab_train_all_idx, y_AB, all_classes) and
            fold_has_all_classes(ab_test_idx, y_AB, all_classes) and
            fold_has_all_classes(pd_train_idx, y_PD, all_classes) and
            fold_has_all_classes(pd_test_idx, y_PD, all_classes)
        )

        attempt += 1

        if not splits_valid:
            print(
                f"fold {fold_idx+1} attempt {attempt}: "
                f"class missing from a split — resampling subjects",
                flush=True
            )
            continue

        #subsample AB train to match PD train count
        n_pd_train = len(pd_train_idx)
        ab_train_sub_idx = subsample_fair(
            ab_train_all_idx, groups_AB, n_pd_train, rng_setup
        )

        n_pd_test = len(pd_test_idx)
        ab_test_idx = subsample_fair(ab_test_idx, groups_AB, n_pd_test, rng_setup)

        #confirm subsampled AB train also has all classes
        if not fold_has_all_classes(ab_train_sub_idx, y_AB, all_classes):
            print(
                f"fold {fold_idx+1} attempt {attempt}: "
                f"class missing after AB subsampling — resampling subjects",
                flush=True
            )
            continue

        if not fold_has_all_classes(ab_test_idx, y_AB, all_classes):
            print(
                f"fold {fold_idx+1} attempt {attempt}: "
                f"class missing after AB test set subsampling — resampling subjects",
                flush=True
            )
            continue

        accepted = True
        print(
            f"fold {fold_idx+1:2d} accepted on attempt {attempt:3d} | "
            f"train subj: {len(train_subjects)} | test subj: {len(test_subjects)} | "
            f"PD_train={n_pd_train} | AB_train_sub={len(ab_train_sub_idx)} | ",
            f"AB_test_sub={len(ab_test_idx)}",
            flush=True
        )

    cross_task_folds.append({
        "fold_idx" : fold_idx,
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "pd_train_idx": pd_train_idx,
        "pd_test_idx": pd_test_idx,
        "ab_test_idx": ab_test_idx,
        "ab_train_sub_idx": ab_train_sub_idx,
        "n_pd_train": n_pd_train,
        "n_ab_train_sub": len(ab_train_sub_idx),
    })

    print(
        f"fold {fold_idx+1:2d}/{N_SPLITS_CROSS} | "
        f"train subj: {len(train_subjects)} | test subj: {len(test_subjects)} | "
        f"PD_train={n_pd_train} | AB_train_sub={len(ab_train_sub_idx)}",
        flush=True
    )

# crosstask classification for each MASK 

cross_task_results = defaultdict(dict)   

for set_idx, (mask_list, mask_names, set_name) in enumerate(zip(
    [list_masks_contrast, list_masks_conjunction],
    [list_masks_names_contrast, list_masks_names_conjunction],
    ["contrast", "conjunction"]
)):
    print(f"\n  Cross-task mask set {set_idx+1}/2: {set_name}", flush=True)

    for j, (mask_img, mask_name) in enumerate(zip(mask_list, mask_names)):
        print(f"    Mask {j+1}/{len(mask_list)}: {mask_name}", flush=True)
        t_mask = time.time()

        # Extract ROI features for all AB and all PD images once
        # NEW: NiftiMasker is used here (same as in permute.py) rather than
        # Nilearn's Decoder, because we need full control over the train/test split.
        masker = NiftiMasker(mask_img=mask_img)
        X_AB_full = masker.fit_transform(x_f_AB)   # (n_AB_samples, n_voxels)
        X_PD_full = masker.transform(x_f_PD)       # (n_PD_samples, n_voxels)

        fold_scores_pd2ab = []
        fold_scores_ab2pd = []

        for fold in cross_task_folds:
            pd_tr = fold["pd_train_idx"]
            pd_te = fold["pd_test_idx"]
            ab_te = fold["ab_test_idx"]
            ab_tr = fold["ab_train_sub_idx"]

            # ── Direction 1: train on PD → test on AB ──────────────
            # Train on all PD data from 36 subjects; test on AB data
            # from the 4 held-out subjects.
            svc_pd2ab = LinearSVC(penalty="l2", max_iter=2000)
            svc_pd2ab.fit(X_PD_full[pd_tr], y_PD[pd_tr])
            score_pd2ab = roc_auc_ovr_scorer(svc_pd2ab, X_AB_full[ab_te], y_AB[ab_te])
            fold_scores_pd2ab.append(score_pd2ab)

            # ── Direction 2: train on AB (subsampled) → test on PD ─
            # Train on the matched-count AB subsample from 36 subjects;
            # test on PD data from the 4 held-out subjects.
            svc_ab2pd = LinearSVC(penalty="l2", max_iter=2000)
            svc_ab2pd.fit(X_AB_full[ab_tr], y_AB[ab_tr])
            score_ab2pd = roc_auc_ovr_scorer(svc_ab2pd, X_PD_full[pd_te], y_PD[pd_te])
            fold_scores_ab2pd.append(score_ab2pd)

        fold_scores_pd2ab = np.array(fold_scores_pd2ab)   # shape (10, 8)
        fold_scores_ab2pd = np.array(fold_scores_ab2pd)   # shape (10, 8)

        mean_per_class_pd2ab = np.mean(fold_scores_pd2ab, axis=0)   # shape (8,)
        mean_per_class_ab2pd = np.mean(fold_scores_ab2pd, axis=0)   # shape (8,)

   
        #store per-class results
        cross_task_results[set_name][mask_name] = {
            class_name: {
                "PD_to_AB": round(float(mean_per_class_pd2ab[i]), 5),
                "AB_to_PD": round(float(mean_per_class_ab2pd[i]), 5),
            }
            for i, class_name in enumerate(all_classes)
        }

        print(f"per-class AUC:", flush=True)
        for i, class_name in enumerate(all_classes):
            print(
                f"{class_name}: PD→AB={mean_per_class_pd2ab[i]:.4f} | "
                f"AB→PD={mean_per_class_ab2pd[i]:.4f}",
                flush=True
            )

#cross task results to excel
for set_name in ["contrast", "conjunction"]:
    rows = []
    for roi, class_scores in cross_task_results[set_name].items():
        for class_name, scores in class_scores.items():
            rows.append({
                "ROI" : roi,
                "Class" : class_name,
                "PD_to_AB" : scores["PD_to_AB"],
                "AB_to_PD" : scores["AB_to_PD"],
            })
    df_ct = pd.DataFrame(rows)
    ct_excel_path = os.path.join(base_dir, f"cross_task_results_{set_name}.xlsx")
    df_ct.to_excel(ct_excel_path, index=False)
    print(f"  Cross-task saved: {ct_excel_path}", flush=True)

# ── SAVE PERMUTATION INPUTS ────────────────────────────────────
print("\nSaving permutation inputs for permute.py...", flush=True)

perm_inputs = {
    "x_f_AB" : x_f_AB,
    "x_f_PD" : x_f_PD,
    "y_AB" : y_AB,
    "y_PD" : y_PD,
    "groups_AB" : groups_AB,
    "groups_PD" : groups_PD,
    "n_splits" : 10,
    "list_masks_contrast" : list_masks_contrast,
    "list_masks_names_contrast" : list_masks_names_contrast,
    "list_masks_conjunction" : list_masks_conjunction,
    "list_masks_names_conjunction" : list_masks_names_conjunction,
    "cross_task_folds" : cross_task_folds,
}

perm_pkl_path = os.path.join(output_dir, "permutation_inputs.pkl")
with open(perm_pkl_path, "wb") as f:
    pickle.dump(perm_inputs, f)

print(f"Saved: {perm_pkl_path}", flush=True)
print("\n" + "=" * 60, flush=True)
print("SETUP COMPLETE — you can now submit the permutation array job", flush=True)
print("=" * 60, flush=True)
