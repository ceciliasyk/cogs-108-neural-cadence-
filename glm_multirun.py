"""
Multi-run first-level GLM helper for Nilearn.

This script concatenates multiple runs for a subject into a single 1st-level design matrix
and adds run-specific intercept regressors so you can contrast conditions that are
present only in some runs (e.g., run 1=classical, run 2=synthetic, ...).

Usage: edit PROJECT_ROOT, SUBJECT, RUNS and TR at the bottom and run the script.

Outputs: z-maps written to PROJECT_ROOT/glm_results/<subject>/<combined_run_name>/

Notes/assumptions:
- Each run has its own preprocessed BOLD NIfTI, confounds .tsv, and events .tsv.
- Event files contain onset/duration/trial_type columns (trial_type values like 'classical' or 'synthetic').
- Confound selection follows the notebook pattern; missing confounds will be ignored.
"""

import os
import numpy as np
import pandas as pd
from nilearn import image, plotting
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix


def build_multirun_design_and_images(project_root, subject, runs, tr,
                                     fmri_template=None, mask_template=None,
                                     confounds_template=None, events_template=None,
                                     confound_vars=None):
    """
    Build concatenated fmri image, concatenated confounds, combined events (with onset offsets),
    and run indicator regressors.

    Returns:
        fmri_img_concat: 4D image with all runs concatenated in time
        confounds_concat: DataFrame of confounds for all scans (rows == total scans)
        design_frame_times: np.array of frame times for the concatenated image
        combined_events: DataFrame suitable for make_first_level_design_matrix (onset in seconds)
        run_regs: ndarray shape (n_timepoints, n_runs) run indicator regressors
        mask_img_path: path or None
    """
    # templates default to same patterns used in the notebook
    if fmri_template is None:
        fmri_template = os.path.join(project_root, "fmriprep", subject, "func",
                                     f"{subject}_task-music_{{run}}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    if mask_template is None:
        mask_template = os.path.join(project_root, "fmriprep", subject, "func",
                                     f"{subject}_task-music_{{run}}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
    if confounds_template is None:
        confounds_template = os.path.join(project_root, "fmriprep", subject, "func",
                                          f"{subject}_task-music_{{run}}_desc-confounds_timeseries.tsv")
    if events_template is None:
        events_template = os.path.join(project_root, "rawdata_or_ds002725", subject, "func",
                                       f"{subject}_task-music_{{run}}_events.tsv")
    if confound_vars is None:
        confound_vars = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "framewise_displacement"]

    fmri_imgs = []
    masks = []
    all_confounds = []
    events_list = []
    n_scans_per_run = []

    # Load images, confounds, events; record scan counts
    for run in runs:
        fmri_path = fmri_template.format(run=run)
        mask_path = mask_template.format(run=run)
        conf_path = confounds_template.format(run=run)
        events_path = events_template.format(run=run)

        if not os.path.exists(fmri_path):
            raise FileNotFoundError(f"fMRI file missing: {fmri_path}")
        fmri_img = image.load_img(fmri_path)
        fmri_imgs.append(fmri_img)

        if os.path.exists(mask_path):
            masks.append(mask_path)

        if os.path.exists(conf_path):
            conf_df = pd.read_table(conf_path)
        else:
            # create an empty confounds frame (we'll fill with zeros later)
            conf_df = pd.DataFrame()
        # keep only desired confounds that exist
        desired = [c for c in confound_vars if c in conf_df.columns]
        conf_df = conf_df[desired].fillna(0)
        all_confounds.append(conf_df)

        if os.path.exists(events_path):
            ev = pd.read_table(events_path)
            # expect columns: onset,duration,trial_type
            if 'onset' not in ev.columns or 'duration' not in ev.columns or 'trial_type' not in ev.columns:
                raise ValueError(f"Events file {events_path} must contain onset,duration,trial_type columns")
        else:
            # empty events for this run
            ev = pd.DataFrame(columns=['onset', 'duration', 'trial_type'])
        events_list.append(ev)

        n_scans = fmri_img.shape[-1]
        n_scans_per_run.append(n_scans)

    # concat fmri imgs
    fmri_img_concat = image.concat_imgs(fmri_imgs)
    n_total_scans = fmri_img_concat.shape[-1]

    # concatenate confounds, aligning columns across runs (missing columns -> fill 0)
    # ensure each confounds df has rows == n_scans for that run; if not, try to pad/truncate
    conf_dfs_fixed = []
    for df, n_scans in zip(all_confounds, n_scans_per_run):
        if df.shape[0] < n_scans:
            # pad with zeros
            pad = pd.DataFrame(0, index=np.arange(n_scans - df.shape[0]), columns=df.columns)
            df = pd.concat([df, pad], ignore_index=True)
        elif df.shape[0] > n_scans:
            df = df.iloc[:n_scans].reset_index(drop=True)
        conf_dfs_fixed.append(df)

    # find union of columns
    all_cols = sorted({c for df in conf_dfs_fixed for c in df.columns})
    for i, df in enumerate(conf_dfs_fixed):
        for c in all_cols:
            if c not in df.columns:
                df[c] = 0
        # ensure same column order
        conf_dfs_fixed[i] = df[all_cols].reset_index(drop=True)

    confounds_concat = pd.concat(conf_dfs_fixed, ignore_index=True)

    # Build combined events with onset offsets (in seconds). Keep trial_type as 'trial_type'.
    combined_events_frames = []
    run_start_time = 0.0
    for ev, n_scans in zip(events_list, n_scans_per_run):
        ev_copy = ev.copy()
        if ev_copy.shape[0] > 0:
            ev_copy['onset'] = ev_copy['onset'] + run_start_time
            combined_events_frames.append(ev_copy)
        run_start_time += n_scans * tr

    if len(combined_events_frames) > 0:
        combined_events = pd.concat(combined_events_frames, ignore_index=True)
    else:
        # empty events
        combined_events = pd.DataFrame(columns=['onset', 'duration', 'trial_type'])

    # frame times for concatenated image
    frame_times = np.arange(n_total_scans) * tr

    # build run indicator regressors
    run_regs = np.zeros((n_total_scans, len(runs)))
    idx0 = 0
    for i, n in enumerate(n_scans_per_run):
        idx1 = idx0 + n
        run_regs[idx0:idx1, i] = 1
        idx0 = idx1

    mask_img_path = masks[0] if len(masks) > 0 else None

    return fmri_img_concat, confounds_concat, frame_times, combined_events, run_regs, mask_img_path


def run_first_level_multirun(project_root, subject, runs, tr, out_subdir_name=None):
    fmri_img_concat, confounds_concat, frame_times, combined_events, run_regs, mask_path = build_multirun_design_and_images(
        project_root, subject, runs, tr)

    # Add names for run regressors
    run_reg_names = [f"run_{i+1}" for i in range(len(runs))]

    print("Design building: building design matrix (this may take a moment)")
    design_matrix = make_first_level_design_matrix(
        frame_times,
        events=combined_events,
        hrf_model='glover',
        add_regs=run_regs,
        add_reg_names=run_reg_names,
    )

    print("Design matrix columns:", design_matrix.columns.tolist())

    # Fit first level model
    flm = FirstLevelModel(
        t_r=tr,
        slice_time_ref=0.5,
        mask_img=mask_path,
        hrf_model='glover',
        drift_model='cosine',
        high_pass=1/128,
    )

    print("Fitting model to concatenated runs...")
    flm = flm.fit(fmri_img_concat, design_matrices=design_matrix, confounds=confounds_concat)

    # Build contrasts
    columns = list(design_matrix.columns)

    def make_contrast(cond_pos, cond_neg=None):
        # cond_pos/cnd_neg can be strings or list of strings
        vec = np.zeros(len(columns))
        if cond_pos is not None:
            for name in (cond_pos if isinstance(cond_pos, (list, tuple)) else [cond_pos]):
                if name in columns:
                    vec[columns.index(name)] = 1
        if cond_neg is not None:
            for name in (cond_neg if isinstance(cond_neg, (list, tuple)) else [cond_neg]):
                if name in columns:
                    vec[columns.index(name)] = -1
        return vec

    # classical > synthetic
    contrast_classical_gt_synth = make_contrast('classical', 'synthetic')
    # synthetic > classical
    contrast_synth_gt_classical = make_contrast('synthetic', 'classical')
    # music (classical + synthetic) > baseline
    contrast_music_vs_baseline = make_contrast(['classical', 'synthetic'], None)

    print("Computing contrasts (z-maps)...")
    z_music = flm.compute_contrast(contrast_music_vs_baseline, output_type='z_score')
    z_class_vs_synth = flm.compute_contrast(contrast_classical_gt_synth, output_type='z_score')
    z_synth_vs_class = flm.compute_contrast(contrast_synth_gt_classical, output_type='z_score')

    # Save outputs
    out_dir = os.path.join(project_root, 'glm_results', subject, out_subdir_name or 'multirun_concat')
    os.makedirs(out_dir, exist_ok=True)

    music_path = os.path.join(out_dir, 'zmap_music_vs_baseline.nii.gz')
    class_path = os.path.join(out_dir, 'zmap_classical_gt_synthetic.nii.gz')
    synth_path = os.path.join(out_dir, 'zmap_synthetic_gt_classical.nii.gz')

    z_music.to_filename(music_path)
    z_class_vs_synth.to_filename(class_path)
    z_synth_vs_class.to_filename(synth_path)

    print('Saved:', music_path)
    print('Saved:', class_path)
    print('Saved:', synth_path)

    # Quick plotting examples (optional)
    try:
        plotting.plot_stat_map(z_music, title=f"{subject} music > baseline", threshold=3.1, display_mode='z', cut_coords=7)
        plotting.show()
    except Exception:
        print('Plotting failed or running headless. Skipping plots.')

    return {'music': music_path, 'classical_gt_synthetic': class_path, 'synthetic_gt_classical': synth_path}


if __name__ == '__main__':
    # === USER EDITS: set these for your environment ===
    PROJECT_ROOT = '/path/to/neurodesktop-storage/project'  # change me
    SUBJECT = 'sub-01'
    # List all run identifiers exactly as they appear in the filenames (e.g., 'run-1', 'run-2', ...)
    RUNS = ['run-1', 'run-2', 'run-3', 'run-4', 'run-5']
    TR = 2.0  # replace with your dataset TR (seconds)

    # Quick run
    out = run_first_level_multirun(PROJECT_ROOT, SUBJECT, RUNS, TR)
    print('Done. Outputs:', out)
