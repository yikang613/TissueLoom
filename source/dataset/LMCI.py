from scipy.io import loadmat
import numpy as np
import torch
import os
from nilearn.connectome import ConnectivityMeasure
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict

def load_LMCI_data(cfg: DictConfig):
    folders = {"LMCI": 1, "CN": 0}

    # --- Input type selection ---
    # All types use 248x248 FC and 248-dim timeseries for fair comparison.
    # gray:             GM-GM block only (top-left 200x200 non-zero)
    # white:            WM-WM block only (bottom-right 48x48 non-zero)
    # white_gray_cross: cross-tissue blocks only (GM-WM off-diagonals non-zero)
    # white_gray_full:  full FC (all blocks non-zero)
    input_type = getattr(cfg.dataset, 'input_type', 'white_gray_cross')
    valid_types = ('gray', 'white', 'white_gray_cross', 'white_gray_full')
    assert input_type in valid_types, \
        f"Unknown input_type: {input_type}. Use: {valid_types}"

    final_timeseries = []
    final_pearson = []
    final_labels = []

    for folder, label in folders.items():
        folder_path = os.path.join(cfg.dataset.path, folder)
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".mat"):
                    file_path = os.path.join(root, file)
                    data = loadmat(file_path)
                    gray_matter = data.get('SCH2_time')  # [200, T]
                    white_matter = np.nan_to_num(data.get('Eve_time'))  # [48, T]
                    gray_matter = gray_matter[:, :140] if gray_matter.shape[1] > 140 else gray_matter
                    white_matter = white_matter[:, :140] if white_matter.shape[1] > 140 else white_matter

                    gray_matter_transposed = gray_matter.T   # [T, 200]
                    white_matter_transposed = white_matter.T  # [T, 48]

                    # Always use full concatenated timeseries [T, 248]
                    # so all input types share the same 248-dim input space
                    timeseries_data = np.hstack([
                        gray_matter_transposed,
                        white_matter_transposed
                    ])                                              # [T, 248]

                    num_gray = gray_matter.shape[0]   # 200
                    num_white = white_matter.shape[0]  # 48

                    # --- Compute full 248x248 correlation matrix ---
                    correlation_measure = ConnectivityMeasure(
                        kind="correlation",
                        standardize="zscore_sample",
                    )
                    correlation_matrix = correlation_measure.fit_transform(
                        [timeseries_data]
                    )[0]
                    correlation_matrix[correlation_matrix < 0.1] = 0

                    # --- Mask FC matrix based on input type ---
                    # All types use 248x248; only the non-zero blocks differ
                    fc = np.zeros_like(correlation_matrix)

                    if input_type == 'gray':
                        # Keep GM-GM block only (top-left 200x200)
                        fc[:num_gray, :num_gray] = \
                            correlation_matrix[:num_gray, :num_gray]

                    elif input_type == 'white':
                        # Keep WM-WM block only (bottom-right 48x48)
                        fc[num_gray:, num_gray:] = \
                            correlation_matrix[num_gray:, num_gray:]

                    elif input_type == 'white_gray_cross':
                        # Cross-tissue only: GM-WM and WM-GM blocks
                        fc[:num_gray, num_gray:] = \
                            correlation_matrix[:num_gray, num_gray:]
                        fc[num_gray:, :num_gray] = \
                            correlation_matrix[num_gray:, :num_gray]

                    elif input_type == 'white_gray_full':
                        # Full FC: all blocks
                        fc = correlation_matrix

                    final_pearson.append(fc)

                    final_timeseries.append(timeseries_data)
                    final_labels.append(label)

    # Convert to tensors
    final_timeseries_tensor = torch.tensor(
        np.array(final_timeseries), dtype=torch.float)
    # Transpose from (B, T, N) to (B, N, T) so models get ROI-first format
    final_timeseries_tensor = final_timeseries_tensor.transpose(1, 2)
    final_pearson_tensor = torch.tensor(
        np.array(final_pearson), dtype=torch.float)
    final_labels_tensor = torch.tensor(
        final_labels, dtype=torch.float)

    with open_dict(cfg):
        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson_tensor.shape[1:]
        cfg.dataset.timeseries_sz = final_timeseries_tensor.shape[2]

    print(f"[Data] input_type={input_type}, "
          f"FC shape={final_pearson_tensor.shape[1:]}, "
          f"n_subjects={len(final_labels)}")

    return final_timeseries_tensor, final_pearson_tensor, final_labels_tensor