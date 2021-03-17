import sys
import warnings
import os
from pathlib import Path

if (sys.version_info < (3, 0)):
    warnings.warn("mldatasets only supports Python 3 (not 2)!")

__version__ = '0.01.15'

from mldatasets.config import init, load
from mldatasets.common import runcmd, make_dummies_with_limits, make_dummies_from_dict, evaluate_multiclass_metrics_mdl, evaluate_multiclass_mdl, evaluate_reg_metrics_mdl, evaluate_reg_mdl, evaluate_class_metrics_mdl, evaluate_class_mdl, plot_3dim_decomposition, encode_classification_error_vector, describe_cf_instance, create_decision_plot, plot_data_vs_ice, img_np_from_fig, compare_img_pred_viz, heatmap_overlay, find_closest_datapoint_idx, approx_predict_ts, compare_confusion_matrices, discretize, plot_prob_progression, plot_prob_contour_map, compare_image_predictions, profits_by_thresh, compare_df_plots, compute_aif_metrics
from mldatasets.sources.kaggle import Kaggle
from mldatasets.sources.url import URL

init(os.path.join(Path().parent.absolute(), 'data'))
