import subprocess
import sys
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt
from matplotlib import cm
from alibi.utils.mapping import ohe_to_ord, ord_to_ohe
import statsmodels.api as sm
from mlxtend.plotting import plot_decision_regions
from pycebox.ice import ice, ice_plot
from itertools import cycle
import seaborn as sns
import io
import cv2
from scipy.spatial import distance
from tqdm.notebook import trange

def runcmd(cmd, verbose=False):
    sproc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, 
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, )
    output = ''
    numlines = 0
    error = True
    while True:
        if error:
            line = sproc.stderr.readline().decode("utf-8")
            if line == '' and (sproc.poll() is None or sproc.poll() == 0):
                error = False
        if not error:
            line = sproc.stdout.readline().decode("utf-8")
        if line == '' and sproc.poll() is not None:
            break
        if verbose:
            sys.stdout.write(line)
            sys.stdout.flush()
        output = output + line
        numlines = numlines + 1
    return error, output.strip(), numlines

def make_dummies_with_limits(df, colname, min_recs=0.005,\
                             max_dummies=20, defcatname='Other',\
                             nospacechr='_'):
    if min_recs < 1:
        min_recs = df.shape[0]*min_recs
    topvals_df = df.groupby(colname).size().reset_index(name="counts").\
                    sort_values(by="counts", ascending=False).reset_index()
    other_l = topvals_df[(topvals_df.index > max_dummies) |\
                         (topvals_df.counts < min_recs)][colname].to_list()
    if len(other_l):
        df.loc[df[colname].isin(other_l), colname] = defcatname
    if len(nospacechr):
        df[colname] = df[colname].str.replace(' ',\
                                                  nospacechr, regex=False)
    return pd.get_dummies(df, prefix=[colname], columns=[colname])

def make_dummies_from_dict(df, colname, match_dict, 
                           drop_orig=True, nospacechr='_'):
    if type(match_dict) is list:
        if len(nospacechr):
            match_dict = {match_key:match_key.\
                              replace(' ', nospacechr)\
                              for match_key in match_dict }
        else:
            match_dict = {match_key:match_key\
                              for match_key in match_dict}
    for match_key in match_dict.keys():
        df[colname+'_'+match_dict[match_key]] =\
                    np.where(df[colname].str.contains(match_key), 1, 0)
    if drop_orig:
        return df.drop([colname], axis=1)
    else:
        return df
    
def evaluate_class_mdl(fitted_model, X_train, X_test, y_train, y_test, predopts={}):
    y_train_pred = fitted_model.predict(X_train, **predopts).squeeze()
    if len(np.unique(y_train_pred)) > 2:
        y_train_pred = np.where(y_train_pred > 0.5, 1, 0)
        y_test_prob = fitted_model.predict(X_test, **predopts).squeeze()
        y_test_pred = np.where(y_test_prob > 0.5, 1, 0)
    else:   
        y_test_prob = fitted_model.predict_proba(X_test, **predopts)[:,1]
        y_test_pred = np.where(y_test_prob > 0.5, 1, 0)
    roc_auc = metrics.roc_auc_score(y_test, y_test_prob)
    plt.figure(figsize = (12,12))
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    fpr, tpr, _ = metrics.roc_curve(y_test, y_test_prob)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # coin toss line
    plt.xlabel('False Positive Rate', fontsize = 14)
    plt.ylabel('True Positive Rate', fontsize = 14)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc="lower right")
    plt.show()
    print('Accuracy_train:  %.4f\t\tAccuracy_test:   %.4f' %\
                        (metrics.accuracy_score(y_train, y_train_pred),\
                         metrics.accuracy_score(y_test, y_test_pred)))
    print('Precision_test:  %.4f\t\tRecall_test:     %.4f' %\
                        (metrics.precision_score(y_test, y_test_pred),\
                         metrics.recall_score(y_test, y_test_pred)))
    print('ROC-AUC_test:    %.4f\t\tF1_test:         %.4f\t\tMCC_test: %.4f' %\
                        (roc_auc,\
                         metrics.f1_score(y_test, y_test_pred),\
                         metrics.matthews_corrcoef(y_test, y_test_pred)))
    return y_train_pred, y_test_prob, y_test_pred

def evaluate_multiclass_mdl(fitted_model, X, y, class_l, ohe=None, plot_roc=False, plot_roc_class=True,\
                           plot_conf_matrix=True, pct_matrix=True, plot_class_report=True, predopts={}):
    if not isinstance(X, (np.ndarray)) or not isinstance(y, (list, tuple, np.ndarray)):
        raise Exception("Data is not in the right format")
    n_classes = len(class_l)
    y = np.array(y)
    if len(y.shape)==1:
        y = np.expand_dims(y, axis=1)
    if y.shape[1] == 1:
        if isinstance(ohe, (preprocessing._encoders.OneHotEncoder)):
            y_ohe = ohe.transform(y)
        else:
            raise Exception("Sklearn one-hot encoder is a required parameter when labels aren't already encoded")
    elif y_ohe.shape[1] == n_classes:
        y_ohe = y.copy()
        y = np.array([[class_l[o]] for o in np.argmax(y_ohe, axis=1)])
    else:
        raise Exception("Labels don't have dimensions that match the classes")
    y_prob = fitted_model.predict(X, **predopts)
    if y_prob.shape[1] == n_classes:
        y_pred_ord = np.argmax(y_prob, axis=1)
        y_pred = [class_l[o] for o in y_pred_ord]
    else:
        raise Exception("List of classes provided doesn't match the dimensions of model predictions")
    if plot_roc:
        #Compute FPR/TPR metrics for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_ohe[:, i], y_prob[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_ohe.ravel(), y_prob.ravel())
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

        # Compute interpolated macro and micro
        fpr["macro"] = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        tpr["macro"] = np.zeros_like(fpr["macro"])
        for i in range(n_classes):
            tpr["macro"] += np.interp(fpr["macro"], fpr[i], tpr[i])
        tpr["macro"] /= n_classes
        roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

        # Plot all ROCs
        plt.figure(figsize = (12,12))
        plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='navy', linestyle=':', linewidth=4)
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='deeppink', linestyle=':', linewidth=4)
        if plot_roc_class:
            for i in range(n_classes):
                plt.plot(fpr[i], tpr[i],
                         label='ROC for class {0} (area = {1:0.2f})'
                         ''.format(class_l[i], roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--') # coin toss line
        plt.xlabel('False Positive Rate', fontsize = 14)
        plt.ylabel('True Positive Rate', fontsize = 14)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.legend(loc="lower right")
        plt.show()

    if plot_conf_matrix: 
        conf_matrix = metrics.confusion_matrix(y, y_pred, labels=class_l)
        plt.figure(figsize=(12, 11))
        if pct_matrix:
            sns.heatmap(conf_matrix/np.sum(conf_matrix), annot=True, xticklabels=class_l, yticklabels=class_l,\
                        fmt='.0%', cmap='Blues', annot_kws={'size':12})
        else:
            sns.heatmap(conf_matrix, annot=True, xticklabels=class_l, yticklabels=class_l,\
                        cmap='Blues', annot_kws={'size':12})
        plt.show()

    if plot_class_report:
        print(metrics.classification_report(y, y_pred, digits=3, zero_division=0))
    
    return y_pred, y_prob

def evaluate_reg_mdl(fitted_model, X_train, X_test, y_train, y_test, scaler=None,\
                     y_truncate=False, predopts={}):
    y_train_ = y_train.copy()
    y_test_ = y_train.copy()
    if not isinstance(X_train, (np.ndarray, tuple, list)) or X_train.shape[1] != y_train.shape[1]: 
        y_train_pred = fitted_model.predict(X_train, **predopts)
    else:
        y_train_pred = X_train.copy()
    if not isinstance(X_test, (np.ndarray, tuple, list)) or X_test.shape[1] != y_test.shape[1]:
        y_test_pred = fitted_model.predict(X_test, **predopts)
    else:
        y_test_pred = X_test.copy()
    if y_truncate:
        y_train_ = y_train_[-y_train_pred.shape[0]:]
        y_test_ = y_test_[-y_test_pred.shape[0]:]
    if scaler is not None:
        y_train_ = scaler.inverse_transform(y_train_)
        y_test_ = scaler.inverse_transform(y_test_)
        y_train_pred = scaler.inverse_transform(y_train_pred)
        y_test_pred = scaler.inverse_transform(y_test_pred)
    RMSE_reg_train = math.sqrt(\
                        metrics.mean_squared_error(y_train_,\
                                                   y_train_pred))
    RMSE_reg_test = math.sqrt(\
                        metrics.mean_squared_error(y_test_,
                                                   y_test_pred))
    R2_reg_test = metrics.r2_score(y_test_,\
                                   y_test_pred)
    plt.figure(figsize = (12,12))
    plt.ylabel('Predicted', fontsize = 14)
    plt.scatter(y_test_, y_test_pred)
    sns.regplot(x=y_test_, y=y_test_pred, color="g", ci)
    plt.xlabel('Observed', fontsize = 14)
    plt.show()     
    print('RMSE_train: %.4f\tRMSE_test: %.4f\tr2: %.4f' %\
                        (RMSE_reg_train, RMSE_reg_test, R2_reg_test))
    if y_truncate:
        return y_train_pred, y_test_pred, y_train_, y_test_
    else:
        return y_train_pred, y_test_pred

def plot_3dim_decomposition(Z, y_labels, y_names):
    if len(y_names) > 2:
        cmap = 'plasma_r'
    else:
        cmap = 'viridis'
    fig, axs = plt.subplots(1, 3, figsize = (16,4))
    fig.subplots_adjust(hspace=0, wspace=0.3)
    scatter = axs[0].scatter(Z[:,0], Z[:,1],\
                             c=y_labels, alpha=0.5, cmap=cmap)
    legend = axs[0].legend(*scatter.legend_elements(), loc='best')
    for n in y_names.keys(): 
        legend.get_texts()[n].set_text(y_names[n])
    axs[0].set_xlabel('x', fontsize = 12)
    axs[0].set_ylabel('y', fontsize = 12)
    scatter = axs[1].scatter(Z[:,1], Z[:,2],\
                   c=y_labels, alpha=0.5, cmap=cmap)
    legend = axs[1].legend(*scatter.legend_elements(), loc='best')
    for n in y_names.keys(): 
        legend.get_texts()[n].set_text(y_names[n])
    axs[1].set_xlabel('y', fontsize = 12)
    axs[1].set_ylabel('z', fontsize = 12)
    axs[2].scatter(Z[:,0], Z[:,2],\
                   c=y_labels, alpha=0.5, cmap=cmap)
    legend = axs[2].legend(*scatter.legend_elements(), loc='best')
    for n in y_names.keys(): 
        legend.get_texts()[n].set_text(y_names[n])
    axs[2].set_xlabel('x', fontsize = 12)
    axs[2].set_ylabel('z', fontsize = 12)
    plt.show()
    
def encode_classification_error_vector(y_true, y_pred):
    error_vector = (y_true * 2) - y_pred
    error_vector = np.where(error_vector==0, 4, error_vector + 1)
    error_vector = np.where(error_vector==3, 0, error_vector - 1)
    error_vector = np.where(error_vector==3, error_vector, error_vector + 1)
    error_labels = {0:'FP', 1:'FN', 2:'TP', 3:'TN'}
    return error_vector, error_labels

def describe_cf_instance(X, explanation, class_names, cat_vars_ohe=None, category_map=None, feature_names=None, eps=1e-2):
    print("Instance Outcomes and Probabilities")
    print("-" * 48)
    max_len = len(max(feature_names, key=len))
    print('{}:  {}\r\n{}   {}'.format('original'.rjust(max_len),
                                        class_names[explanation.orig_class],
                                        " " * max_len,
                                        explanation.orig_proba[0]))
    if explanation.cf != None:     
        print('{}:  {}\r\n{}   {}'.format('counterfactual'.rjust(max_len),
                                            class_names[explanation.cf['class']],
                                            " " * max_len,
                                            explanation.cf['proba'][0]))
        print("\r\nCategorical Feature Counterfactual Perturbations")
        print("-" * 48)
        X_orig_ord = ohe_to_ord(X, cat_vars_ohe)[0]
        try:
            X_cf_ord = ohe_to_ord(explanation.cf['X'], cat_vars_ohe)[0]
        except:
            X_cf_ord = ohe_to_ord(explanation.cf['X'].transpose(), cat_vars_ohe)[0].transpose()
        delta_cat = {}
        for _, (i, v) in enumerate(category_map.items()):
            cat_orig = v[int(X_orig_ord[0, i])]
            cat_cf = v[int(X_cf_ord[0, i])]
            if cat_orig != cat_cf:
                delta_cat[feature_names[i]] = [cat_orig, cat_cf]
        if delta_cat:
            for k, v in delta_cat.items():
                print("\t{}:  {}  -->  {}".format(k.rjust(max_len), v[0], v[1]))
        print("\r\nNumerical Feature Counterfactual Perturbations")
        print("-" * 48)
        num_idxs = [i for i in list(range(0,len(feature_names)))\
                    if i not in category_map.keys()]
        delta_num = X_cf_ord[0, num_idxs] - X_orig_ord[0, num_idxs]
        for i in range(delta_num.shape[0]):
            if np.abs(delta_num[i]) > eps:
                print("\t{}:  {:.2f}  -->  {:.2f}".format(feature_names[i].rjust(max_len),
                                                X_orig_ord[0,i],
                                                X_cf_ord[0,i]))
    else:
        print("\tNO COUNTERFACTUALS")

def create_decision_plot(X, y, model, feature_index, feature_names, X_highlight, 
                         filler_feature_values, filler_feature_ranges, ax=None):
    filler_values = dict((k, filler_feature_values[k]) for k in filler_feature_values.keys() if k not in feature_index)
    filler_ranges = dict((k, filler_feature_ranges[k]) for k in filler_feature_ranges.keys() if k not in feature_index)
    ax = plot_decision_regions(sm.add_constant(X).to_numpy(), y.to_numpy(), clf=model, 
                          feature_index=feature_index,
                          X_highlight=X_highlight,
                          filler_feature_values=filler_values, 
                          filler_feature_ranges=filler_ranges, 
                          scatter_kwargs = {'s': 48, 'edgecolor': None, 'alpha': 0.7},
                          contourf_kwargs = {'alpha': 0.2}, legend=2, ax=ax)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    return ax

def plot_data_vs_ice(pred_function, ylabel, X, feature_name, feature_label, color_by=None, legend_key=None, alpha=0.15):
    ice_df = ice(X, feature_name,\
             pred_function, num_grid_points=None)
    fig, axs = plt.subplots(2, 1, sharex=False, sharey=True,\
                            figsize=(15,20))
    fig.subplots_adjust(hspace=0.15, wspace=0)
    if color_by is None or legend_key is None:
        scatter = axs[0].scatter(X[feature_name],\
                                 pred_function(X),\
                                 alpha=alpha)
        ice_plot(ice_df, alpha=alpha, ax=axs[1])
    else:
        scatter = axs[0].scatter(X[feature_name],\
                                 pred_function(X),\
                                 c=X[color_by], alpha=alpha)
        legend = axs[0].legend(*scatter.legend_elements(), loc='best')
        for s in legend_key.keys(): 
            legend.get_texts()[s].set_text(legend_key[s])
        ice_plot(ice_df, color_by=color_by, alpha=alpha, ax=axs[1])
    axs[0].set_xlabel(feature_label, fontsize=12)
    axs[0].set_ylabel(ylabel, fontsize=12)
    axs[0].set_title('Data', fontsize=16)
    axs[1].set_xlabel(feature_label, fontsize=12)
    axs[1].set_ylabel(ylabel, fontsize=12)
    axs[1].set_title('ICE Curves', fontsize=16)
    plt.show()

def img_np_from_fig(fig, dpi=144):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    buffer.seek(0)
    img_np = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    buffer.close()
    img_np = cv2.imdecode(img_np, 1)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    return img_np
    
def compare_img_pred_viz(img_np, viz_np, y_true, y_pred, probs_s=None, title=None):
    if isinstance(probs_s, (pd.core.series.Series)):
        p_df = probs_s.sort_values(ascending=False)[0:4].to_frame().reset_index()
        p_df.columns = ['class', 'prob']
        fig = plt.figure(figsize=(5, 2))
        ax = sns.barplot(x="prob", y="class", data=p_df)
        ax.set_xlim(0, 120)
        for p in ax.patches:
            ax.annotate(format(p.get_width(), '.2f')+'%', 
                           (p.get_x() + p.get_width() + 1.2, p.get_y()+0.6), 
                           size=13)
        ax.set(xticklabels=[], ylabel=None, xlabel=None)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 16)
        fig.tight_layout()
        barh_np = img_np_from_fig(fig)
        plt.close(fig)
    else:
        barh_np = [[]]
        
    fig = plt.figure(figsize=(15, 7.15))
    gridspec = plt.GridSpec(3, 5, wspace=0.5,\
                            hspace=0.4, figure=fig)
    orig_img_ax = plt.subplot(gridspec[:2, :2])
    orig_img_ax.imshow(img_np, interpolation='lanczos')
    orig_img_ax.grid(b=None)
    orig_img_ax.set_title("Actual Label: " + r"$\bf{" + str(y_true) + "}$")
    viz_img_ax = plt.subplot(gridspec[:3, 2:])
    viz_img_ax.imshow(viz_np, interpolation='spline16')
    viz_img_ax.grid(b=None)
    pred_ax = plt.subplot(gridspec[2, :2])
    pred_ax.set_title("Predicted Label: " + r"$\bf{" + str(y_pred) + "}$")
    pred_ax.imshow(barh_np, interpolation='spline36')
    pred_ax.grid(b=None)
    pred_ax.axes.get_xaxis().set_visible(False)
    pred_ax.axes.get_yaxis().set_visible(False)
    if title is not None:
        fig.suptitle(title, fontsize=18, weight='bold', x=0.65)
        plt.subplots_adjust(bottom=0, top=0.92)
    plt.show()
    
def heatmap_overlay(bg_img, overlay_img, cmap='jet'):
    img = np.uint8(bg_img[..., :3] * 255)
    if len(overlay_img.shape) == 2:
        overlay_img = cm.get_cmap(cmap)(overlay_img)
    heatmap = np.uint8(overlay_img[..., :3] * 255)
    return cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

def find_closest_datapoint_idx(point, points, metric_or_fn='euclidean', find_exact_first=0,\
                               distargs={}, scaler=None):
    if len(point.shape)!=1 or len(points.shape)!=2 or point.shape[0]!=points.shape[1]:
        raise Exception("point must be a 1d and points 2d where their number of features match")
    closest_idx = None
    if find_exact_first==1:
        sums_pts = np.sum(points, axis=1)
        sum_pt = np.sum(point, axis=0)
        s = (sums_pts==sum_pt)
        if isinstance(s, pd.core.series.Series):
            closest_idxs = s[s==True].index.to_list()
        else:
            closest_idxs = s.nonzero()[0]
        if len(closest_idxs) > 0:
            closest_idx = closest_idxs[-1]
    elif find_exact_first==2:
        if isinstance(points, pd.core.frame.DataFrame):
            for i in reversed(range(points.shape[0])):
                if np.allclose(point, points.iloc[i]):
                    closest_idx = points.iloc[i].name
                    break
        else:
            for i in reversed(range(points.shape[0])):
                if np.allclose(point, points[i]):
                    closest_idx = i
                    break
    if closest_idx is None:
        if scaler is not None:
            point_ = scaler.transform([point])
            points_ = scaler.transform(points)
        else:
            point_ = [point]
            points_ = points
        if isinstance(metric_or_fn, str):
            closest_idx = distance.cdist(point_, points, metric=metric_or_fn, **distargs).argmin()
        elif callable(metric_or_fn):
            dists = []
            if isinstance(points, pd.core.frame.DataFrame):
                for i in range(points.shape[0]):
                    dists.append(metric_or_fn(point_[0], points.iloc[i], **distargs))
            else:
                for i in range(points.shape[0]):
                    dists.append(metric_or_fn(point_[0], points[i], **distargs))
            closest_idx = np.array(dists).argmin()
        else:
            raise Exception("`metric_or_fn` must be a string of a distance metric or valid distance function")
        if isinstance(points, pd.core.frame.DataFrame):
            closest_idx = points.iloc[closest_idx].name
            
    return closest_idx

def approx_predict_ts(X, X_df, gen_X, ts_mdl, dist_metric='euclidean', lookback=672,\
                    filt_fn=None, X_scaler=None, y_scaler=None, progress_bar=False, no_info=np.array([[0]])):
    b_size = gen_X[0][0].shape[0]
    preds = None
    if progress_bar:
        rng = trange(X.shape[0], desc='Predicting')
    else:
        rng = range(X.shape[0])
    for i in rng: 
        x = X[i]
        if filt_fn is not None:
            X_filt_df, x = filt_fn(X_df, x, lookback)
        else:
            X_filt_df = X_df
        idx = find_closest_datapoint_idx(x, X_filt_df, dist_metric, find_exact_first=1, scaler=X_scaler)
        
        nidx = idx - lookback
        pred = ts_mdl.predict(gen_X[nidx//b_size][0])[nidx%b_size].reshape(1,-1)
        if i==0:
            preds = pred
        else:
            preds = np.vstack((preds,pred))
    if preds is not None:
        if y_scaler is not None:
            return y_scaler.inverse_transform(preds)
        else:
            return preds
    else:
        return no_info
    
def compare_confusion_matrices(y_true_1, y_pred_1, y_true_2, y_pred_2, group_1, group_2,\
                               plot=True, compare_fpr=False):
    """Compare two confusion matrices and display FPR ratio metrics. 
    Return FPR ratio between matrices.
    
    Keyword arguments:
    y_true_1 -- ground truth values for first confusion matrix (pandas series or 1D array)
    y_pred_1 -- predictions for first confusion matrix  (pandas series or 1D array)
    y_true_2 -- ground truth values for second confusion matrix (pandas series or 1D array)
    y_pred_2 -- predictions for second confusion matrix  (pandas series or 1D array)
    group_1 -- name of group represented by first matrix (string)
    group_2 -- name of group represented by second matrix (string)
    plot -- whether to plot the confusion matrices or not (boolean)
    """
    
    #Create confusion matrices for two different groups.
    conf_matrix_1 = metrics.confusion_matrix(y_true_1, y_pred_1)
    conf_matrix_2 = metrics.confusion_matrix(y_true_2, y_pred_2)

    #Plot both confusion matrices side-by-side.
    if plot:
        fig, ax = plt.subplots(1,2,figsize=(12,5))
        sns.heatmap(conf_matrix_1/np.sum(conf_matrix_1), annot=True,\
                    fmt='.2%', cmap='Blues', annot_kws={'size':16}, ax=ax[0])
        ax[0].set_title(group_1 + ' Confusion Matrix', fontsize=14)
        sns.heatmap(conf_matrix_2/np.sum(conf_matrix_2), annot=True,\
                    fmt='.2%', cmap='Blues', annot_kws={'size':16}, ax=ax[1])
        ax[1].set_title(group_2 + ' Confusion Matrix', fontsize=14)
        plt.show()

    #Calculate False Positive Rates (FPR) for each Group.
    tn, fp, _, _ = conf_matrix_1.ravel()
    fpr_1 = fp/(fp+tn)
    tn, fp, _, _ = conf_matrix_2.ravel()
    fpr_2 = fp/(fp+tn)

    #Print the FPRs and the ratio between them.
    if compare_fpr:
        if fpr_2 > fpr_1:
            print("\t" + group_2 + " FPR:\t%.1f%%" % (fpr_2*100))
            print("\t" + group_1 + " FPR:\t\t%.1f%%" % (fpr_1*100))
            print("\tRatio FPRs:\t\t%.2f x" % (fpr_2/fpr_1))
            return (fpr_2/fpr_1)
        else:
            print("\t" + group_1 + " FPR:\t%.1f%%" % (fpr_1*100))
            print("\t" + group_2 + " FPR:\t\t%.1f%%" % (fpr_2*100))
            print("\tRatio FPRs:\t\t%.2f x" % (fpr_1/fpr_2))
            return (fpr_1/fpr_2)