import subprocess
import sys
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
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
from aif360.metrics import ClassificationMetric

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

def evaluate_class_metrics_mdl(fitted_model, y_train_pred, y_test_prob, y_test_pred, y_train, y_test):      
    eval_dict = {}
    eval_dict['fitted'] = fitted_model
    eval_dict['preds_train'] = y_train_pred
    if y_test_prob is not None:
        eval_dict['probs_test'] = y_test_prob
    eval_dict['preds_test'] = y_test_pred
    eval_dict['accuracy_train'] = metrics.accuracy_score(y_train, y_train_pred)
    eval_dict['accuracy_test'] = metrics.accuracy_score(y_test, y_test_pred)
    eval_dict['precision_train'] = metrics.precision_score(y_train, y_train_pred, zero_division=0)
    eval_dict['precision_test'] = metrics.precision_score(y_test, y_test_pred, zero_division=0)
    eval_dict['recall_train'] = metrics.recall_score(y_train, y_train_pred, zero_division=0)
    eval_dict['recall_test'] = metrics.recall_score(y_test, y_test_pred, zero_division=0)
    eval_dict['f1_train'] = metrics.f1_score(y_train, y_train_pred, zero_division=0)
    eval_dict['f1_test'] = metrics.f1_score(y_test, y_test_pred, zero_division=0)
    eval_dict['mcc_train'] = metrics.matthews_corrcoef(y_train, y_train_pred)
    eval_dict['mcc_test'] = metrics.matthews_corrcoef(y_test, y_test_pred)
    if y_test_prob is not None:
        eval_dict['roc-auc_test'] = metrics.roc_auc_score(y_test, y_test_prob)
    return eval_dict
    
def evaluate_class_mdl(fitted_model, X_train, X_test, y_train, y_test, plot_roc=True, plot_conf_matrix=False,\
                       pct_matrix=True, predopts={}, show_summary=True, ret_eval_dict=False):
    y_train_pred = fitted_model.predict(X_train, **predopts).squeeze()
    if len(np.unique(y_train_pred)) > 2:
        y_train_pred = np.where(y_train_pred > 0.5, 1, 0)
        y_test_prob = fitted_model.predict(X_test, **predopts).squeeze()
        y_test_pred = np.where(y_test_prob > 0.5, 1, 0)
    else:   
        y_test_prob = fitted_model.predict_proba(X_test, **predopts)[:,1]
        y_test_pred = np.where(y_test_prob > 0.5, 1, 0)
        
    roc_auc = metrics.roc_auc_score(y_test, y_test_prob)
    if plot_roc:
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
    
    if plot_conf_matrix: 
        cf_matrix = metrics.confusion_matrix(y_test,\
                                             y_test_pred)
        plt.figure(figsize=(6, 5))
        if pct_matrix:
            sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,\
                        fmt='.2%', cmap='Blues', annot_kws={'size':16})
        else:
            sns.heatmap(cf_matrix, annot=True,\
                        fmt='d',cmap='Blues', annot_kws={'size':16})
        plt.show()
    
    if show_summary:
        print('Accuracy_train:  %.4f\t\tAccuracy_test:   %.4f' %\
                            (metrics.accuracy_score(y_train, y_train_pred),\
                             metrics.accuracy_score(y_test, y_test_pred)))
        print('Precision_test:  %.4f\t\tRecall_test:     %.4f' %\
                            (metrics.precision_score(y_test, y_test_pred, zero_division=0),\
                             metrics.recall_score(y_test, y_test_pred, zero_division=0)))
        print('ROC-AUC_test:    %.4f\t\tF1_test:         %.4f\t\tMCC_test: %.4f' %\
                            (roc_auc,\
                             metrics.f1_score(y_test, y_test_pred, zero_division=0),\
                             metrics.matthews_corrcoef(y_test, y_test_pred)))
    if ret_eval_dict:
        return evaluate_class_metrics_mdl(fitted_model, y_train_pred, y_test_prob, y_test_pred, y_train, y_test)
    else:
        return y_train_pred, y_test_prob, y_test_pred

def evaluate_multiclass_metrics_mdl(fitted_model, y_test_prob, y_test_pred, y_test, ohe=None):      
    eval_dict = {}
    eval_dict['fitted'] = fitted_model
    if y_test_prob is not None:
        eval_dict['probs'] = y_test_prob
    eval_dict['preds'] = y_test_pred
    eval_dict['accuracy'] = metrics.accuracy_score(y_test, y_test_pred)
    eval_dict['precision'] = metrics.precision_score(y_test, y_test_pred, zero_division=0, average='micro')
    eval_dict['recall'] = metrics.recall_score(y_test, y_test_pred, zero_division=0, average='micro')
    eval_dict['f1'] = metrics.f1_score(y_test, y_test_pred, zero_division=0, average='micro')
    eval_dict['mcc'] = metrics.matthews_corrcoef(y_test, y_test_pred)
    if y_test_prob is not None:
        if ohe is not None:
            eval_dict['roc-auc'] = metrics.roc_auc_score(ohe.transform(y_test), y_test_prob)
        else:
            eval_dict['roc-auc'] = metrics.roc_auc_score(y_test, y_test_prob)
    return eval_dict
    
def evaluate_multiclass_mdl(fitted_model, X, y, class_l, ohe=None, plot_roc=False, plot_roc_class=True,\
                            plot_conf_matrix=True, pct_matrix=True, plot_class_report=True, ret_eval_dict=False,\
                            predopts={}):
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
    if len(y_prob.shape)==1:
        y_prob = np.expand_dims(y_prob, axis=1)
    if y_prob.shape[1] == 1:
        y_prob = fitted_model.predict_proba(X, **predopts)
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
                        fmt='.1%', cmap='Blues', annot_kws={'size':12})
        else:
            sns.heatmap(conf_matrix, annot=True, xticklabels=class_l, yticklabels=class_l,\
                        cmap='Blues', annot_kws={'size':12})
        plt.show()

    if plot_class_report:
        print(metrics.classification_report(y, y_pred, digits=3, zero_division=0))
    
    if ret_eval_dict:
        return evaluate_multiclass_metrics_mdl(fitted_model, y_prob, y_pred, y, ohe)
    else:
        return y_pred, y_prob

def evaluate_reg_metrics_mdl(fitted_model, y_train_pred, y_test_pred, y_train, y_test):      
    eval_dict = {}
    eval_dict['fitted'] = fitted_model
    eval_dict['preds_train'] = y_train_pred
    eval_dict['preds_test'] = y_test_pred
    eval_dict['trues_train'] = y_train
    eval_dict['trues_test'] = y_test
    eval_dict['rmse_train'] = metrics.mean_squared_error(y_train, y_train_pred, squared=False)
    eval_dict['rmse_test'] = metrics.mean_squared_error(y_test, y_test_pred, squared=False)
    eval_dict['r2_train'] = metrics.r2_score(y_train, y_train_pred)
    eval_dict['r2_test'] = metrics.r2_score(y_test, y_test_pred)
    return eval_dict

def evaluate_reg_mdl(fitted_model, X_train, X_test, y_train, y_test, scaler=None,\
                     plot_regplot=True, y_truncate=False, show_summary=True,\
                     ret_eval_dict=False, predopts={}):
    y_train_ = y_train.copy()
    y_test_ = y_test.copy()
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

    if plot_regplot:
        plt.figure(figsize = (12,12))
        plt.ylabel('Predicted', fontsize = 14)
        plt.scatter(y_test_, y_test_pred)
        sns.regplot(x=y_test_, y=y_test_pred, color="g")
        plt.xlabel('Observed', fontsize = 14)
        plt.show()     
    
    if show_summary:
        print('RMSE_train: %.4f\tRMSE_test: %.4f\tr2: %.4f' %\
                (metrics.mean_squared_error(y_train_, y_train_pred, squared=False),\
                 metrics.mean_squared_error(y_test_, y_test_pred, squared=False),\
                 metrics.r2_score(y_test_, y_test_pred)))
        
    if ret_eval_dict:
        return evaluate_reg_metrics_mdl(fitted_model, y_train_pred, y_test_pred, y_train_, y_test_)
    elif y_truncate:
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

def approx_predict_ts(X, X_df, gen_X, ts_mdl, dist_metric='euclidean', lookback=0,\
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
        
def discretize(v, v_intervals, use_quartiles=False, use_continuous_bins=False):
    if isinstance(v, (pd.core.series.Series, np.ndarray)) and isinstance(v_intervals, (list, np.ndarray)) and len(np.unique(v)) != len(v_intervals):
        raise Exception("length of interval must match unique items in array")
        
    if isinstance(v, (str)) and isinstance(v_intervals, (list, np.ndarray)):
        #name of variable instead of array and list of intervals used
        if isinstance(v_intervals, list): v_intervals = np.array(v_intervals)
        return v, v_intervals
    
    if (np.isin(v.dtype, [int, float, 'int8', 'int16', 'int32', 'float16', 'float32'])) and (isinstance(v_intervals, (int))) and (len(np.unique(v)) >= v_intervals) and (max(v) > min(v)):
        #v is discretizable, otherwise assumed to be already discretized
        if use_continuous_bins:
            if use_quartiles:
                v, bins = pd.qcut(v, v_intervals, duplicates='drop', retbins=True, labels=True, precision=2)
            else:
                v, bins = pd.cut(v, v_intervals, duplicates='drop', retbins=True, labels=True, precision=2)
        else:
            if use_quartiles:
                v = pd.qcut(v, v_intervals, duplicates='drop', precision=2)
            else:
                v = pd.cut(v, v_intervals, duplicates='drop', precision=2)
        
    if np.isin(v.dtype, [object, 'category']):
        if not isinstance(v, (pd.core.series.Series)):
            v = pd.Series(v)
        bins = np.sort(np.unique(v)).astype(str)
        v = v.astype(str)
        bin_dict = {bins[i]:i for i in range(len(bins))} 
        v = v.replace(bin_dict)
    else:
        bins = np.unique(v)
        
    if isinstance(v_intervals, (list, np.ndarray)) and len(bins) == len(v_intervals):
        bins = v_intervals
                       
    return v, bins

def plot_prob_progression(x, y, x_intervals=7, use_quartiles=False,\
                          xlabel=None, ylabel=None, title=None, model=None, X_df=None, x_col=None,\
                         mean_line=False, figsize=(12,6), x_margin=0.01):
    if isinstance(x, list): x = np.array(x)
    if isinstance(y, list): y = np.array(y)
    if (not isinstance(x, (str, pd.core.series.Series, np.ndarray))) or (not isinstance(y, (str, pd.core.series.Series, np.ndarray))):
        raise Exception("x and y must be either lists, pandas series or numpy arrays. x can be string when dataset is provided seperately")
    if (isinstance(x, (pd.core.series.Series, np.ndarray)) and (len(x.shape) != 1)) or ((isinstance(y, (pd.core.series.Series, np.ndarray))) and (len(y.shape) != 1)):
        raise Exception("x and y must have a single dimension")
    if (isinstance(x_intervals, (int)) and (x_intervals < 2)) or (isinstance(x_intervals, (list, np.ndarray)) and (len(x_intervals) < 2)):
        raise Exception("there must be at least two intervals to plot")
    if not np.isin(y.dtype, [int, float, 'int8', 'int16', 'int32', 'float16', 'float32']):
        raise Exception("y dimension must be a list, pandas series or numpy array of integers or floats")
    if max(y) == min(y):
        raise Exception("y dimension must have at least two values")
    elif len(np.unique(y)) == 2 and ((max(y) != 1) or (min(y) != 0)):
        raise Exception("y dimension if has two values must have a max of exactly 1 and min of exactly zero")
    elif len(np.unique(y)) > 2 and ((max(y) <= 1) or (min(y) >= 0)):
        raise Exception("y dimension if has more than two values must have range between between 0-1")
    x_use_continuous_bins = (model is not None) and (isinstance(x_intervals, (list, np.ndarray)))
    x, x_bins = discretize(x, x_intervals, use_quartiles, x_use_continuous_bins)
    x_range = [*range(len(x_bins))]
    plot_df = pd.DataFrame({'x':x_range})
    if (model is not None) and (X_df is not None) and (x_col is not None):
        preds = model.predict(X_df).squeeze()
        if len(np.unique(preds)) <= 2:
            preds = model.predict_proba(X_df)[:,1]
        x_, _ = discretize(X_df[x_col], x_intervals, use_quartiles, x_use_continuous_bins)
        xy_df = pd.DataFrame({'x':x_, 'y':preds})
    else:
        xy_df = pd.DataFrame({'x':x,'y':y})
    probs_df = xy_df.groupby(['x']).mean().reset_index()
    probs_df = pd.merge(plot_df, probs_df, how='left', on='x').fillna(0)
    
    sns.set()
    x_bin_cnt = len(x_bins)
    l_width = 0.933
    r_width = 0.05
    w, h = figsize
    wp = (w-l_width-r_width)/9.27356902357
    xh_margin = ((wp-(x_margin*2))/(x_bin_cnt*2))+x_margin
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=figsize,\
                                   gridspec_kw={'height_ratios': [3, 1]})
    if title is not None:
        fig.suptitle(title, fontsize=21)
        plt.subplots_adjust(top = 0.92, bottom=0.01, hspace=0.001, wspace=0.001)
    else:
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.001, wspace=0.001)
    ax0.minorticks_on()
    sns.lineplot(data=probs_df, x='x', y='y', ax=ax0)
    ax0.set_ylabel('Probability', fontsize=15)
    ax0.set_xlabel('')
    ax0.grid(b=True, axis='x', which='minor', color='w', linestyle=':')
    #ax0.set_xticks([], [])
    ax0.margins(x=xh_margin)
    if mean_line:
        ax0.axhline(y=xy_df.y.mean(), c='red', linestyle='dashed', label="mean")
        ax0.legend()
    sns.histplot(xy_df, x="x", stat='probability', bins=np.arange(x_bin_cnt+1)-0.5, ax=ax1)
    ax1.set_ylabel('Observations', fontsize=15)
    ax1.set_xlabel(xlabel, fontsize=15)
    ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
    ax1.set_xticklabels(['']+list(x_bins))
    ax1.margins(x=x_margin)
    plt.show()
    
def plot_prob_contour_map(x, y, z, x_intervals=7, y_intervals=7, use_quartiles=False, plot_type='contour',\
                          xlabel=None, ylabel=None, title=None, model=None, X_df=None, x_col=None, y_col=None,\
                          diff_to_mean=False, annotate=False):
    if isinstance(x, list): x = np.array(x)
    if isinstance(y, list): y = np.array(y)
    if isinstance(z, list): z = np.array(z)
    if (not isinstance(x, (str, pd.core.series.Series, np.ndarray))) or (not isinstance(y, (str, pd.core.series.Series, np.ndarray))) or (not isinstance(z, (pd.core.series.Series, np.ndarray))):
        raise Exception("x, y and z must be either lists, pandas series or numpy arrays. x and y can be strings when dataset is provided seperately")
    if (isinstance(x, (pd.core.series.Series, np.ndarray)) and (len(x.shape) != 1)) or ((isinstance(y, (pd.core.series.Series, np.ndarray))) and (len(y.shape) != 1)) or (len(z.shape) != 1):
        raise Exception("x, y and z must have a single dimension")
    if (isinstance(x_intervals, (int)) and (x_intervals < 2)) or (isinstance(x_intervals, (list, np.ndarray)) and (len(x_intervals) < 2)) or (isinstance(y_intervals, (int)) and (y_intervals < 2)) or (isinstance(y_intervals, (list, np.ndarray)) and (len(y_intervals) < 2)):
        raise Exception("there must be at least two intervals to contour")
    if not np.isin(z.dtype, [int, float, 'int8', 'int16', 'int32', 'float16', 'float32']):
        raise Exception("z dimension must be a list, pandas series or numpy array of integers or floats")
    if max(z) == min(z):
        raise Exception("z dimension must have at least two values")
    elif len(np.unique(z)) == 2 and ((max(z) != 1) or (min(z) != 0)):
        raise Exception("z dimension if has two values must have a max of exactly 1 and min of exactly zero")
    elif len(np.unique(z)) > 2 and ((max(z) <= 1) or (min(z) >= 0)):
        raise Exception("z dimension if has more than two values must have range between between 0-1")
    x_use_continuous_bins = (model is not None) and (isinstance(x_intervals, (list, np.ndarray)))
    y_use_continuous_bins = (model is not None) and (isinstance(y_intervals, (list, np.ndarray)))
    x, x_bins = discretize(x, x_intervals, use_quartiles, x_use_continuous_bins)
    y, y_bins = discretize(y, y_intervals, use_quartiles, y_use_continuous_bins)
    x_range = [*range(len(x_bins))]
    #if isinstance(y_intervals, (int)):
    y_range = [*range(len(y_bins))]
    #else:
    #y_range = y_intervals
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    plot_df = pd.DataFrame(positions.T, columns=['x', 'y'])
    
    if (model is not None) and (X_df is not None) and (x_col is not None) and (y_col is not None):
        preds = model.predict(X_df).squeeze()
        if len(np.unique(preds)) <= 2:
            preds = model.predict_proba(X_df)[:,1]
        x_, _ = discretize(X_df[x_col], x_intervals, use_quartiles, x_use_continuous_bins)
        y_, _ = discretize(X_df[y_col], y_intervals, use_quartiles, y_use_continuous_bins)
        xyz_df = pd.DataFrame({'x':x_, 'y':y_, 'z':preds})
    else:
        xyz_df = pd.DataFrame({'x':x,'y':y,'z':z})
    probs_df = xyz_df.groupby(['x','y']).mean().reset_index()        
    probs_df = pd.merge(plot_df, probs_df, how='left', on=['x','y']).fillna(0)
    if diff_to_mean:
        expected_value = xyz_df.z.mean()
        probs_df['z'] = probs_df['z'] - expected_value
        cmap = plt.cm.RdYlBu
    else:
        cmap = plt.cm.viridis
    grid_probs = np.reshape(probs_df.z.to_numpy(), x_grid.shape)

    x_bin_cnt = len(x_bins)
    y_bin_cnt = len(y_bins)
    fig, (ax_top, ax_bottom) = plt.subplots(2, 2, figsize=(12,9),\
                                   gridspec_kw={'height_ratios': [1, 7], 'width_ratios': [6, 1]})
    if title is not None:
        fig.suptitle(title, fontsize=21)
        plt.subplots_adjust(top = 0.95, bottom=0.01, hspace=0.001, wspace=0.001)
    else:
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.001, wspace=0.001)

    sns.set_style(None)
    sns.set_style({'axes.facecolor':'white', 'grid.color': 'white'})
    sns.histplot(xyz_df, x='x', stat='probability', bins=np.arange(x_bin_cnt+1)-0.5, color=('dimgray',), ax=ax_top[0])
    ax_top[0].set_xticks([])
    ax_top[0].set_yticks([])
    ax_top[0].set_xlabel('')
    ax_top[0].set_ylabel('')
    ax_top[1].set_visible(False)

    if plot_type == 'contour':
        ax_bottom[0].contour(
            x_grid,
            y_grid,
            grid_probs,
            colors=('w',)
        )
        mappable = ax_bottom[0].contourf(
            x_grid,
            y_grid,
            grid_probs,
            cmap=cmap
        ) 
    else:
        mappable = ax_bottom[0].imshow(grid_probs, cmap=plt.cm.viridis,\
                                      interpolation='nearest', aspect='auto')
        if annotate:
            for i in range(y_bin_cnt):
                for j in range(x_bin_cnt):
                    text = ax_bottom[0].text(j, i, "{:.1%}".format(grid_probs[i, j]), fontsize=16,
                                             ha="center", va="center", color="w")
            ax_bottom[0].grid(False)
            
    ax_bottom[0].xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
    ax_bottom[0].set_xticklabels([''] + list(x_bins))
    ax_bottom[0].yaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
    ax_bottom[0].set_yticklabels([''] + list(y_bins))
    #ax_bottom[0].margins(x=0.04, y=0.04)

    if xlabel is not None:
        ax_bottom[0].set_xlabel(xlabel, fontsize=15)
        
    if ylabel is not None:
        ax_bottom[0].set_ylabel(ylabel, fontsize=15)

    cbar = plt.colorbar(mappable, ax=ax_bottom[1])
    cbar.ax.set_ylabel('Probability', fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    sns.histplot(xyz_df, y="y", stat='probability', bins=np.arange(y_bin_cnt+1)-0.5, color=('dimgray',), ax=ax_bottom[1])
    ax_bottom[1].set_xticks([])
    ax_bottom[1].set_yticks([])
    ax_bottom[1].set_xlabel('')
    ax_bottom[1].set_ylabel('')
    sns.set_style(None)

    plt.show()
    
def compare_image_predictions(X_mod, X_orig, y_mod, y_orig, y_mod_prob=None, y_orig_prob=None,\
                              num_samples=3, title_mod_prefix="Modified: ", title_orig_prefix="Original: ",\
                              calc_difference=True, title_difference_prefix="Average difference: ",\
                              max_width=14, use_misclass=True):
    if calc_difference:
        X_difference = np.mean(np.abs((X_mod - X_orig)))
        diff_title = (title_difference_prefix + '{:4.3f}').format(X_difference)
    if num_samples > 0:
        if use_misclass:
            misclass_idx = np.unique(np.where(y_orig != y_mod)[0])
        else:
            misclass_idx = np.unique(np.where(y_orig == y_mod)[0])
        if misclass_idx.shape[0] > 0:
            if misclass_idx.shape[0] < num_samples:
                num_samples = misclass_idx.shape[0]
                samples_idx = misclass_idx
            else:
                np.random.shuffle(misclass_idx)
                samples_idx = misclass_idx[0:num_samples]
            if num_samples > 2:
                width = max_width
                lg = math.log(num_samples)
            elif num_samples == 2:
                width = round(max_width*0.6)
                lg = 0.6
            else:
                width = round(max_width*0.3)
                lg = 0.3
            img_ratio = X_mod.shape[1]/X_mod.shape[2]
            height = round((((width - ((num_samples - 1)*(0.75 / lg)))/num_samples)*img_ratio))*2
            plt.subplots(figsize=(width,height))
            for i, s in enumerate(samples_idx, start=1):
                plt.subplot(2, num_samples, i)
                plt.imshow(X_mod[s])
                plt.grid(b=None)
                if num_samples > 3:
                    plt.axis('off')
                if y_mod_prob is None:
                    plt.title("%s%s" % (title_mod_prefix, y_mod[s]))
                else:
                    plt.title("%s%s (%.1f%%)" % (title_mod_prefix, y_mod[s], y_mod_prob[s]*100))
            for i, s in enumerate(samples_idx, start=1):
                plt.subplot(2, num_samples, i+num_samples)
                plt.imshow(X_orig[s])
                plt.grid(b=None)
                if num_samples > 3:
                    plt.axis('off')
                if y_orig_prob is None:
                    plt.title("%s%s" % (title_orig_prefix, y_orig[s]))
                else:
                    plt.title("%s%s (%.1f%%)" % (title_orig_prefix, y_orig[s], y_orig_prob[s]*100))
            if calc_difference:
                plt.subplots_adjust(bottom=0, top=0.88)
                fs = 21 - num_samples
                plt.suptitle(diff_title, fontsize=fs)
            plt.show()
        else:
            if calc_difference:
                print(diff_title)
            print("No Different Classifications")

def profits_by_thresh(y_profits, y_pred, threshs, var_costs=1, min_profit=None, fixed_costs=0):
    profits_dict = {}
    for thresh in threshs:
        profits_dict[thresh] = {}
        profits_dict[thresh]["revenue"] = sum(y_profits[y_pred > thresh])
        profits_dict[thresh]["costs"] = (sum(y_pred > thresh)*var_costs)+var_costs
        profits_dict[thresh]["profit"] = profits_dict[thresh]["revenue"] -\
                                         profits_dict[thresh]["costs"]
        if profits_dict[thresh]["costs"] > 0:
            profits_dict[thresh]["roi"] = profits_dict[thresh]["profit"]/\
                                          profits_dict[thresh]["costs"]
        else:
            profits_dict[thresh]["roi"] = 0
            
    profits_df = pd.DataFrame.from_dict(profits_dict, 'index')
    if min_profit is not None:
        profits_df = profits_df[profits_df.profit >= min_profit]
    return profits_df
            
def compare_df_plots(df1, df2, title1=None, title2=None, y_label=None, x_label=None,\
                     y_formatter=None, x_formatter=None, plot_args={}):
    if y_formatter is None:
        y_formatter = plt.FuncFormatter(lambda x, loc: "${:,}K".format(x/1000))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5), sharey=True)
    df1.plot(ax=ax1, fontsize=13, **plot_args)
    if title1 is not None:
        ax1.set_title(title1, fontsize=20)
    if y_label is not None:
        ax1.set_ylabel(y_label, fontsize=14)
    if x_label is not None:
        ax1.set_xlabel(x_label, fontsize=14)
    if 'secondary_y' in plot_args:
        ax1.get_legend().set_bbox_to_anchor((0.7, 0.99))
    if y_formatter is not None:
        ax1.yaxis.set_major_formatter(y_formatter)
    if x_formatter is not None:
        ax1.xaxis.set_major_formatter(x_formatter)
    ax1.grid(b=True)
    ax1.right_ax.grid(False)
    df2.plot(ax=ax2, fontsize=13, **plot_args)
    if title2 is not None:
        ax2.set_title(title2, fontsize=20)
    if x_label is not None:
        ax2.set_xlabel(x_label, fontsize=14)
    if 'secondary_y' in plot_args:
        ax2.get_legend().set_bbox_to_anchor((0.7, 0.99))
    if x_formatter is not None:
        ax2.xaxis.set_major_formatter(x_formatter)
    ax2.grid(b=True)
    ax2.right_ax.grid(False)
    fig.tight_layout()
    plt.show()
    
def compute_aif_metrics(dataset_true, dataset_pred, unprivileged_groups, privileged_groups,\
                        ret_eval_dict=True):

    metrics_cls = ClassificationMetric(dataset_true, dataset_pred, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    metrics_dict = {}
    metrics_dict["BA"] = 0.5*(metrics_cls.true_positive_rate()+
                                             metrics_cls.true_negative_rate())
    metrics_dict["SPD"] = metrics_cls.statistical_parity_difference()
    metrics_dict["DI"] = metrics_cls.disparate_impact()
    metrics_dict["AOD"] = metrics_cls.average_odds_difference()
    metrics_dict["EOD"] = metrics_cls.equal_opportunity_difference()
    metrics_dict["DFBA"] = metrics_cls.differential_fairness_bias_amplification()
    metrics_dict["TI"] = metrics_cls.theil_index()
    
    if ret_eval_dict:
        return metrics_dict, metrics_cls
    else:
        return metrics_cls