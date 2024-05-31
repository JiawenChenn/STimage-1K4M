#py3.10
# this code is adopted from PLIP GitHub repository: https://github.com/PathologyFoundation/plip/tree/main/reproducibility
import os
import pandas as pd
import numpy as np
from sklearn.metrics import auc, roc_curve, f1_score, recall_score, precision_score, matthews_corrcoef, accuracy_score, classification_report,silhouette_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder

import glob

def eval_metrics(y_true, y_pred, y_pred_proba = None, average_method='weighted'):
    assert len(y_true) == len(y_pred)
    if y_pred_proba is None:
        auroc = np.nan
    elif len(np.unique(y_true)) > 2:
        print('Multiclass AUC is not currently available.')
        auroc = np.nan
    else:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auroc = auc(fpr, tpr)
    f1 = f1_score(y_true, y_pred, average = average_method)
    #print(classification_report(y_true, y_pred))
    #precision = precision_score(y_true, y_pred, average = average_method)
    #recall = recall_score(y_true, y_pred, average = average_method)
    #mcc = matthews_corrcoef(y_true, y_pred)
    #acc = accuracy_score(y_true, y_pred)
    tp,fp,tn,fn = 0,0,0,0
    for i in range(len(y_pred)):
        if y_true[i]==y_pred[i]==1:
           tp += 1
        if y_pred[i]==1 and y_true[i]!=y_pred[i]:
           fp += 1
        if y_true[i]==y_pred[i]==0:
           tn += 1
        if y_pred[i]==0 and y_true[i]!=y_pred[i]:
           fn += 1
    if (tp+fn) == 0: sensitivity = np.nan
    else: sensitivity = tp/(tp+fn) # recall
    if (tn+fp) == 0: specificity = np.nan
    else: specificity = tn/(tn+fp)
    if (tp+fp) == 0: ppv = np.nan
    else: ppv = tp/(tp+fp) # precision or positive predictive value (PPV)
    if (tn+fn) == 0: npv = np.nan
    else: npv = tn/(tn+fn) # negative predictive value (NPV)
    if (tp+tn+fp+fn) == 0: hitrate = np.nan
    else: hitrate = (tp+tn)/(tp+tn+fp+fn) # accuracy (ACC)
    performance = {#'Accuracy': acc,
                   'AUC': auroc,
                   'WF1': f1,
                   #'precision': precision,
                   #'recall': recall,
                   #'mcc': mcc,
                   'tp': tp,
                   'fp': fp,
                   'tn': tn,
                   'fn': fn,
                   'sensitivity': sensitivity,
                   'specificity': specificity,
                   'ppv': ppv,
                   'npv': npv,
                   'hitrate': hitrate,
                   'instances' : len(y_true)}
    return performance



def run_classification(train_x, train_y, test_x, test_y, val_x, val_y, seed=1, alpha=0.1,penalty="l2"):
    classifier = SGDClassifier(random_state=seed, loss="log_loss",
                                alpha=alpha, verbose=0,
                                penalty="l2", max_iter=10000, class_weight="balanced")
    train_y = train_y.to_numpy()
    test_y = test_y.to_numpy()
    val_y = val_y.to_numpy()
    classifier.fit(train_x, train_y)
    test_pred = classifier.predict(test_x)
    train_pred = classifier.predict(train_x)
    val_pred = classifier.predict(val_x)
    train_matrics = eval_metrics(train_y, train_pred, average_method="macro")
    test_metrics = eval_metrics(test_y, test_pred, average_method="macro")
    val_metrics = eval_metrics(val_y, val_pred, average_method="macro")
    return {'train_f1': train_matrics['WF1'], 'test_f1': test_metrics['WF1'], 'val_f1': val_metrics['WF1'], 'alpha': alpha}


# path to the embedding
model_path = glob.glob('./embedding/withanno/*')

# path to the annotation
anno_path = glob.glob('./anno_forlp/*')
anno_path = sorted(anno_path)
anno_name = [i.replace('./anno_forlp/','') for i in anno_path]
anno_name = [i.replace('_anno.csv','') for i in anno_name]

data_table = pd.DataFrame({'anno_name':anno_name,'anno_path':anno_path})

#for dataset in np.unique(data_table['data_name']):

model_record_best = []
for j in range(len(model_path)):
    embedding_all = pd.read_csv(model_path[j],index_col=0)
    model_name = model_path[j].replace('./embedding/withanno/','')
    model_name = model_name.replace('_withanno.csv','')
    print(model_name)
    for i in range(data_table.shape[0]):
        print(data_table['anno_path'][i])
        anno_temp = pd.read_csv(data_table['anno_path'][i],sep='\t',index_col=0)
        anno_temp = anno_temp[anno_temp.V2 != '']
        anno_temp = anno_temp[anno_temp.V2.notnull()]
        anno_temp = anno_temp[anno_temp.V2 != 'undetermined']
        anno_temp = anno_temp[anno_temp.V2 != 'Exclude']
        #print(anno_temp.V2.value_counts(dropna=False))
        #
        index_keep = anno_temp.index.intersection(embedding_all.index)
        embedding = embedding_all.loc[index_keep]
        anno_temp = anno_temp.loc[index_keep]
        le = LabelEncoder()
        anno_temp['mapped_y'] = le.fit_transform(anno_temp.V2)
        #
        #samples = np.unique(anno_temp.sample_name)
        all_records_dataset = []
        for k in range(5):
            # random split 80% train 10% test 10% validation
            np.random.seed(k)
            train_index = np.random.choice(anno_temp.index, int(0.8*anno_temp.shape[0]), replace=False)
            test_index = np.random.choice(list(set(anno_temp.index)-set(train_index)), int(0.1*anno_temp.shape[0]), replace=False)
            val_index = list(set(anno_temp.index)-set(train_index)-set(test_index))
            train_y = anno_temp.loc[train_index].mapped_y
            train_x = embedding.loc[train_index].to_numpy()
            test_y = anno_temp.loc[test_index].mapped_y
            test_x = embedding.loc[test_index].to_numpy()
            val_y = anno_temp.loc[val_index].mapped_y
            val_x = embedding.loc[val_index].to_numpy()
            all_records = []
            for alpha in [1.0, 0.1, 0.01, 0.001,0.0001]:
                metrics = run_classification(train_x, train_y, test_x, test_y, val_x, val_y, alpha = alpha,penalty='l2')
                metrics["alpha"] = alpha
                metrics["test_on"] = 'split'+str(k)
                metrics["model_name"] = model_name
                all_records.append(metrics)
            all_records_dataset.extend(all_records)
            #
        all_records_dataset_df = pd.DataFrame(all_records_dataset)
        #print(all_records_dataset_df)
        best_alpha = all_records_dataset_df.groupby('alpha')['val_f1'].mean().idxmax()
        mean_wf1 = all_records_dataset_df[all_records_dataset_df['alpha'] == best_alpha]['test_f1'].mean()
        std_wf1 = all_records_dataset_df[all_records_dataset_df['alpha'] == best_alpha]['test_f1'].std()
        record_best = {'model_name':model_name,'best_alpha':best_alpha,'mean_wf1':mean_wf1,'std_wf1':std_wf1,'data_name':data_table['anno_name'][i]}
        #print(record_best)
        model_record_best.append(record_best)



model_record_best_df = pd.DataFrame(model_record_best)
model_record_best_df.to_csv('linear_probing_result.csv',index=False,sep='\t')


model_name_zero = ['CLIP','PLIP','uni']
model_record_best = []
for model_name in model_name_zero:
    embedding1 = pd.read_csv(f'./zero_shot_embedding/{model_name}_human_image_feature.csv',index_col=0)
    embedding2 = pd.read_csv(f'./zero_shot_embedding/{model_name}_mouse_image_feature.csv',index_col=0)
    embedding_all = pd.concat([embedding1,embedding2])
    print(model_name)
    for i in range(data_table.shape[0]):
        print(data_table['anno_path'][i])
        anno_temp = pd.read_csv(data_table['anno_path'][i],sep='\t',index_col=0)
        anno_temp = anno_temp[anno_temp.V2 != '']
        anno_temp = anno_temp[anno_temp.V2.notnull()]
        anno_temp = anno_temp[anno_temp.V2 != 'undetermined']
        anno_temp = anno_temp[anno_temp.V2 != 'Exclude']
        #print(anno_temp.V2.value_counts(dropna=False))
        #
        index_keep = anno_temp.index.intersection(embedding_all.index)
        embedding = embedding_all.loc[index_keep]
        anno_temp = anno_temp.loc[index_keep]
        le = LabelEncoder()
        anno_temp['mapped_y'] = le.fit_transform(anno_temp.V2)
        #
        #samples = np.unique(anno_temp.sample_name)
        all_records_dataset = []
        for k in range(5):
            # random split 80% train 10% test 10% validation
            np.random.seed(k)
            train_index = np.random.choice(anno_temp.index, int(0.8*anno_temp.shape[0]), replace=False)
            test_index = np.random.choice(list(set(anno_temp.index)-set(train_index)), int(0.1*anno_temp.shape[0]), replace=False)
            val_index = list(set(anno_temp.index)-set(train_index)-set(test_index))
            train_y = anno_temp.loc[train_index].mapped_y
            train_x = embedding.loc[train_index].to_numpy()
            test_y = anno_temp.loc[test_index].mapped_y
            test_x = embedding.loc[test_index].to_numpy()
            val_y = anno_temp.loc[val_index].mapped_y
            val_x = embedding.loc[val_index].to_numpy()
            all_records = []
            for alpha in [1.0, 0.1, 0.01, 0.001,0.0001]:
                metrics = run_classification(train_x, train_y, test_x, test_y, val_x, val_y, alpha = alpha)
                metrics["alpha"] = alpha
                metrics["test_on"] = 'split'+str(k)
                metrics["model_name"] = model_name
                all_records.append(metrics)
            all_records_dataset.extend(all_records)
            #
        all_records_dataset_df = pd.DataFrame(all_records_dataset)
        best_alpha = all_records_dataset_df.groupby('alpha')['val_f1'].mean().idxmax()
        mean_wf1 = all_records_dataset_df[all_records_dataset_df['alpha'] == best_alpha]['test_f1'].mean()
        std_wf1 = all_records_dataset_df[all_records_dataset_df['alpha'] == best_alpha]['test_f1'].std()
        record_best = {'model_name':model_name,'best_alpha':best_alpha,'mean_wf1':mean_wf1,'std_wf1':std_wf1,'data_name':data_table['anno_name'][i]}
        model_record_best.append(record_best)

model_record_best_df = pd.DataFrame(model_record_best)
model_record_best_df.to_csv('linear_probing_zero_result.csv',index=False,sep='\t')
