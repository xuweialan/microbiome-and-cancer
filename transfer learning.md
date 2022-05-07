```python
from __future__ import division
import threading
import numpy as np
import pandas as pd
from joblib import Parallel
#from sklearn.utils.fixes import delayed
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score,precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle
from sys import argv

import sys  #添加路径变量
sys.path.append('C:/Users/Alan‘s Lenovo/Desktop/TransferRandomForest/Doc')
import TransferRandomForest as trf

def label_convert(labels, dic = False):
    Nclasses = np.unique(labels).tolist()
    labeldic = {}
    labels = labels.tolist()
    for i in Nclasses:
        labeldic[i] = Nclasses.index(i)
    for i in range(len(labels)):
        labels[i] = labeldic[labels[i]]
    if dic == True:
        return np.array(labels), labeldic
    else:
        return np.array(labels)  
    
def convert_to_dic(narray_key, narray_value):
    list_key = narray_key.tolist()
    list_value = narray_value.tolist()

    convert_dic = {}
    for i in range(len(list_key)):
        convert_dic[list_key[i]] = list_value[i]

    return convert_dic

def Balanced_acc(y_pred, ytest, Nclass):
    acc_c = 0
    acc_class = {}
    for c in np.unique(ytest):
        i = ytest == c
        correct = y_pred[i] == ytest[i]
        acc_c += sum(correct) / len(correct)
        acc_class[c] = format(sum(correct) / len(correct), '.5f')

    Bacc = format(acc_c / len(np.unique(ytest)), '.5f')
    return Bacc, acc_class

def roc_auc_calculate(y_query, y_proba):
    classes = np.unique(y_query)
    Nclasses = len(classes)
    y_test = np.zeros((len(y_query), Nclasses))
    for i in range(len(y_query)):
        y_test[i][y_query[i]] = 1

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(Nclasses):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_proba[:, i])
        roc_auc[i] = float(format(auc(fpr[i], tpr[i]), '.5f'))
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(Nclasses)]))#数组拼接得到fpr的矩阵

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)  #构造数字都为0的矩阵，为做平均做准备
    for i in range(Nclasses):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= Nclasses

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = float(format(auc(fpr["macro"], tpr["macro"]), '.5f'))
    

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_proba.ravel())
    roc_auc["micro"] = float(format(auc(fpr["micro"], tpr["micro"]), '.5f'))
    
    roc_auc["ovr"] = float(format(roc_auc_score(y_query, y_proba, multi_class='ovr'), '.5f'))
    roc_auc["ovo"] = float(format(roc_auc_score(y_query, y_proba, multi_class='ovo'), '.5f'))
    return roc_auc, fpr, tpr

def roc_curve_plot(roc_auc, fpr, tpr, filename):
    plt.figure()
    lw = 2
    
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "deeppink", "navy"])
    for i, color in zip(range(Nclasses), colors):
        plt.plot( fpr[i], tpr[i], color=color, lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),)

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(filename + " ROC curve")
    plt.legend(bbox_to_anchor=(2.4, 0.5),loc=5,ncol=2)
    filepath = "./" + filename + ".pdf"
    plt.savefig(filepath,bbox_inches = 'tight')
    plt.show()
```


```python
#组织数据
metadata = pd.read_csv("C://Users//Alan‘s Lenovo//Desktop//tissue_meta.csv")
snmdata = pd.read_csv("C://Users//Alan‘s Lenovo//Desktop//tissue_snm.csv")
mldataX = snmdata.values[:,1:]
mldataY = metadata.values[:,8]
#血液数据
metadata_blood = pd.read_csv("C://Users//Alan‘s Lenovo//Desktop//blood_meta.csv")
snmdata_blood = pd.read_csv("C://Users//Alan‘s Lenovo//Desktop//blood_snm.csv")
mldataX_blood = snmdata_blood.values[:,1:]
mldataY_blood = metadata_blood.values[:,8]
```


```python
#去除组织数据量较少的三种疾病
delete_list = []
i = 0
for i in range(len(mldataY)):
    if mldataY[i] == 'Kidney Chromophobe' or mldataY[i] == 'Kidney Renal Clear Cell Carcinoma' or mldataY[i] == 'Lymphoid Neoplasm Diffuse Large B-cell Lymphoma':
        delete_list.append(i)
mldataY = np.delete(mldataY,delete_list,axis=0)
mldataX = np.delete(mldataX,delete_list,axis=0)
```


```python
#去除血液数据量较少的三种疾病
delete_list_blood = []
i = 0
for i in range(len(mldataY_blood)):
    if mldataY_blood[i] == 'Kidney Chromophobe' or mldataY_blood[i] == 'Kidney Renal Clear Cell Carcinoma' or mldataY_blood[i] == 'Lymphoid Neoplasm Diffuse Large B-cell Lymphoma':
        delete_list_blood.append(i)
mldataY_blood = np.delete(mldataY_blood,delete_list_blood,axis=0)
mldataX_blood = np.delete(mldataX_blood,delete_list_blood,axis=0)
```


```python
x_train,x_test,y_train,y_test = train_test_split(mldataX,mldataY,test_size = 0.3,random_state = 0)
x_train_blood,x_test_blood,y_train_blood,y_test_blood = train_test_split(mldataX_blood,mldataY_blood,test_size = 0.9,random_state = 0)
```


```python
#各个模型训练集、测试集数据统计
#组织模型训练集
tissue_train_classes, tissue_train_counts = np.unique(y_train, return_counts=True)
tissue_train_statistic = convert_to_dic(tissue_train_classes, tissue_train_counts)
#组织模型测试集
tissue_test_classes, tissue_test_counts = np.unique(y_test, return_counts=True)
tissue_test_statistic = convert_to_dic(tissue_test_classes, tissue_test_counts)
#血液模型训练集
blood_train_classes, blood_train_counts = np.unique(y_train_blood, return_counts=True)
blood_train_statistic = convert_to_dic(blood_train_classes, blood_train_counts)
#血液模型测试集
blood_test_classes, blood_test_counts = np.unique(y_test_blood, return_counts=True)
blood_test_statistic = convert_to_dic(blood_test_classes, blood_test_counts)
```


```python
dataframe = pd.DataFrame(blood_test_counts)
dataframe.to_excel('22.xls')
```


```python
#*** Data filtering ***
#组织训练集
x_tissue = x_train
y_tissue = y_train
y_tissue, label_dic = label_convert(y_tissue, dic=True)
#组织测试集
x_query_tissue = x_test
y_query_tissue = y_test
y_query_tissue = label_convert(y_query_tissue)
#血液/迁移训练集
x_blood = x_train_blood
y_blood = y_train_blood
y_blood, label_dic = label_convert(y_blood, dic=True)
#血液/迁移测试集
x_query_blood = x_test_blood
y_query_blood = y_test_blood
y_query_blood = label_convert(y_query_blood)

Nclasses = len(tissue_train_classes)
```


```python
#组织模型
tissue_RF = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=13)
tissue_RF = tissue_RF.fit(x_tissue, y_tissue)

tissue_importances = tissue_RF.feature_importances_
tissue_indices = np.argsort(tissue_importances)[::-1]

tissue_y_pred = tissue_RF.predict(x_query_tissue)
tissue_y_proba = tissue_RF.predict_proba(x_query_tissue)

tissue_acc, tissue_acc_class = Balanced_acc(tissue_y_pred, y_query_tissue, Nclasses)
tissue_roc_auc, tissue_fpr, tissue_tpr = roc_auc_calculate(y_query_tissue, tissue_y_proba)
tissue_filename = "tissue-tissue RandomForest"
roc_curve_plot(tissue_roc_auc, tissue_fpr, tissue_tpr, tissue_filename)
```


    
![png](output_8_0.png)
    



```python
from sklearn.metrics import precision_recall_fscore_support
tissue_precision,tissue_recall,tissue_fscore,tissue_mirco = precision_recall_fscore_support(y_query_tissue,tissue_y_pred)
dataframe = pd.DataFrame(tissue_precision)
dataframe.to_excel('precision.xls')
dataframe = pd.DataFrame(tissue_recall)
dataframe.to_excel('recall.xls')
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1221: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    


```python
roc = []
for i in range(21):
    roc.append(tissue_roc_auc[i])
dataframe = pd.DataFrame(roc)
dataframe.to_excel('tissue_roc_auc.xls')
```


```python
dataframe = pd.DataFrame(tissue_importances)
dataframe.to_excel('s.xls')
```


```python
#血液模型
blood_RF = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=13)
blood_RF = blood_RF.fit(x_blood, y_blood)

blood_importances = blood_RF.feature_importances_
blood_indices = np.argsort(blood_importances)[::-1]

blood_y_pred = blood_RF.predict(x_query_blood)
blood_y_proba = blood_RF.predict_proba(x_query_blood)

blood_acc, blood_acc_class = Balanced_acc(blood_y_pred, y_query_blood, Nclasses)
blood_roc_auc, blood_fpr, blood_tpr = roc_auc_calculate(y_query_blood, blood_y_proba)
blood_filename = "blood-blood RandomForest"
roc_curve_plot(blood_roc_auc, blood_fpr, blood_tpr, blood_filename)
```


    
![png](output_12_0.png)
    



```python
roc = []
for i in range(21):
    roc.append(blood_roc_auc[i])
dataframe = pd.DataFrame(roc)
dataframe.to_excel('blood_roc_auc.xls')
```


```python
dataframe = pd.DataFrame(blood_importances)
dataframe.to_excel('feature贡献度.xls')
```


```python
from sklearn.metrics import precision_recall_fscore_support
blood_precision,blood_recall,blood_fscore,blood_mirco = precision_recall_fscore_support(y_query_blood,blood_y_pred)
dataframe = pd.DataFrame(blood_precision)
dataframe.to_excel('precision.xls')
dataframe = pd.DataFrame(blood_recall)
dataframe.to_excel('recall.xls')
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1221: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    


```python
#*** Use MIX(SER + STRUT) to enhance forest trained on Source with target data ***
##*** SER ***
gRF_list = trf.forest_convert(tissue_RF)
ser_RF = trf.forest_SER(gRF_list, x_blood, y_blood, C=Nclasses)
ser_y_pred = trf.predict(ser_RF, x_query_blood)
ser_y_proba = trf.predict_proba(ser_RF, x_query_blood)

ser_acc, ser_acc_class = Balanced_acc(ser_y_pred, y_query_blood, Nclasses)
ser_roc_auc, ser_fpr, ser_tpr = roc_auc_calculate(y_query_blood, ser_y_proba)
```


```python
##*** STRUT ***
strut_RF = trf.STRUT(x_tissue, y_tissue, x_blood, y_blood, n_trees=100, verbos=False)
strut_y_pred = trf.predict(strut_RF, x_query_blood)
strut_y_proba = trf.predict_proba(strut_RF, x_query_blood)

strut_acc, strut_acc_class = Balanced_acc(strut_y_pred, y_query_blood, Nclasses)
strut_roc_auc, strut_fpr, strut_tpr = roc_auc_calculate(y_query_blood, strut_y_proba)
```

    C:\ProgramData\Anaconda3\lib\site-packages\scipy\sparse\_index.py:124: SparseEfficiencyWarning: Changing the sparsity structure of a csr_matrix is expensive. lil_matrix is more efficient.
      self._set_arrayXarray(i, j, x)
    C:/Users/Alan‘s Lenovo/Desktop/TransferRandomForest/Doc\TransferRandomForest.py:356: RuntimeWarning: invalid value encountered in true_divide
      return np.sum(np.where(p != 0,p * np.log10((p / q)), 0))
    C:/Users/Alan‘s Lenovo/Desktop/TransferRandomForest/Doc\TransferRandomForest.py:356: RuntimeWarning: divide by zero encountered in log10
      return np.sum(np.where(p != 0,p * np.log10((p / q)), 0))
    C:/Users/Alan‘s Lenovo/Desktop/TransferRandomForest/Doc\TransferRandomForest.py:356: RuntimeWarning: invalid value encountered in multiply
      return np.sum(np.where(p != 0,p * np.log10((p / q)), 0))
    C:/Users/Alan‘s Lenovo/Desktop/TransferRandomForest/Doc\TransferRandomForest.py:356: RuntimeWarning: divide by zero encountered in true_divide
      return np.sum(np.where(p != 0,p * np.log10((p / q)), 0))
    


```python
##*** MIX ***
mix_y_pred = trf.mix_predict(ser_RF, strut_RF, x_query_blood)
mix_y_proba = trf.mix_predict_proba(ser_RF, strut_RF, x_query_blood)

mix_acc, mix_acc_class = Balanced_acc(mix_y_pred, y_query_blood, Nclasses)
mix_roc_auc, mix_fpr, mix_tpr = roc_auc_calculate(y_query_blood, mix_y_proba)

mix_filename = "tissue-blood transfer RandomForest"
roc_curve_plot(mix_roc_auc, mix_fpr, mix_tpr, mix_filename)
```


    
![png](output_18_0.png)
    



```python
from sklearn.metrics import precision_recall_fscore_support
mix_precision,mix_recall,mix_fscore,mix_mirco = precision_recall_fscore_support(y_query_blood,mix_y_pred)
dataframe = pd.DataFrame(mix_precision)
dataframe.to_excel('precision.xls')
dataframe = pd.DataFrame(mix_recall)
dataframe.to_excel('recall.xls')
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1221: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    


```python
roc = []
for i in range(21):
    roc.append(mix_roc_auc[i])
dataframe = pd.DataFrame(roc)
dataframe.to_excel('blood_roc_auc.xls')
```


```python
lw = 2
plt.figure()
plt.plot(tissue_fpr["macro"], tissue_tpr["macro"], color="deeppink", linestyle=":", linewidth=4,
    label="tissue-tissue ROC curve (area = {0:0.2f})".format(tissue_roc_auc["macro"]),)
plt.plot(blood_fpr["macro"], blood_tpr["macro"], color="navy", linestyle=":", linewidth=4,
    label="blood-blood ROC curve (area = {0:0.2f})".format(blood_roc_auc["macro"]),)
plt.plot(mix_fpr["macro"], mix_tpr["macro"], color="cornflowerblue", linestyle=":", linewidth=4,
    label="tissue-blood tranfer ROC curve (area = {0:0.2f})".format(mix_roc_auc["macro"]),)
plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Transfer Random Forest ROC curve")
plt.legend(loc="lower right")
filename = "Transfer Random Forest ROC curve"
filepath = "./" + filename + ".pdf"
plt.savefig(filepath,bbox_inches = 'tight')
plt.show()
```


    
![png](output_21_0.png)
    



```python

```


```python

```
