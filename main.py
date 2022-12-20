#%%
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

from mySqlConn import df2table, table2df

#Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import ShuffleSplit, learning_curve
import collections

#My modules
from myModule import underSample, corrHeatmapVis

#Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_recall_curve, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# from dataprep.eda import plot, create_report

# %%
file = r'data\creditcard.csv'
df = pd.read_csv(file)
print(f"shape df: {df.shape}")
df.head()

df2table(df, 'root', '949700', 'creditdb', 'creditcardTBL')

# df = table2df('root', '949700', 'creditdb', 'creditcardtbl')

# report = create_report(df)
# report.save('creditcard_summary.html')
#%%
## imbalanced data set check()
nonFraudsRate = np.round((len(df[df.Class==0]) / len(df.Class) * 100), 2)
fraudsRate = np.round((len(df[df.Class==1]) / len(df.Class) * 100), 2)

print('No Frauds {0} %'.format(nonFraudsRate), 'of the dataset')
print('Frauds {0} %'.format(fraudsRate), 'of the dataset')

plt.title('Class Distributions\n(0: No Fraud || 1: Fraud)')
sns.countplot(df, x='Class')


#%%
# 컬럼명 명시된 feature의 분포 확인
cols = ['Amount', 'Time']
plt.figure(figsize=(10,5))

for i, col in enumerate(cols):
    plt.subplot(1, 2, i+1)
    plt.title('distribution of {0}'.format(col))
    sns.distplot(df[col])


# %%
## 데이터 스케일링
rb_sc = RobustScaler()

df['scaled_amount'] = rb_sc.fit_transform(df[['Amount']])
df['scaled_time'] = rb_sc.fit_transform(df[['Time']])

df.drop(['Time', 'Amount'], axis=1, inplace=True)

# %%
## 스케일링된 컬럼 앞으로 가져오기
newCols = df.columns[:-2].to_list()
newCols.insert(0, 'scaled_time')
newCols.insert(0, 'scaled_amount')

df = df[newCols]
df.head()

# %%
## 데이터 분할(& skf의 각 fold에서 y값의 비중 확인)
X = df.drop(['Class'], axis=1)
y = df['Class']
print(X.shape, y.shape)

skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)

for train_idx, test_idx in skf.split(X, y):
    # print('train_idx :', train_idx, 'test_idx :', test_idx)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    print(y_test.value_counts()[1] / len(y_test))

# np.unique()
# %%
# data under sampling
newDf = underSample(df, 'Class')

# %%
print('Class 0과 1의 비율')
print(newDf.Class.value_counts()[0] / newDf.Class.value_counts()[1], ':', newDf.Class.value_counts()[1] / newDf.Class.value_counts()[1])

sns.countplot(data=newDf, x='Class')
# %%'
mask = np.zeros_like(df.corr(), dtype=bool)
mask[np.triu_indices_from(mask)] = True

dfList = [df, newDf]    

plt.figure(figsize=(6,12))
for i, myDf in enumerate(dfList):
    corrHeatmapVis(myDf, len(dfList), i, mask, 'coolwarm_r')

# %%
# 강한 음의 상관 관계 : 10 12 14 17
# 강한 양의 상관 관계 : 2 4 11 19

cols1 = [10, 12, 14, 17]
cols2 = [2, 4, 11, 19]

plt.figure(figsize=(20,10))
for i, col in enumerate(cols1):
    plt.subplot(1, 4, i+1)
    sns.boxplot(data=newDf, x='Class', y='V{0}'.format(col))

plt.figure(figsize=(20,10))
for i, col in enumerate(cols2):
    plt.subplot(1, 4, i+1)
    sns.boxplot(data=newDf, x='Class', y='V{0}'.format(col))

# %%
# 상관관계가 높은 애들의 이상치 제거
# 이 변수들은 학습시 가중치가 높을 것인데 이 경우 이상치의 영향이 커질 우려가 있음
# 따라서 정확도를 높이기 위해 이상치 제거..

# V14
v14q1 = np.quantile(newDf['V14'].values, 0.25)
v14q3 = np.quantile(newDf['V14'].values, 0.75)
v14Iqr = v14q3 - v14q1
v14Lower = v14q1 - (1.5 * v14Iqr)
v14Upper = v14q3 + (1.5 * v14Iqr)
print('-- v14의 upper, lower --\n', v14Upper, ',', v14Lower, '\n\n')

# V12
v12q1 = np.quantile(newDf['V12'].values, 0.25)
v12q3 = np.quantile(newDf['V12'].values, 0.75)
v12Iqr = v12q3 - v12q1
v12Lower = v12q1 - (1.5 * v12Iqr)
v12Upper = v12q3 + (1.5 * v12Iqr)
print('-- v12의 upper, lower --\n', v12Upper, ',', v12Lower, '\n\n')

# V10
v10q1 = np.quantile(newDf['V10'].values, 0.25)
v10q3 = np.quantile(newDf['V10'].values, 0.75)
v10Iqr = v10q3 - v10q1
v10Lower = v10q1 - (1.5 * v10Iqr)
v10Upper = v10q3 + (1.5 * v10Iqr)
print('-- v10의 upper, lower --\n', v10Upper, ',', v10Lower, '\n\n')

newDf = newDf[(newDf['V14']>=v14Lower) & (newDf['V14']<=v14Upper)]
newDf = newDf[(newDf['V12']>=v12Lower) & (newDf['V12']<=v12Upper)]
newDf = newDf[(newDf['V10']>=v10Lower) & (newDf['V10']<=v10Upper)]

newDf
# %%
X = newDf.drop(['Class'], axis=1)
y = newDf['Class']

tsne = TSNE(n_components=2, random_state=42)
time0 = time.time()
X_tsne = tsne.fit_transform(X.values)
time1 = time.time()
print('T-SNE 소요시간 :', time1-time0)

pca = PCA(n_components=2, random_state=42)
time0 = time.time()
X_pca = pca.fit_transform(X.values)
time1 = time.time()
print('PCA 소요시간 :', time1-time0)

tSvd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42)
time0 = time.time()
X_tsvd = tSvd.fit_transform(X.values)
time1 = time.time()
print('tSvd 소요시간 :', time1-time0)

# %%
# noFrPatch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
# frPatch = mpatches.Patch(color='#AF0000', label='Fraud')

algos = [(X_tsne, 'T-SNE'), (X_pca, 'PCA'), (X_tsvd, 'TruncatedSVD')]

plt.figure(figsize=(24,6))
for i, algo in enumerate(algos):
    ax = plt.subplot(1,3,i+1)
    ax.scatter(algo[0][:,0], algo[0][:,1], c=(y==0), cmap='coolwarm', label='No Fraud', linewidths=2)
    ax.scatter(algo[0][:,0], algo[0][:,1], c=(y==1), cmap='coolwarm', label='No Fraud', linewidths=2)
    ax.set_title(algo[1], fontsize=14)
    ax.legend()
    ax.grid(True)

# %%
X = newDf.drop(['Class'], axis=1)
y = newDf.Class

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# %%
# 4개의 분류 알고리즘으로 정확도 측정
lgRg = LogisticRegression()
knn = KNeighborsClassifier()
svc = SVC()
dtClf = DecisionTreeClassifier()
algos = [(lgRg,'LogisticRegression'), 
        (knn, 'KNeigborsClassifier'), 
        (svc, 'SupportVectorClassifier'), 
        (dtClf, 'DecisionTreeClassifier')]

for i, algo in enumerate(algos):
    algo[0].fit(X_train, y_train)
    pred = algo[0].predict(X_test)
    acc_sc = np.round(accuracy_score(y_test, pred),4)
    print('--', algo[1], '--')
    print('예측 정확도 :', acc_sc*100,'% \n')

# %%
# cross_validation
for i, algo in enumerate(algos):
    accs = cross_val_score(algo[0], X_train, y_train, cv=5)
    # pred = algo[0].predict(X_test)
    # acc_sc = np.round(accuracy_score(y_test, pred),4)
    print('--', algo[1], '--')
    print(f'{algo[0].__class__.__name__} 교차 검증 정확도 :', np.round(accs.mean()*100, 3),'% \n')

# %%
# 하이퍼 파라미터 튜닝
lgRg = LogisticRegression()
knn = KNeighborsClassifier()
svc = SVC()
dtClf = DecisionTreeClassifier()

lrParams = {"penalty" : ['l1', 'l2'],
            'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
knnParams = {"n_neighbors" : list(range(2,5,1)), 
             "algorithm" : ['auto', 'ball_tree', 'kd_tree', 'brute']}
svcParams = {'C' : [0.5, 0.7, 0.9, 1], 
             'kernel' : ['rbf', 'poly', 'sigmid', 'linear']}
dtParams = {'criterion' : ['entropy', 'gini'],
            'max_depth' : list(range(2,4,1)),
            'min_samples_leaf' : list(range(5,7,1))}


def myGridSearch(algo, params, X_test):
    gridCV = GridSearchCV(algo, params, cv=5)
    gridCV.fit(X_train, y_train)
    bestEst = gridCV.best_estimator_
    bestPred = bestEst.predict(X_test)
    acc_sc = np.round(accuracy_score(y_test, bestPred), 3) * 100
    return bestEst, acc_sc

myLr, myLrAcc = myGridSearch(lgRg, lrParams, X_test)
myKnn, myKnnAcc = myGridSearch(knn, knnParams, X_test)
mySvc, mySvcAcc = myGridSearch(svc, svcParams, X_test)
myDt, myDtAcc = myGridSearch(dtClf, dtParams, X_test)

print('-----' ,'하이퍼 파라미터 튜닝 결과', '-----')
print(f"{myLr.__class__.__name__}의 최종 accuracy : {myLrAcc:.1f} %")
print(f"{myKnn.__class__.__name__}의 최종 accuracy : {myKnnAcc:.1f} %")
print(f"{mySvc.__class__.__name__}의 최종 accuracy : {mySvcAcc:.1f} %")
print(f"{myDt.__class__.__name__}의 최종 accuracy : {myDtAcc:.1f} %")

# %%
algos = [myLr, myKnn, mySvc, myDt]

def learningCurveDraw(algo):
    trainSizes, trainScores, testScores = learning_curve(algo, X, y, cv=10, n_jobs=1, train_sizes = np.linspace(.1, 1.0, 50))
    trainScoresMean = np.mean(trainScores, axis=1)
    testScoresMean = np.mean(testScores, axis=1)
    plt.plot(trainSizes, trainScoresMean, 'o-', color='blue', label='Training score')
    plt.plot(trainSizes, testScoresMean, 'o-', color='red', label='Cross validation score')
    plt.legend(loc='best')

plt.figure(figsize=(20,20))
for i, algo in enumerate(algos):
    plt.subplot(2,2,i+1)
    learningCurveDraw(algo)

# %%
