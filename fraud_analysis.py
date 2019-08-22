import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# Expand the size of terminal window to display all rows
pd.set_option('display.max_columns', 500)

# Read file including all generated variables
df_total = pd.read_csv('click_count.csv')

# Split dataset (train/test: 80/20)
row = int(0.8*len(df_total))
train = df_total.iloc[:row,]
test = df_total.iloc[row:,]


# Transform high-cardinality categorical features

# transform "channel"
trans_channel = pd.DataFrame(train.query('is_attributed ==1').
                             groupby(['channel','is_attributed']).
                             count()['os']).merge(
    pd.DataFrame(train.groupby(['channel']).count()['os']),
    on = 'channel', how = 'right').fillna(0)
trans_channel['trans_channel'] = trans_channel['os_x']/trans_channel['os_y']
trans_channel = trans_channel.drop(['os_x','os_y'], axis = 1)

train = train.merge(trans_channel, on = 'channel', how = 'left')
test = test.merge(trans_channel, on = 'channel', how = 'left')


# transform "app"
trans_app = pd.DataFrame(train.query('is_attributed ==1').
                             groupby(['app','is_attributed']).
                             count()['os']).merge(
    pd.DataFrame(train.groupby(['app']).count()['os']),
    on = 'app', how = 'right').fillna(0)
trans_app['trans_app'] = trans_app['os_x']/trans_app['os_y']
trans_app = trans_app.drop(['os_x','os_y'], axis = 1)

train = train.merge(trans_app, on = 'app', how = 'left')
test = test.merge(trans_app, on = 'app', how = 'left')


# transform "os"
trans_os = pd.DataFrame(train.query('is_attributed ==1').
                             groupby(['os','is_attributed']).
                             count()['channel']).merge(
    pd.DataFrame(train.groupby(['os']).count()['channel']),
    on = 'os', how = 'right').fillna(0)
trans_os['trans_os'] = trans_os['channel_x']/trans_os['channel_y']
trans_os = trans_os.drop(['channel_x','channel_y'], axis = 1)

train = train.merge(trans_os, on = 'os', how = 'left')
test = test.merge(trans_os, on = 'os', how = 'left')


# transform "device"
trans_device = pd.DataFrame(train.query('is_attributed ==1').
                             groupby(['device','is_attributed']).
                             count()['os']).merge(
    pd.DataFrame(train.groupby(['device']).count()['os']),
    on = 'device', how = 'right').fillna(0)
trans_device['trans_device'] = trans_device['os_x']/trans_device['os_y']
trans_device = trans_device.drop(['os_x','os_y'], axis = 1)

train = train.merge(trans_device, on = 'device', how = 'left')
test = test.merge(trans_device, on = 'device', how = 'left')


# EAD
# Correlation Heatmap of Numerical Features
col_list = ['channel', 'os', 'device', 'app', 'ip', 'is_attributed']

df_corr = df_total.drop(col_list, axis = 1)
corr = df_corr.corr()
plt.figure(figsize=(12,10))
heat = sns.heatmap(data=corr)
plt.title('Heatmap of Correlation')

X = train.drop(['is_attributed'], axis=1)
y = train['is_attributed']

# Data Balancing (Random Under-Sampling)
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(return_indices=True, random_state=1234)
X_rus, y_rus, id_rus = rus.fit_sample(X, y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  # Initialise the Scaler
X_scaled = scaler.fit_transform(X_rus)

# PCA
from sklearn.decomposition import PCA

pca = PCA(.95)  # Choose minimum number of principal components such that 95% of the variance is retained
X_pca = pca.fit_transform(X_scaled)

pca.n_components_


# EAD

## t-SNE
from sklearn.manifold import TSNE

X_reduced_tsne = TSNE(n_components=2, random_state=0).fit_transform(X_rus)

# t-SNE scatter plot  -  Random Under-sampling
import matplotlib.patches as mpatches

f, ax = plt.subplots(figsize=(24,16))


blue_patch = mpatches.Patch(color='#0A0AFF', label='Not_Attributed')
red_patch = mpatches.Patch(color='#AF0000', label='Is_Attributed')

ax.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y_rus == 0), cmap='coolwarm', label='Not_Attributed', linewidths=2)
ax.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y_rus == 1), cmap='coolwarm', label='Is_Attributed', linewidths=2)
ax.set_title('t-SNE', fontsize=18)

ax.grid(True)

ax.legend(handles=[blue_patch, red_patch])


# Model Selection

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier



# Spot-Checking Algorithms

models = []

models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))
models.append(('XGB', XGBClassifier()))
models.append(('RF', RandomForestClassifier()))

#testing models

results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10, random_state=42, shuffle = True)
    if name == ['LR', 'LDA']:
        cv_results = cross_val_score(model, X_pca, y_rus, cv=kfold, scoring='roc_auc')
    elif name == ['KNN', 'SVM']:
        cv_results = cross_val_score(model, X_scaled, y_rus, cv=kfold, scoring='roc_auc')
    else:
        cv_results = cross_val_score(model, X_rus, y_rus, cv=kfold, scoring='roc_auc')
    results.append(cv_results)
    names.append(name)
    msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)



# Compare Algorithms

fig = plt.figure(figsize=(12,10))
plt.title('Comparison of Classification Algorithms')
plt.xlabel('Algorithm')
plt.ylabel('ROC-AUC Score')
plt.boxplot(results)
ax = fig.add_subplot(111)
ax.set_xticklabels(names)
plt.show()

# Parameter tuning - XGBoost
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics  # Additional scklearn functions



def modelfit(alg, dtrain, predictors, dtest, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values,
                              label=dtrain['is_attributed'].values)
        cvresult = xgb.cv(xgb_param, xgtrain,
                          num_boost_round=alg.get_params()['n_estimators'],
                          nfold=cv_folds, metrics='auc',
                          early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['is_attributed'], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Predict test (validation) set:
    dtest_predictions = alg.predict(dtest[predictors])
    dtest_predprob = alg.predict_proba(dtest[predictors])[:, 1]

    # Print model report:
    print(alg)
    print("\nModel Report")
    print("Accuracy (Train) : %.4g" % metrics.accuracy_score(dtrain['is_attributed'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['is_attributed'], dtrain_predprob))

    print("Accuracy (Test) : %.4g" % metrics.accuracy_score(dtest['is_attributed'].values, dtest_predictions))
    print("AUC Score (Test): %f" % metrics.roc_auc_score(dtest['is_attributed'], dtest_predprob))

    feat_imp = pd.Series(alg.get_booster().get_score()).sort_values(ascending=False).head(20)
    feat_imp.plot(kind='barh', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.gca().invert_yaxis()


# train.columns
train_bal_X = pd.DataFrame(X_rus)
train_bal_X.columns = ['channel', 'os', 'device', 'app', 'ip', 'app_1s',
       'channel_1s', 'device_1s', 'ip_1s', 'os_1s', 'app_30s', 'channel_30s',
       'device_30s', 'ip_30s', 'os_30s', 'app_60s', 'channel_60s',
       'device_60s', 'ip_60s', 'os_60s', 'app_3s', 'channel_3s', 'device_3s',
       'ip_3s', 'os_3s', 'app_channel_3s', 'app_device_3s', 'app_ip_3s',
       'app_os_3s', 'channel_app_3s', 'channel_device_3s', 'channel_ip_3s',
       'channel_os_3s', 'device_app_3s', 'device_channel_3s', 'device_ip_3s',
       'device_os_3s', 'ip_app_3s', 'ip_channel_3s', 'ip_device_3s',
       'ip_os_3s', 'os_app_3s', 'os_channel_3s', 'os_device_3s', 'os_ip_3s',
       'app_10s', 'channel_10s', 'device_10s', 'ip_10s', 'os_10s',
       'app_channel_10s', 'app_device_10s', 'app_ip_10s', 'app_os_10s',
       'channel_app_10s', 'channel_device_10s', 'channel_ip_10s',
       'channel_os_10s', 'device_app_10s', 'device_channel_10s',
       'device_ip_10s', 'device_os_10s', 'ip_app_10s', 'ip_channel_10s',
       'ip_device_10s', 'ip_os_10s', 'os_app_10s', 'os_channel_10s',
       'os_device_10s', 'os_ip_10s', 'trans_channel', 'trans_app', 'trans_os',
       'trans_device']
train_bal_y = pd.DataFrame(y_rus)
train_bal_y.columns = ['is_attributed']
train_bal = pd.concat([train_bal_X, train_bal_y], axis = 1).set_index('ip')

# feature_list = [...]   ## list of features selected to train the model


# Step 1: Fix learning rate and number of estimators for tuning tree-based parameters

predictors = [x for x in feature_list]
xgb1 = XGBClassifier(learning_rate = 0.01,
                     n_estimators = 1000,
                     max_depth = 5,
                     min_child_weight = 1,
                     gamma = 0,
                     subsample = 0.8,
                     colsample_bytree = 0.8,
                     objective = 'binary:logistic',
                     nthread = 4,
                     scale_pos_weight = 1,
                     seed = 27)

modelfit(alg = xgb1, dtrain = train_bal, predictors = predictors, dtest = test)

# Keep learning_rate and n_estimators fix for following tuning steps.



# Step 2: Tune max_depth and min_child_weight

from sklearn.model_selection import GridSearchCV

param_test1b = {
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2)
}
gsearch1b = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.01, n_estimators = 232,
                                                  max_depth = 5, min_child_weight = 1,
                                                  gamma = 0, subsample = 0.8,
                                                  colsample_bytree = 0.8,
                                                  objective = 'binary:logistic',
                                                  nthread = 4, scale_pos_weight = 1,
                                                  seed = 27),
                        param_grid = param_test1b, scoring = 'roc_auc', n_jobs = 4,
                        iid = False, cv = 5)

gsearch1b.fit(train_bal[predictors],train_bal['is_attributed'])
gsearch1b.cv_results_, gsearch1b.best_params_, gsearch1b.best_score_

# Set the optimized pair of max_depth and min_child_weight for following tuning steps.


# Step 3: Tune gamma -  the minimum loss reduction required to make a split

param_test3 = {
 'gamma':[i/10.0 for i in range(0,6)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.01, n_estimators = 232,
                                                  max_depth = 5, min_child_weight = 1,
                                                  gamma = 0, subsample = 0.8,
                                                  colsample_bytree = 0.8,
                                                  objective = 'binary:logistic',
                                                  nthread = 4, scale_pos_weight = 1,
                                                  seed = 27),
                        param_grid = param_test3, scoring = 'roc_auc', n_jobs = 4,
                        iid = False, cv = 5)

gsearch3.fit(train_bal[predictors],train_bal['is_attributed'])
gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_

# Set the optimized value of gamma for following tuning steps.


# Step 4: Tune subsample and colsample_bytree

param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(3,7)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.01, n_estimators = 232,
                                                  max_depth = 5, min_child_weight = 1,
                                                  gamma = 0, subsample = 0.6,
                                                  colsample_bytree = 0.6,
                                                  objective = 'binary:logistic',
                                                  nthread = 4, scale_pos_weight = 1,
                                                  seed = 27),
                        param_grid = param_test4, scoring = 'roc_auc', n_jobs = 4,
                        iid = False, cv = 5)

gsearch4.fit(train_bal[predictors],train_bal['is_attributed'])
gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_


## Try values in 0.05 interval

param_test5 = {
 'subsample':[i/100.0 for i in range(70,100,5)],
 'colsample_bytree':[i/100.0 for i in range(20,50,5)]
}
gsearch5 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.01, n_estimators = 232,
                                                  max_depth = 5, min_child_weight = 1,
                                                  gamma = 0, subsample = 0.6,
                                                  colsample_bytree = 0.6,
                                                  objective = 'binary:logistic',
                                                  nthread = 4, scale_pos_weight = 1,
                                                  seed = 27),
                        param_grid = param_test5, scoring = 'roc_auc', n_jobs = 4,
                        iid = False, cv = 5)

gsearch5.fit(train_bal[predictors],train_bal['is_attributed'])
gsearch5.cv_results_, gsearch5.best_params_, gsearch5.best_score_

