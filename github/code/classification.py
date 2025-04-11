import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, \
    precision_recall_curve, average_precision_score



# 初始化十折交叉验证
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=12)

# 初始化存储结果的列表
accuracies, precisions, recalls, f1s, roc_aucs, avg_precisions = [], [], [], [], [], []
all_yrecalls, all_xprecisions = [], []

# LightGBM参数
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'colsample_bytree': 0.9,
    'subsample': 0.8,
    'random_state': 12
}

# 交叉验证
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    early_stopping_callback = lgb.early_stopping(stopping_rounds=10)
    log_callback = lgb.log_evaluation(period=10)

    clf = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[valid_data],
        callbacks=[early_stopping_callback, log_callback]
    )
    # 输出最佳迭代次数
    print(f"Best iteration for this fold: {clf.best_iteration}")

    y_pred = clf.predict(X_test, num_iteration=clf.best_iteration)
    y_pred_labels = (y_pred >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred_labels)
    precision = precision_score(y_test, y_pred_labels)
    recall = recall_score(y_test, y_pred_labels)
    f1 = f1_score(y_test, y_pred_labels)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    average_precision = average_precision_score(y_test, y_pred)

    xprecision, yrecall, _ = precision_recall_curve(y_test, y_pred)
    all_yrecalls.append(yrecall)
    all_xprecisions.append(xprecision)

    # 将结果添加到列表
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    roc_aucs.append(roc_auc)
    avg_precisions.append(average_precision)

mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
mean_precision = np.mean(precisions)
std_precision = np.std(precisions)
mean_recall = np.mean(recalls)
std_recall = np.std(recalls)
mean_f1 = np.mean(f1s)
std_f1 = np.std(f1s)
mean_roc_auc = np.mean(roc_aucs)
std_roc_auc = np.std(roc_aucs)
mean_avg_precision = np.mean(avg_precisions)
std_avg_precision = np.std(avg_precisions)

# 打印评估指标
print("Mean Accuracy: {:.4f} ± {:.4f}".format(mean_accuracy, std_accuracy))
print("Mean Precision: {:.4f} ± {:.4f}".format(mean_precision, std_precision))
print("Mean Recall: {:.4f} ± {:.4f}".format(mean_recall, std_recall))
print("Mean F1 Score: {:.4f} ± {:.4f}".format(mean_f1, std_f1))
print("Mean ROC AUC: {:.4f} ± {:.4f}".format(mean_roc_auc, std_roc_auc))
print("Mean Average Precision: {:.4f} ± {:.4f}".format(mean_avg_precision, std_avg_precision))
