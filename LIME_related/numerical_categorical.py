"""
下面的是离散型特征和连续性特征同时存在的情况下的解释
代码参考 Numerical and Categorical features in the same dataset 部分
https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20continuous%20and%20categorical%20features.html

建立的模型是预测一个人的年收入是否在50K以上。
"""
from __future__ import print_function
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
np.random.seed(1)


feature_names = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status", "Occupation",
                 "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hours per week", "Country"]
data = np.genfromtxt('D:\\练手\\PPP\\LIME_related\\adult.data', delimiter=', ', dtype=str)
labels = data[:, 14]
le = sklearn.preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
class_names = le.classes_
data = data[:, :-1]
categorical_features = [1, 3, 5, 6, 7, 8, 9, 13]
categorical_names = {}
for feature in categorical_features:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(data[:, feature])
    data[:, feature] = le.transform(data[:, feature])
    categorical_names[feature] = le.classes_
data = data.astype(float)
encoder = sklearn.preprocessing.OneHotEncoder(categorical_features=categorical_features)
np.random.seed(1)
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.80)
encoder.fit(data)
encoded_train = encoder.transform(train)

# 这次用xgboost建模

import xgboost

gbtree = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
gbtree.fit(encoded_train, labels_train)
print(gbtree)
print("accuracy: %f " % (sklearn.metrics.accuracy_score(labels_test, gbtree.predict(encoder.transform(test)))))
predict_fn = lambda x: gbtree.predict_proba(encoder.transform(x)).astype(float)

# 解释器
explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=feature_names, class_names=class_names,
                                                   categorical_features=categorical_features,
                                                   categorical_names=categorical_names, kernel_width=3)

np.random.seed(1)
i = 1653
exp = explainer.explain_instance(test[i], predict_fn, num_features=10)
# exp.show_in_notebook(show_all=False)
exp.save_to_file(file_path="D:\\练手\\PPP\\numeric_category_feat_01", show_table=True, show_all=True)

i = 10
exp = explainer.explain_instance(test[i], predict_fn, num_features=10)
# exp.show_in_notebook(show_all=False)
exp.save_to_file(file_path="D:\\练手\\PPP\\numeric_category_feat_02", show_table=True, show_all=True)

