from __future__ import print_function
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular

"""
代码参考：
https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20continuous%20and%20categorical%20features.html
"""

# 设置随机种子
np.random.seed(1)

# 划分训练-测试集，比例8：2
iris = sklearn.datasets.load_iris()
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(iris.data, iris.target,
                                                                                  train_size=0.80)
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train, labels_train)
print("accuracy: %f"  %(sklearn.metrics.accuracy_score(labels_test, rf.predict(test))) )

# 创建解释器对象
"""
解释器对象的创建需要 训练集，train参数，对于连续型变量，计算了均值和方差，然后进行了离散化，对于离散型变量，
计算了每个取值出现的频次。
计算这些统计量有2个作用：
1.用于scale数据，使得取值范围不一样
2.用于抽样数据
"""
explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=iris.feature_names,
                                                   class_names=iris.target_names, discretize_continuous=True)

i = np.random.randint(0, test.shape[0])
print("pick i-th: %d" %(i)) # 27
print(iris.feature_names)
print("test[i]: %s" %(str(test[i])) )
exp = explainer.explain_instance(test[i], rf.predict_proba, num_features=4, top_labels=1)


"""
show_in_notebook() 不能直接用在IDE里，显示不了的。
改用save_to_file()，把它保存成一个html文件，然后用浏览器打开。

num_features=4, 一共就4个特征，也就是查看所有特征的影响。

HTML文件里，最右边的表格，feature，value是该条样本的特征名称和值。



"""
# exp.show_in_notebook(show_table=True, show_all=False)

html_file_path = "D:\练手\PPP\LIME_related\123"
exp.save_to_file(file_path=html_file_path, show_table=True, show_all=True)
print("html file is saved to: %s" %(html_file_path))


feature_index = lambda x: iris.feature_names.index(x)
print('Increasing petal width')
temp = test[i].copy()
print('P(setosa) before:', rf.predict_proba(temp.reshape(1, -1))[0, 0])
temp[feature_index('petal width (cm)')] = 1.5
print('P(setosa) after:', rf.predict_proba(temp.reshape(1, -1))[0, 0])
print()
print('Increasing petal length')
temp = test[i].copy()
print('P(setosa) before:', rf.predict_proba(temp.reshape(1, -1))[0, 0])
temp[feature_index('petal length (cm)')] = 3.5
print('P(setosa) after:', rf.predict_proba(temp.reshape(1, -1))[0, 0])
print()
print('Increasing both')
temp = test[i].copy()
print('P(setosa) before:', rf.predict_proba(temp.reshape(1, -1))[0, 0])
temp[feature_index('petal width (cm)')] = 1.5
temp[feature_index('petal length (cm)')] = 3.5
print('P(setosa) after:', rf.predict_proba(temp.reshape(1, -1))[0, 0])

exp.show_in_notebook(show_table=True, show_all=True)


