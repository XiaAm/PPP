"""
下面的是离散型特征解释
代码参考 Categorical features
https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20continuous%20and%20categorical%20features.html
"""
from __future__ import print_function
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
np.random.seed(1)


data = np.genfromtxt('D:\\练手\\PPP\\LIME_related\\agaricus-lepiota.data', delimiter=',', dtype='<U20')
labels = data[:, 0]
le = sklearn.preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
class_names = le.classes_
data = data[:, 1:]
categorical_features = range(22)
feature_names = 'cap-shape,cap-surface,cap-color,bruises?,odor,gill-attachment,gill-spacing,gill-size,gill-color,stalk-shape,stalk-root,stalk-surface-above-ring, stalk-surface-below-ring, stalk-color-above-ring,stalk-color-below-ring,veil-type,veil-color,ring-number,ring-type,spore-print-color,population,habitat'.split(
    ',')

# 把字母与单词做上对应
categorical_names = '''bell=b,conical=c,convex=x,flat=f,knobbed=k,sunken=s
fibrous=f,grooves=g,scaly=y,smooth=s
brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
bruises=t,no=f
almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
attached=a,descending=d,free=f,notched=n
close=c,crowded=w,distant=d
broad=b,narrow=n
black=k,brown=n,buff=b,chocolate=h,gray=g,green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
enlarging=e,tapering=t
bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
fibrous=f,scaly=y,silky=k,smooth=s
fibrous=f,scaly=y,silky=k,smooth=s
brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
partial=p,universal=u
brown=n,orange=o,white=w,yellow=y
none=n,one=o,two=t
cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d'''.split('\n')
for j, names in enumerate(categorical_names):
    values = names.split(',')
    values = dict([(x.split('=')[1], x.split('=')[0]) for x in values])
    data[:, j] = np.array(list(map(lambda x: values[x], data[:, j])))

# 解释器只能用连续性的特征，绝大多数分类器也只能处理连续型的特征
# 此处，将特征编码成整数型
# 把 a,b,c,d 编码成 0,1,2,3
# ！ 此方法有缺陷，不能认为a与b的距离是1而a与c的距离是2

categorical_names = {}
for feature in categorical_features:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(data[:, feature])
    data[:, feature] = le.transform(data[:, feature])
    categorical_names[feature] = le.classes_

data = data.astype(float)
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.80)

# 为了不让它被当成连续值，进行onehot编码
# 但是编码后的东西只给classifier用，不给explainer用
encoder = sklearn.preprocessing.OneHotEncoder(categorical_features=categorical_features)
encoder.fit(data)
encoded_train = encoder.transform(train)
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(encoded_train, labels_train)

predict_fn = lambda x: rf.predict_proba(encoder.transform(x))

sklearn.metrics.accuracy_score(labels_test, rf.predict(encoder.transform(test)))


# 运行explainer 解释样本的预测值
np.random.seed(1)
explainer = lime.lime_tabular.LimeTabularExplainer(train, class_names=['edible', 'poisonous'],
                                                   feature_names=feature_names,
                                                   categorical_features=categorical_features,
                                                   categorical_names=categorical_names, kernel_width=3, verbose=False)
i = 137
exp = explainer.explain_instance(test[i], predict_fn, num_features=10)
exp.show_in_notebook()
exp.save_to_file(file_path="D:\\练手\\PPP\\categorical_feat", show_table=True, show_all=True)

odor_idx = feature_names.index('odor')
print("explainer categorical names: %s" %(str(explainer.categorical_names[odor_idx])))
print("explainer feature frequencies: %s" %(str(explainer.feature_frequencies[odor_idx])))


"""
explainer categorical names: ['almond' 'anise' 'creosote' 'fishy' 'foul' 'musty' 'none' 'pungent'
 'spicy']
explainer feature frequencies: [ 0.0489306   0.04908447  0.02261886  0.06862594  0.43683644  0.26788737
  0.06908755  0.03231266  0.00461609]
"""
"""
注意上面 foul对应是数字是0.2679，0.5-0.26=0.24，也就是说如果odor气味不是foul恶臭，平均来说，
这个预测值将be 0.24 less poisonous
"""



foul_idx = 4
non_foul = np.delete(explainer.categorical_names[odor_idx], foul_idx)
non_foul_normalized_frequencies = explainer.feature_frequencies[odor_idx].copy()
non_foul_normalized_frequencies[foul_idx] = 0
non_foul_normalized_frequencies /= non_foul_normalized_frequencies.sum()
print('Making odor not equal foul')
temp = test[i].copy()
print('P(poisonous) before:', predict_fn(temp.reshape(1, -1))[0, 1])

average_poisonous = 0
for idx, (name, frequency) in enumerate(zip(explainer.categorical_names[odor_idx], non_foul_normalized_frequencies)):
    if name == 'foul':
        continue
    temp[odor_idx] = idx
    p_poisonous = predict_fn(temp.reshape(1, -1))[0, 1]
    average_poisonous += p_poisonous * frequency
    print('P(poisonous | odor=%s): %.2f' % (name, p_poisonous))
print()
print('P(poisonous | odor != foul) = %.2f' % average_poisonous)

"""
Making odor not equal foul
P(poisonous)
before: 1.0
P(poisonous | odor = almond): 0.86
P(poisonous | odor = anise): 0.87
P(poisonous | odor = creosote): 0.89
P(poisonous | odor = fishy): 0.89
P(poisonous | odor = musty): 0.88
P(poisonous | odor = none): 0.70
P(poisonous | odor = pungent): 0.89
P(poisonous | odor = spicy): 0.88
P(poisonous | odor != foul) = 0.77
"""
"""
从上面的数字可以看出
We see that in this particular case, the linear model is pretty close: it predicted that on average odor 
increases the probability of poisonous by 0.26, when in fact it is by 0.23. Notice though that we only 
changed one feature (odor), when the linear model takes into account perturbations of all the features 
at once.
"""
