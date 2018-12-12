#!/usr/bin/env python
# coding: utf-8

# # 使用 scikit-learn 玩转机器学习——逻辑回归

# 蜉蝣扶幽

# ## 小引

# 逻辑回归，咋一听这名字，真的跟一个正儿八经的回归模型似的，实际上从原理上讲他是一个如假包换的分类器，英文名是 Logistics regression，也叫 logit regression，maximum-entropy classification，或者 log-linear classifier。在逻辑回归中，会将样本的所有特征与样本属于某个种类的概率联系起来，即使每个特征都对应一个模型参数，通过训练不断修正模型参数，最后使用 logistic function （也叫 sigmoid 函数，该函数使输入从（-inf, inf）映射到输出（0，1），其图形如下）建模求出样本属于某个种类的概率。

# ![image.png](attachment:image.png)

# 下图是 Kaggle 上出现的统计在各个行业使用机器学习方法的使用率的情况。从图中可以看出，逻辑回归以领先第二名13.6%的绝对优势牢牢地占据了第一位。逻辑回归之所以能够坚挺在 C 位，绝对有与之相匹配的实力，我们会在代码实战中看到相关证明。

# ![image.png](attachment:image.png)

# 在 scikit-learn 中封装的逻辑回归，可以用来解决二分类和基于 OvR 和 OvO 的多分类问题。逻辑回归的损失函数是一个凸函数，存在全局最优解，为避免模型过拟合，常需要对模型进行正则化。所以，scikit-learn 中逻辑回归的实现的损失函数常加有惩罚项来对模型正则化。加上 L1 正则化项的损失函数为：

# ![image.png](attachment:image.png)

# 加上 L1 正则项的损失函数为：

# ![image.png](attachment:image.png)

# ## 实战

# 下面我们就用代码具体的展示下 scikit-learn 中，逻辑回归的使用、性能、以及进行一定的调参后的表现。

# In[3]:


# 引入必要的包
import numpy as np
import matplotlib.pyplot as plt

# 手工制造数据集
np.random.seed(2333)
X = np.random.normal(0, 3, size=(400, 2))
y = np.array(X[:,0]**2 + X[:,1]**2 < 10.5, dtype=np.int)

# 画出两类数据的分布情况
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()


# In[11]:


# 从 scikit-learn 中引入必要的机器学习模型
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 将训练集和测试集分离
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=233)

# 实例化一个逻辑回归模型，进行训练，并打印模型精度
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print("Accuracy: ", log_reg.score(X_test, y_test))


# 默认模型的表现好像很差劲呐，还不如去抛硬币呢！那我们接下来就介绍下 scikit-learn 中封装的逻辑回归的一些超参数，并进行一定的调整，看看能提高多大程度的性能。

# 可调整的超参数：
# 
# penalty: str 类型，可取 ‘l1’ 或者 ‘l2’，默认值为 ‘l2’，用于明确损失函数惩罚项的正则类型；
# 
# tol: float 类型，默认值为 1e-4，表示容差，用于决定是否停止搜索；
# 
# C: float 类型，默认值为1.0，表示正则项的系数，用来决定模型需要正则化的程度；
# 
# solver: str 类型，默认取 'liblinear'，可取{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’} 中任一个，表示在优化过程中所使用的算法；
# 
# max_iter: int 型，默认取100，表示在优化收敛的过程中最大迭代的次数；
# 
# 更多超参数，更详细的说明，请小伙伴们查阅官方文档。

# 我们先变化下两个超参数，令 C=0.1，penalty='l1' 试试：

# In[21]:


# 实例化一个逻辑回归模型，并传入相应的超参数，接着训练，打印模型准确度
log_reg2 = LogisticRegression(C=0.1, penalty='l1')
log_reg2.fit(X_train, y_train)
print("Accuracy: ", log_reg2.score(X_test, y_test))


# 这精度还是不行，稍微分析下数据可知，样本的标签值是与样本的特征有多项式关系，这就很容易想到可以用 scikit-learn 中的 PolynomialFeatures 类对数据进行预处理，这个过程可以用 Pipeline 进行简化处理。实现如下：

# In[23]:


# 引入必要的包
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

# 使用 Pipeline 将数据特征的多项式处理与模型实例化封装在一起为一个函数
def LogisticPolyRegression(degree, C=1.0):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std', StandardScaler()),
        ('Log_reg', LogisticRegression(C=C))
    ])

# 实例化一个多项式回归模型，并进行训练，打印模型的精度
log_polyreg = LogisticPolyRegression(2)
log_polyreg.fit(X_train, y_train)
print("Accuracy: ",log_polyreg.score(X_test, y_test))


# 这回还不错，模型的精度达到了97%，为了进行比较，我们又引入两个其他模型———kNN 和 ANN，虽然都有两个 NN ，但这两个模型的确是没什么血缘关系，唯一的共同点估计就是都在分类领域有着不错的性能了吧！kNN 是 k 近邻算法，ANN 是人工神经网络，下面我们先看下 kNN 的表现。

# In[24]:


# 实例化一个 kNN 模型，并进行训练
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
# 打印模型的准确度
print("Accuracy: ", knn_clf.score(X_test, y_test))


# kNN 精度还不错，再看 ANN 的：

# ![image.png](attachment:image.png)

# 经过100轮（EPOCHES=100）的训练，ANN的精度为 85%，还行，但不是特别的出色，相比已经达到百分之九十多的 kNN 和多项式回归来说。不过当训练500轮时，模型的准确率可以达到100%，但随之增长的是计算代价。

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# 这次分享就到这里，小伙伴们下次再见。
