#!/usr/bin/env python
# coding: utf-8

# # 使用 scikit-learn 玩转机器学习——PCA

# PCA 的全称是 Principal Component Analysis，翻译过来就是主成分分析法，是数据分析中常用的数据降维方法，亦是一种学习数据表示的无监督学习算法。在讨论 PCA 之前，让我们先考虑下机器学习中的数据。
# 
# 对于一般的机器学习模型，每一种确定的机器学习模型都可以用一个确定的函数或函数族来表示。而我们都知道深度学习与一般的机器学习是有差异的，深度学习是个“能吃能干”的人。如果你不让他吃够，他一般也不会好好给你干活，当然他也特别能干，什么都会干，翻译过来就是：只要数据集足够大，质量有保障，就会有不错的输出。也就是说，你给了他输入和输出，他就能学习到输入映射到输出的函数（通用近似定律），这就相当于它能够学习到各种一般机器学习所代表的模型。所以，无论是机器学习，还是深度学习，都需要学习一个训练数据集所表示的模型，先验模型与学习到的模型符合度高的话，当然会有不错的准确率。而这个学习的过程，当然学习的就是各个训练样本的差异，而我们所希望的就是所有训练样本的差异性组合起来能够准确地表示整个模型，这样模型会具有更好的泛化能力。
# 
# PCA 在降维时，就是着重保留这种差异性，使得数据在降维后，数据的差异性损失最小化。PCA 常使用的差异性指标是方差。比方说，我们有一组 N 维数据，要将这组数据降到 N-1 维，那我们所要做的就是在一个 N 维空间中找一个 N-1 维子空间，使得该数据集投影到 N-1 维子空间后的方差最大。所以，PCA 降维问题最后又归结为最值优化问题。下图左为原始数据，图右为降维后数据。

# image41![image.png](attachment:image.png)

# scikit-learn 中的 PCA 默认使用奇异值分解将数据降维到低维空间。同时 scikit-learn 也提供了丰富且简洁友好的 API 接口供使用者调用。以下用代码具体展示：

# In[15]:


# 引入必要的包
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 手动制造数据
X = np.empty((100, 2))
X[:,0] = np.random.uniform(0., 100., size=100)
X[:,1] = X[:,0] * 0.66 + 3.0 + np.random.normal(0, 10., size=100)

# 可视化数据
plt.scatter(X[:,0], X[:,1])
plt.show()


# In[2]:


# 实例化一个 PCA 类，并使数据降维一维
pca = PCA(n_components=1)
pca.fit(X)
# PCA 的 components_ 方法可返回在特征空间中使方差投影最大的轴线方向表示，由于计算方法的
# 关系，符号应取反
print(pca.components_)


# In[3]:


# 使用训练好的PCA模型，对特征空间降维
X_reduction = pca.transform(X)
# 为便于可视化，再将特征空间升维
X_restore = pca.inverse_transform(X_reduction)
# 绘图
plt.title("PCA")
plt.scatter(X[:,0], X[:,1], alpha=0.8)
plt.scatter(X_restore[:,0], X_restore[:,1], color='r', alpha=0.7)
plt.show()


# 在上面的代码中，我们创建了一个符合线性趋势带有噪音的数据集，然后使用 PCA 将这个数据集降维，为了便于在坐标系中可视化，使用 scikit-learn 中 PCA 模型的 inverse_transform 方法重新升维成二维，所有在点就很神奇的落在一条直线上。其实，只要数据发生降维，数据的有效信息就会不可避免的发生丢失，信息一旦丢失就找不回来了。所以，在上图中，我们将降维后的数据重新升维（变成二维），将产生的数据可视化，数据呈现线性关系，全部落在一条直线上，而原始数据则在这条直线附近上下波动， 用 PCA 降维后，的确让原始数据的部分信息丢失，从而让数据更简单，更有规律。但是，如果数据本来就是线性的，只是因为测量设备精度问题，测量方法原理问题，或者测量人员的马虎大意导致了数据的波动，那么使用PCA处理数据可以使建模更精准，因此 PCA 也广泛用于数据降噪。
# 当然，有时为了数据的处理或者可视化等目的，就必学做出某种取舍，还是没有免费的午餐定理，每一种算法是完美的，所有优势兼得的。接下来我们介绍下 scikit-learn 中的模型超参数，并换一个稍微正规点的数据集演练下（主要对比训练时间和准确率）。
# 
# n_components：需要保留成分的个数，默认 n_components = min(n_samples, n_features)，这时 n_components 为正整数。如果 n_components 为小数(0<n_components<1)，则算法自动确定保留成分个数以确保原数据不少于 n_components 的方差被保留下来百分比；
# 
# copy：默认为 TRUE，如果被设成 False，则被传入 fit() 方法的数据会被复写；
# 
# svd_solver：用于选择奇异值分解的方法，默认 svd_solver=’auto’；
# 
# tol：表示由 svd_solver == ‘arpack’ 计算奇异值的容差，默认tol=0.0；
# 
# random_state：用于播种随机种子；

# In[4]:


# 引入必要的包
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 加载 scikit-learn内部手写数字识别数据集
digit = datasets.load_digits()

X = digit.data
y = digit.target

# 训练集和测试集的分离
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)


# In[5]:


def plot_digits(digits):
    fig, axes = plt.subplots(2, 5, figsize=(10,10), 
                            subplot_kw={'xticks':[], 'yticks':[]},
                           gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(digits[i].reshape(8, 8), cmap='bone')
    plt.show()
    
# 可视化几个数字样例
example_digits = X[:10]
plot_digits(example_digits)


# In[6]:


get_ipython().run_cell_magic('time', '', '# 实例化一个 kNN 模型，并训练，记录训练时间\nknn_CLF = KNeighborsClassifier()\nknn_CLF.fit(X_train, y_train)')


# In[7]:


# 测试模型准确率
print("识别准确率：",knn_CLF.score(X_test, y_test))


# 然后我们对手写数字识别数据集使用 PCA 降维，然后创建模型进行训练。

# In[8]:


# 将原数据使用PCA从64维降到2维
pca = PCA(n_components=2)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)


# In[9]:


get_ipython().run_cell_magic('time', '', '# 再实例化一个kNN模型，并使用降维后的数据训练模型，记录训练时间\nknn_CLF2 = KNeighborsClassifier()\nknn_CLF2.fit(X_train_reduction, y_train)')


# In[10]:


# 测试模型准确率
print('识别准确率：',knn_CLF2.score(X_test_reduction, y_test))


# 从以上两个模型的对比可知，模型2通过 PCA 数据降维的确加快了模型的训练速度，但是也牺牲了模型的准确度。也可能是我们是我们降维降过头了，已知手写字符识别数据集是 8X8 的灰度图，也就是64维，我们直接只取2维确实有些过分。接下来我们试下超参数 n_components 的另外一种用法：

# In[11]:


# 实例化一个至少保留90%信息的PCA模型，并训练
pca2 = PCA(n_components=0.9)
pca2.fit(X_train, y_train)
# 打印降维后的维数
print("Number of components:", pca2.n_components_)


# In[12]:


pca3 = PCA(n_components=0.8)
pca3.fit(X_train, y_train)
print("Number of components:", pca3.n_components_)


# 由 n_components 所代表的超参数含义可知，以上两个实例化的 PCA 模型分别表示：当要保留90%的方差时，则至少要保留21维主成分的数据；当要保留80%的方差时，至少要保留13维主成分的数据。其实，在 scikit-learn 的 PCA 类中，还封存了一些比较逆天的方法—— explained_variance_，实例化一个 PCA 类后，直接调用它可以返回每一个成分对应可代表的方差的数量；explained_variance_ratio_ 可以直接返回各个方差所可表示的方差的百分比。例如，我们下面直接令 n_components=X.shape[1]，即保留数据集的所有维度，然后调用 explained_variance_ratio_ 方法就可以得到手写数字识别数据集变形后的主成分每一维可解释的方差的比率。可以看到，经过 PCA 计算之后的各个主成分所能够表示的方差所占的比率是降序的。

# In[13]:


pca4 = PCA(n_components=X.shape[1])
pca4.fit(X_train, y_train)
print('降序排列的各主成分所占方差比率：','\n',pca4.explained_variance_ratio_)


# 最后可视化下 scikit-learn 中手写数字识别数据集 PCA 降维后的二维分类结果来结束今天的分享。

# In[14]:


# 将数据降为2维便于可视化
pca5 = PCA(n_components=2)
pca5.fit(X)
X_reduction = pca5.transform(X)

for i in range(10):
    plt.scatter(X_reduction[y==i,0], X_reduction[y==i,1], alpha=0.8)
plt.show()

