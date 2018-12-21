
# coding: utf-8

# # 使用 scikit-learn 玩转机器学习——支持向量机

# 蜉蝣扶幽

# ## 小引

# 支持向量机（SVM）是监督学习中最有影响的方法之一。它的大致思想是找出距离两个类别（暂时以二分类问题为例）最近的点作为支持向量，然后找出一个最佳决策边界，以使从决策边界到支持向量的距离最大化。因为对于一个二分类问题来说，往往有无数个决策边界可以将两类数据分开，但我们只能选择一条作为我们的决策边界。

# ![image.png](attachment:image.png)

# 继续对上述问题进行讨论，SVM 最终还是转化为一个最值优化问题，它认为这样找的决策边界能够使两类事物区分的最好，将来对于未知种类的样本，它能够给出最正确的样本分类，即有着最好的泛化能力。用大白话翻译过来就是：苹果是苹果，梨就是梨，上帝在造苹果和梨的时候就在它们中间画了一条线，线的这边就是苹果，线的那边就是梨，我们要做的就是不断地逼近上帝画的那条线，这样能够更好地把梨和苹果分开。

# ![image.png](attachment:image.png)

# 上述讨论的问题是线性可分的，在 SVM 中对应着 hard margin 来解决，从名字可以看出来似乎还对应着 soft margin。的确，soft margin 的确存在，而且就像 softmax（不是强硬的直接输出最后分类结果0和1，而是给出属于对应结果的概率）和 softplus（softplus正是 ReLu 的圆角版）一样包含着缓冲和协调的作用。soft margin 引入了容错空间的的概念，从而允许原本属于不同类别的空间交叉重叠。

# $$min \frac{1}{2}||w||^2$$

# $$s.t. y^{(i)}(w^{T}x^{(i)} + b) >= 1$$

# 上述公式对应的是 hard margin 的损失函数和约束条件，w 表示各个特征的权重向量，在一个二分类问题中，标签值y取+1和-1，$(w^{T}x^{(i)} + b) = 0$ 表示我们求得的决策边界，$ (w^{T}x^{(i)} + b) >= 1$表示经学习后分得的正类，$ (w^{T}x^{(i)} + b) <= -1$表示经学习后分得的负类，$-1 < (w^{T}x^{(i)} + b) < 1$表示的应该是经过支持向量且与决策边界平行的区域，在 hard margin 情形下，该区域是没有任何点的。又因为标签值 y 取值为+1和-1，则正负类可以用一个不等式表示，然后就可以用拉格朗日乘子法等来解决这类约束优化问题。

# SVM 中另一个经常会出现的概念恐怕就是核了。通过核技巧，可以避免大量的点积运算，是计算更加高效，它同时保证了有效收敛的凸优化技术来学习线性模型。一般常用的核有高斯核（又叫做 RBF 核，radical basis function 的缩写）和多项式核（假装常用），高斯核函数如下所示：$$K(x, y) = e^{-\gamma||x-y||^{2}}$$

# ## 代码演练（分类大作战）

# 我们会先实例化一个朴素的 SVM 分类器（不调任何超参数，全部取默认参数），看看其表现如何，然后会跟小伙伴们介绍下一些重要的超参数，并试着调参来优化 SVM 分类器的性能，顺便跟我们以往介绍过的分类器做下比较。
# 
# 1、实例化一个朴素的 SVM 分类器，并看下其准确率

# In[1]:


from sklearn import datasets
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

X = digits.data
y = digits.target


# show一下数据集的几个数字样例：

# In[2]:


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


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2333)

svc = SVC()
svc.fit(X_train, y_train)

print("Accuracy: ", svc.score(X_test, y_test))


# 2、介绍下 SVM 中一些重要的超参数(包含linear_svc 和 SVC 两个模型的部分超参数)

# >penalty: 字符串，可选‘l1’或’l2‘，默认’l2‘，指定模型的正则方式；
# 
# >loss: 字符串，可选’hinge‘或’squared_hinge‘，默认’squared_hinge‘，用于指定模型的损失函数；
# 
# >kenel: 字符串，可选‘linear’,'poly','rbf','sigmoid','precomputed';
# 
# >degree: 整型数字，当使用多项式核时，用来确定多项式的阶次；
# 
# >dual: 布尔值，默认值为’True‘，选择算法以解决双优化或原始优化问题；
# 
# >tol: 浮点数，默认为 1e-4，用于判断是否停止迭代的容差；
# 
# >C: 浮点数， 默认为1.0，容错空间系数，用于调整容错空间在优化迭代中所占的重要性；
# 
# >multi_class: 字符串，可选’ovr‘和’crammer_singer‘，但面临多分类问题时，用于确定多分类策略，’ovr‘指定了使用 One VS Rest 策略进行多分类任务，而’crammer_singer‘则是在所有的类上建立一个联合的目标损失函数进行优化；
# 
# >verbose: 整型数字，默认值为0，若为大于0的整数，则会在训练过程中不断输入与训练相关的条件与参数；
# 
# >max_iter: 整型数字，默认值为1000，用于指定迭代的最大次数。

# 3、通过给 SVM 分类器调参，可以获得性能不错的分类器，如下：

# In[4]:


get_ipython().run_cell_magic('time', '', 'svc2 = SVC(C=10000, kernel=\'rbf\', gamma=0.001)\nsvc2.fit(X_train, y_train)\nprint("Accuracy: ", svc2.score(X_test, y_test))')


# 4、看看其他的分类器都有什么样的表现呢

# In[5]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import GridSearchCV\n\nparam_grid = [\n    {\n        \'weights\': [\'uniform\'],\n        \'n_neighbors\': [i for i in range(3, 11)],\n        \'p\': [i for i in range(1, 4)]\n    },\n    {\n        \'weights\': [\'distance\'],\n        \'n_neighbors\': [i for i in range(3, 11)],\n        \'p\': [i for i in range(1, 4)]\n    }\n]\nknn_gs_clf = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, n_jobs=-1)\nknn_gs_clf.fit(X_train, y_train)\nprint("Accuracy: ", knn_gs_clf.score(X_test, y_test))')


# 经过调参，都能达到很高的精度啦，但同样是达到99.11%的准确度，SVM 用了118ms，KNN 用了13.4s，当然了，这跟KNN 模型使用了网格搜索寻找部分最佳超参数也有关系了，再看看其他分类器的表现咯！

# In[7]:


get_ipython().run_cell_magic('time', '', 'log_clf = LogisticRegression()\nlog_clf.fit(X_train, y_train)\nprint("Accuracy: ", log_clf.score(X_test, y_test))')


# 喔！逻辑回归这老哥简直不要太给力，仅使用朴素模型准确度就能达到95.6%，没必要再调参了。

# 那再用一个单层的神经网络模型试试（训练50个EPOCH，输入层128个神经元，输出层10个神经元）：

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# 结果好像还不错呢，训练集和测试集上都有着98%的精度。那么这次的分享就到这里了，小伙伴们下次再见！！！
