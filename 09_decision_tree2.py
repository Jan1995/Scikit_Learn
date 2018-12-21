
# coding: utf-8

# # 使用 scikit-learn 玩转机器学习——决策树

# 蜉蝣扶幽

# ## 小引

# 决策树算法是计算机科学家罗斯.昆兰（下图大佬，没错，是图灵，因为我没找到昆兰大佬的照片）在学术休假时提出的。期间，他到斯坦福大学访问，选修了图灵的助手 D.Michie 开设的一门研究生课程。课上布置的一个大作业就是用程序写出一个完备正确的规则，以判定国际象棋的残局是否会在2步后被将死，昆兰在这个任务中得到灵感，之后又将该部分工作整理出来于1979年发表，并命名为 ID3 算法。之后很多其他的决策树算法也相继问世，比如ID4、ID5、C4.5、和 CART（Classification and Regression Tree） 等。scikit-learn 中决策树的实现是基于 CART。

# ![turing.png](attachment:image.png)

# 决策树是一类常见的机器学习方法。它把分类和回归问题归结为做出一系列子决策，通过一系列子决策组合得到的结果来做出最终决策。当使用 CART 解决分类问题时，会使用待预测样本所在的叶子节点所有的数据进行投票，来决定未知样本的类别；当使用 CART 解决回归问题时，会使用待预测样本所在的叶子节点所有的样本输出的平均值，来表示未知样本的输出值。下面我们举个栗子。

# aquaman.png![image.png](attachment:image.png)

# 最近一部叫做《海王》的电影很热，小詹也打算去看，在去看之前，小詹做出了如下的决策。1）是好莱坞大片吗？当然是，DC 巨制；2）导演是谁？水不水？温子仁，拍过电锯惊魂等恐怖片为代表的佳作；3）投资1.6亿美元，据说光看特效就值了。考虑之后，小詹兴高采烈的买了2张晚上7点半的票。

# dt_workflow.png![image.png](attachment:image.png)

# 决策树在使用数据训练的过程中会建立一棵树，使用这棵树来预测未知样本的类别或回归值。在构建决策树时，我们会遍历数据的每一维特征，并在每一位特征上进行插值，以搜索最大信息增益或最小的子区间的信息熵之和。这涉及到信息熵和基尼系数的概念。

# 我第一次接触到熵的概念是高中化学（学霸勿喷），它用来表示物质的混乱度。这里的信息熵用来代表随机变量不确定度的度量，其表达式为：$$H = -\sum_{i=1}^n p_i log(p_i)$$
# 
# 基尼系数与信息熵类似，可以起到大概相同的作用。scikit-learn 中默认使用基尼系数进行计算，因为基尼系数的计算是多项式运算，比熵计算更快，大多数情况下区别不明显，基尼系数表达式如下：$$ G = 1 - \sum_{i=1}^n p_i^2 $$

# ## 代码演练

# 1、我们先加载一个鸢尾花数据集，并实例化一棵朴素的决策树分类器，绘出该决策树的决策边界，看看是什么样子。

# In[5]:


# 引入必要的包
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

# 加载鸢尾花数据集
iris = datasets.load_iris()
# 为便于可视化，只取两个特征
X = iris.data[:, 2:]
y = iris.target

# 实例化一颗决策树分类器
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X, y)

# 绘制决策边界的函数
def plot_decision_boundary(model, axis):
    
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3]-axis[2])*100)).reshape(-1, 1)
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)


# 下面我们绘制出刚才实例化并训练过的决策树模型的决策边界，和鸢尾花数据集样本点。

# In[6]:


plot_decision_boundary(dt_clf, axis=[0.5, 7.5, 0, 3])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.scatter(X[y==2, 0], X[y==2, 1])
plt.show()


# 2、接下来介绍下决策树的一些重要的超参数。
# 
# >criterion: 字符串，可选‘gini’或者‘entropy’，分别表示要使用基尼系数或熵进行决策区间的划分，默认选‘gini’；
# 
# >max_depth: 整型型数字，用来规定决策树的最大深度；
# 
# >min_samples_split: 可以使整型或浮点型数字，用来规定如果进行一次决策区间的划分至少要包含多少个样本；
# 
# >min_samples_leaf: 可以使整型或浮点型数字，用来规定每个叶子节点至少要包含多少个样本；
# 
# >max_features: 在寻找最佳划分时，最多考虑多少样本特征；
# 
# >min_impurity_decrese: 浮点数，设定了一个阈值，只有一次划分使得不纯度的减少量超过该阈值，该划分才会被允许。

# 3、给小伙伴们介绍一个很方便的 Python 模块 —— graphviz。我们可以先在 scikit-learn 中的 tree 的 export_graphviz() 函数中传入必要的信息来实例化一个图例，将图例传给 graphviz 的 source() 函数即可绘制出你训练过的决策树的结构。如下是刚才实例化的朴素决策树的结构图：

# iris_default.png![image.png](attachment:image.png)

# 鸢尾花数据集是一个著名的数据集，它含有4个特征分别是花萼长度（sepal length）、花萼宽度（sepal width）、花瓣长度（petal length）和花瓣宽度（petal length），上述决策树在生成时使用了鸢尾花数据集的全部特征。根据这些特征，鸢尾花最终分为3类，山鸢尾（iris Setosa）、杂色鸢尾（iris Versicolour）、和维吉尼亚鸢尾（iris Virginica）。然后我们再去看上述的决策树结构图，从根节点开始，第一个最优划分特征是花瓣长度（petal length），最优划分值为2.45，此时的基尼系数为0.667，共150个样本，第一次划分后所得的左分支节点全是山鸢尾，共50个样本，基尼系数为0，停止继续划分；所得的右分支节点基尼系数不为0，需要继续划分，第二次最佳划分属性为花瓣宽度，最佳划分值为1.75......

# 4、上面的决策树似乎有些过拟合了，因为是默认模型，我们可以传入一些超参数给决策树模型剪枝，以此防止模型的过拟合，具体如下：

# In[7]:


dt_clf2 = DecisionTreeClassifier(criterion='gini',
                                 max_depth=2,
                                 min_samples_split=6
                                )
dt_clf2.fit(X, y)
plot_decision_boundary(dt_clf2, axis=[0.5, 7.5, 0, 3])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.scatter(X[y==2, 0], X[y==2, 1])
plt.show()


# 如上图所示，经过传参剪枝的决策树模型的决策边界好像是简洁多了，不过过度的剪枝会导致决策树模型的欠拟合，具体要看模型在训练集和测试集上的精度来调参，不再赘述。该模型对应的决策树结构如下：

# iris_pruning.png![image.png](attachment:image.png)

# 经过剪枝之后的决策树结构也变的十分简洁，篇幅好像够了，那这次分享就到这里，再见！
