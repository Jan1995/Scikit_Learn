
# coding: utf-8

# # 使用 scikit-learn 玩转机器学习——集成学习

# ## 小引

# 集成学习是结合多个单一估计器的预测结果对给定问题给出预测的一种算法，集成学习相对于单一的估计器来说会有更好的泛化能力和鲁棒性，教科书式的定义的确会让人头昏脑涨，以下我们就来拿小华做作业来举个栗子。

# cheat![image.png](attachment:image.png)

# 小华是个学渣，每次做作业都要抱学霸 A 的大腿，学霸A也不介意让他看作业，暂且不管背后是不是有什么XX交易，反正每次作业被批改后发下来得分还算过得去。但小华并不满足于此，他不是一个一般的学渣，它是一个有追求的学渣，他还想拿更高的分数。于是某天之后，小华又召集了班里的其他4个学霸 B、C、D、E 为他提供答案。有了5名学霸作业答案的小明一开始曾不知所措，因为当各个学霸的答案不一致时他不知道该抄谁的，于是他想到一个少数服从多数的原则来确定最后答案，之后小华的作业的得分果然更进一步。一段时间后，小华又总结出一个经验：学霸 B 一直是班级第一、年级前十的存在，无论是考试，还是作业，他的正确率总是比其他一般的学霸更高，所以当5位学霸的作业题答案出现分歧时，应该多考虑下学霸 B 的答案。小华这个参考同学作业的栗子就体现出了集成学习的思想。下面我们依次看下几个典型的集成学习：

# ## Voting Classifier
# 
# Voting classifier 可能是思想最朴素的集成学习分类器了，它就是利用了上述小华同学想到的“少数服从多数的原则”或者是平均化多个分类器对于未知样本属于某个类别的概率的思想。下面我们用 SVM、逻辑回归、决策树和 kNN 来演示下该算法：

# 1、先引入一些必要的包和数据，并将数据可视化

# In[15]:


# 引入必要的包
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# 引入数据，设置标准差为 0.15, 设置随机种子
X, y = make_circles(n_samples=300, noise=0.15, factor=0.5, random_state=233)

# 展示数据
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()


# 2、分离训练集和测试集，实例化一个 KNN 模型，训练并打印其精度。

# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2333)

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
print("KNN Accuracy: ", knn_clf.score(X_test, y_test))


# 3、实例化一个逻辑回归模型，训练并打印模型精度。

# In[17]:


from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression()
log_clf.fit(X_train, y_train)
print("Logistic Regression Accuracy: ", log_clf.score(X_test, y_test))


# 4、实例化一个支持向量机模型，训练并打印模型精度。

# In[18]:


from sklearn.svm import SVC

svm_clf = SVC()
svm_clf.fit(X_train, y_train)
print("SVM Accuracy: ", svm_clf.score(X_test, y_test))


# 5、实例化一个决策树模型，训练并打印模型精度。

# In[19]:


from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
print("Decision Tree Accuracy: ", dt_clf.score(X_test, y_test))


# 6、传入上述的各个机器学习模型，实例化一个 VotingClassifier 模型，训练并打印模型精度。

# In[20]:


from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators=[
    ('knn', KNeighborsClassifier()),
    ('logistic', LogisticRegression()),
    ('SVM', SVC()),
    ('decision tree', DecisionTreeClassifier())
],
                              voting='hard'
                             )
voting_clf.fit(X_train, y_train)
print("Voting Classifier Accuracy: ", voting_clf.score(X_test, y_test))


# 在这一集成学习-- Voting Classifier 的例子中并没有取得比任一个单个分类器都更好的结果，与 SVM 和 KNN 算法相比，该集成学习算法的精度却下降了。原因之一就是我们在实例化上述 Voting Classifier 的过程中传入一个超参数 voting='hard'，其含义是严格遵循少数服从多数的原则。严格遵循这个原则会导致多数人的暴政，就像上面的小华做作业的例子里，对于一道很难的题目，只有学霸B做对了，其他学霸都错了的情况下，小华因为采用“少数服从多数”的原则也跟着错了。这种情况下，可以为不同水平的分类器赋权重，或者对所有参与分类的分类器对未知样本属于某类得出一个概率，然后所有概率相加求平均来确定种类。对于这种情况，Voting Classifier 类只需将超参数 voting='soft' 即可，但上例中的 KNN 算法在分类时并不产生概率，所以就不调参演示了。

# ## Random Forests（随机森林）

# 我们都知道森林是由树构成的（手动滑稽，QAQ），所以随机森林也不例外，随机森林里面的树叫做决策树。上次我们刚聊过决策树，相信小伙伴们还有些印象，决策树是由一系列节点构成的，每划分一个节点都要在所有的特征维度的每个特征可能取到的值上进行搜索，以取得信息熵的最小和，或最大的信息增益。随机森林里面的树的节点划分可能稍有些变化，随机森林算法的高明之处之一就是利用随机性，使得模型更鲁棒。假如森林中有 N 棵树，那么就随机取出 N 个训练数据集，对 N 棵树分别进行训练，通过统计每棵树的预测结果来得出随机森林的预测结果。

# random_forests![image.png](attachment:image.png)

# 因为随机森林的主要构件是决策树，所以随机森林的超参数很多与决策树相同。除此之外，有2个比较重要的超参数值得注意，一个是 bootstrap，取 true 和 false，表示在划分训练数据集时是否采用放回取样；另一个是 [oob_score](http://blog.sina.com.cn/s/blog_4c9dc2a10102vl24.html)，因为采用放回取样时，构建完整的随机森林之后会有大约 33% 的数据没有被取到过，所以当 oob_score 取 True 时，就不必再将数据集划分为训练集和测试集了，直接取未使用过的数据来验证模型的准确率。下面我们用代码演示下随机森林分类器：

# In[39]:


# 引入随机森林类
from sklearn.ensemble import RandomForestClassifier

# 实例化一个随机森林模型
rf_clf = RandomForestClassifier(
    n_estimators=500, # 确定森林的规模，500棵树
    max_depth=6,      #确定每棵树的深度
    bootstrap=True,   # 放回取样
    oob_score=True    # 使用 out of bag 的数据测试模型
)
# 训练模型，并打印精度
rf_clf.fit(X, y)
print("Random Forests Accuracy: ", rf_clf.oob_score_)


# ## Extremely Randomized Trees

# 随机森林的一大特点就是利用随机划分的数据集构建决策树，其实还有其他算法更是把“随机”二字心法发挥到更高水准，真是山外青山楼外楼。这就是 Extremely Randomized Trees 算法了，它不仅在构建数据子集时对样本的选择进行随机抽取，而且还会对样本的特征进行随机抽取（即在建树模型时，采用部分特征而不是全部特征进行训练）。换句话说，就是对于特征集 X，随机森林只是在行上随机，Extremely Randomized Trees是在行和列上都随机，下面我们调用演示下 scikit-learn 中的 Extremely Randomized Trees 的分类器：

# In[8]:


from sklearn.ensemble import ExtraTreesClassifier

et_clf = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=5,
    bootstrap=True,
    oob_score=True
)
et_clf.fit(X, y)
print("Extremely Randomized Trees Accuracy: ", et_clf.oob_score_)


# ## AdaBoost

# Boosting 是一族将弱学习器提升为强学习器的一种算法。这族算法的工作机制类似：首先是根据初始训练集训练出一个基学习器，然后根据基学习器的表现调整样本分布，使得让基学习器犯错的样本再对下一个学习器训练时得到更大的权重，使得下一个学习器提高其在使上一个分类器犯错的样本集中的表现；然而该学习器仍会犯错，我们就将该步骤反复进行，直到达到某个指标。
# 
# 我们继续来拿上面小华参考同学作业的情况来打比方。小华经过观察后发现，原来学霸 A、B、C、D、E 们都有错题本，尤其是学霸 B，他的错题本比牛津高阶词典还厚，错题本当然是用来收集学霸们各次模拟考试、平时作业的错题，在期末考试前一个月，学霸 B 会把原来的错题本上的错题重新做一遍，并把这次又做错的题目放到一个新的错题本上。隔几天后，学霸 B 会把新的错题本再做一遍，再次重新整理错题......就是这样，不断重复这个步骤，学霸 B 班级第一的地位经历大大小小无数次模拟考试而无人撼动。那么我们刚刚讲到的 Boosting 算法是不是跟学霸 B 的学习方法一模一样呢？！？！AdaBoost 正是将 Boosting 算法学习过程中学到的各个模型线性组合起来！

# AdaBoost![image.png](attachment:image.png)

# 下面我们来看下 scikit-learn 中 AdaBoost 分类器的调用：

# In[9]:


from sklearn.ensemble import AdaBoostClassifier

adab_clf = AdaBoostClassifier(DecisionTreeClassifier(),
                              n_estimators=500,
                              learning_rate=0.3
                             )
adab_clf.fit(X_train, y_train)
print("AdaBoost Accuracy: ", adab_clf.score(X_test, y_test))


# 下图是 scikit-learn 官网贴出的 [机器学习算法小抄](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)，如果你还是机器学习的算法小白，可以从 START 点开始，根据图示的步骤结合你的数据和需求来选择合适的算法。这是这个系列的最后一篇了，希望小伙伴们都学的开心。

# sklearn-cheat-sheet![image.png](attachment:image.png)
