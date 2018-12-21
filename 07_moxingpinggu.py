#!/usr/bin/env python
# coding: utf-8

# # 使用 scikit-learn 玩转机器学习——模型评价

# 蜉蝣扶幽

# ## 小引

# 对于分类模型来说，我们一般会用模型的准确率来进行模型的评价，模型的准确率是用预测正确的样本数除以模型的总数。如果一个模型的准确率达到了95%，那么在我们的印象中，是不是这个模型表现的还挺不错的，那如果达到了99%呢，岂不是更好。但是，在样本类别不平衡的情况下，仅仅使用模型的准确率并不能体现出模型的优劣。
# 
# 就拿微博抽奖来举个栗子，IG 夺冠时王思聪发微博称：点赞、转发本条庆祝 IG 夺冠的微博可以参与获奖者每人一万的抽奖。假设10000人参与了该活动，共抽取了10名幸运者。现在问题来了，这次抽奖也成功的吸引了你女票的注意，她也知道你在机器学习领域浸淫多年，于是就命令你去建一个机器学习模型来预测她拿奖的准确率，通过研究中奖用户的特征来以此保证她下次一定抽中奖，不然就跟你分手。你一听慌了，一宿没睡狂撸代码，第二天一大早就拿着自己的劳动成果去邀功请赏，宣称你的模型准确率能到达99%，你女友一听脸色顿时铁青......于是你成了单身狗，可怜的是你居然还不知道到底出了什么问题。

# ![image.png](attachment:image.png)

# 好了，段子讲完了，言归正传。你想想，10000 个人抽10个人，中奖率都 0.1%，那么最朴素的一个模型就是无论是谁，我都宣称他的中奖率为0.1%，就这，这样的模型的准确率都能达到99.9%，那么准确率为99%的模型简直不要太垃圾好吧！你说你不单身谁单身。这同时也说明了，单一的使用准确率来评价分类模型的好坏是不严谨的，那么接下来就进入我们今天的正题。

# ### 混淆矩阵

# ![image.png](attachment:image.png)

# 我们拿二分类问题来举个栗子，上图中行代表真实值，列代表预测值，0、1分别代表我们研究的2个种类。预测正确为 True，用 T 表示，预测错误为 False，用 F 表示，预测为0类，我们称其呈阴性，用 N 表示，预测为1类被称为阳性，用 P 表示。在上表中合起来就是 TN、TP、FN、FP这四个值。下表就是上述提到的微博抽奖的混淆矩阵的其中一种情况。

# ![image.png](attachment:image.png)

# 在上表中，实际上没中奖同时也预测正确的人数，即TN值为9978，实际上中奖了也预测正确人数，即TP值为8，没中奖且预测错误的人数，即FP值为12，中了奖但预测错误的人数，即FN值为2.

# ### 精准率与召回率

# 精准率是TP值与TP值和FP值的和的比值，在上例中表示预测对的中奖人数占按预测应该中奖的人数的比值，表示如下：

# $$Precision = TP / (TP + FP)$$

# 召回率是TP值与TP值和FN值的和的比值，在上例中表示预测对的中奖人数占实际中奖人数的比率，表示如下：

# $$Recall = TP / (TP + FN)$$

# 然后我们可以得到我们所据上述例子中的混淆矩阵：

# ![image.png](attachment:image.png)

# 根据精准率和召回率的定义可得，$precision = 0/0$ 出现除0情况而无意义，$recall = 0 / (10 + 0) =0$，召回率为0，根据召回率的定义也可知，召回率表示的是对于特定的目标群，预测正确的比率。完美的解决了准确率在偏斜数据中不作为的问题。

# ### F1 Score

# 在不同的应用场景下，我们通常会关注不同的指标，因为有些时候精准率更为重要，有些时候召回率更为重要。为了同时权衡这两个指标的重要性，就出现了 F1 Score，表达式如下：

# $$F1 score = 2 * PrecisionScore * RecallScore / (PrecisionScore + RecallScore)$$

# 由上式我们可以看出，F1 Score 其实就是精准率与召回率的调和平均值，因为召回率和精准率都大于0，由极限的性质可知，只有精准率和召回率都打的时候，F1 Score 才会比较大。

# ### ROC 曲线

# 说到 ROC 曲线（Receiver Operating Characteristic, 受试者工作特性曲线），就得从 TPR 和 FPR，其分别表示 被正确预测的目标类别占目标类别的比率，和被错误的预测为目标类表占非目标类别的比率。其分别对应的表格和表达式如下：

# ![image.png](attachment:image.png)

# $$TPR = Recall Score = TP / (TN + TP)$$

# ![image.png](attachment:image.png)

# $$FPR = FP / (FP + TN)$$

# ROC 曲线源于二战中用于敌机检测的雷达信号分析技术，后来才被引入机器学习领域。在进行机器学习模型的比较时，如果一个模型的 ROC 曲线被另一个模型的曲线完全包住，则可断言后者的性能优于前者；若两个模型的 ROC 曲线发生交叉，则在一般情况下很难判定2个模型孰优孰劣，这时，一种较为合理的评比标准便是比较这两个 ROC 曲线之下的面积，即 AUC（Area under curve）。

# ## 代码演练

# 接下来我们用代码来具体的实现下相关的评判标准和判别式。
# 
# 引入必要的包 -> 调用数据集 -> 使数据集中不同类别数量偏斜 -> 分离训练、测试数据集 -> 实例化一个逻辑回归模型 -> 预测并求出模型准确率

# In[1]:


# 引入必要的包
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 使用库内自带的手写数字集
digits = load_digits()
X = digits.data
y = digits.target.copy()

# 故意致使数据集类别的不平衡
y[digits.target==5]=1
y[digits.target!=5]=0

# 分离训练和测试数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=233)
# 实例化一个逻辑回归并训练
logistic_clf = LogisticRegression()
logistic_clf.fit(X_train, y_train)

accuracy = logistic_clf.score(X_test, y_test)
y_predict = logistic_clf.predict(X_test)
print("ACCURACY: ", accuracy)


# 为增加我们对上述有关术语和评判标准的感性认识，我们具体实现了下一些函数，如下：

# In[2]:


import numpy as np

def TN(y_true, y_predict):
    assert len(y_true)==len(y_predict)
    "the vectors y_true and y_predict must have the same length."
    return sum((y_true==0) & (y_predict==0))

def TP(y_true, y_predict):
    assert len(y_true)==len(y_predict)
    "the vectors y_true and y_predict must have the same length."
    return sum((y_true==1) & (y_predict==1))

def FN(y_true, y_predict):
    assert len(y_true)==len(y_predict)
    "the vectors y_true and y_predict must have the same length."
    return sum((y_true==1) & (y_predict==0))

def FP(y_true, y_predict):
    assert len(y_true)==len(y_predict)
    "the vectors y_true and y_predict must have the same length."
    return sum((y_true==0) & (y_predict==1))

# 计算混淆矩阵
def confusion_matrix(y_true, y_predict):
    assert len(y_true)==len(y_predict)
    "the vectors y_true and y_predict must have the same length."
    return np.array([
        [TN(y_true, y_predict), FP(y_true, y_predict)],
        [FN(y_true, y_predict), TP(y_true, y_predict)]
    ])

# 计算精准率
def precision_score(y_true, y_test):
    assert len(y_true)==len(y_predict)
    "the vectors y_true and y_predict must have the same length."
    tp = TP(y_true, y_predict)
    fp = FP(y_true, y_predict)
    try:
        return tp / (tp + fp)
    except:
        return 0
    
# 计算召回率
def recall_score(y_true, y_test):
    assert len(y_true)==len(y_predict)
    "the vectors y_true and y_predict must have the same length."
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0
    
# 计算 F1 Score
def f1_score(y_true, y_predict):
    assert len(y_true)==len(y_predict)
    "the vectors y_true and y_predict must have the same length."
    recall = recall_score(y_true, y_test)
    precision = precision_score(y_true, y_test)
    return 2*recall*precision/(precision+recall)


# 当然了，如果每次使用精准率和召回率时都要自己亲手撸出来可能骚微还是有一些的麻烦，不过 贴心的 scikit-learn 找就为我们准备好了一切，在 metrics 中封装了所有我们在上述实现的度量，如下是调用演示：

# In[3]:


# 引入必要的包
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import f1_score, recall_score

confusion_mat = confusion_matrix(y_test, y_predict)

precision = precision_score(y_test, y_predict)

recall = recall_score(y_test, y_predict)

F1_score = f1_score(y_test, y_predict)

print("混淆矩阵：",'\n',confusion_mat)
print("精准率：", precision)
print("召回率：", recall)


# 对于机器学习模型的性能而言，不光是各样本的特征系数，而且阈值（或称之为截距）的取法对其也有着重要的影响。如下代码是用于绘制精准率与召回率和阈值取值的关系，并绘出其图形：

# In[6]:


import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

decision_scores = logistic_clf.decision_function(X_test)
precisions, recalls, thresholds = precision_recall_curve(y_test,decision_scores)

plt.plot(thresholds, recalls[: -1])
plt.plot(thresholds, precisions[: -1])
plt.show()


# PR 曲线对研究机器学习模型也有着重要的作用，我们也可以从 scikit-learn 中调用相关的函数来绘制 PR 曲线，如下：

# In[7]:


plt.plot(precisions, recalls)
plt.show()


# 绘制出 ROC 曲线：

# In[8]:


from sklearn.metrics import roc_curve, roc_auc_score

fprs, tprs, thresholds = roc_curve(y_test, decision_scores)
plt.plot(fprs, tprs)
plt.show()


# ROC 曲线和 PR 曲线有着很强的相似性，因为这两图的各自的两个指标的取值范围都是0到1，因此都可以用曲线与 y=0 围成的面积可以用来表征模型的优劣，且用面积作为指标来衡量模型优劣对指标某个部分的具体变化不敏感，稳定性更强。关于以上所有概念更为严谨和全面的定义和证明请参考周大佬的西瓜书。

