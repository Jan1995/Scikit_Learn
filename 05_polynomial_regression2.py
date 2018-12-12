#!/usr/bin/env python
# coding: utf-8

# # 使用 scikit-learn 玩转机器学习——多项式回归

# ## 多项式回归

# 上次刚和小伙伴们学习过 PCA，PCA 主要用来降低数据特征空间的维度，以达到方便处理数据，减小计算开销，和数据降噪提高模型准确率的目的。而这节我们要一起看的多项式回归，它为了提高模型预测的准确率恰恰做了一件相反的事情。多项式回归的基本思想是：以线性回归为基础，拓展数据集特征空间的维度，且被拓展的特征空间维度上的数据是给定数据集相关项的多项式项。下面我们举个小栗子，来引入我们今天的主角。

# In[11]:


# 引入相关的包
import numpy as np
import matplotlib.pyplot as plt

# 手工制作数据
np.random.seed(666)
x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)

y = -0.5 * x**2 + x + np.random.normal(0, 1, size=100)

# 可视化数据
plt.scatter(x, y)
plt.show()


# In[2]:


from sklearn.linear_model import LinearRegression

# 实例化一个线性回归模型并训练
lin_reg = LinearRegression()
lin_reg.fit(X, y)

y_predict = lin_reg.predict(X)

# 可视化原始数据和预测数据
plt.scatter(x, y)
plt.plot(x, y_predict, color='r')
plt.show()


# In[3]:


# 模拟一个简单的多项式回归
X2 = np.hstack([X, X**2])

lin_reg2 = LinearRegression()
lin_reg2.fit(X2, y)
y_predict2 = lin_reg2.predict(X2)

# 可视化原始数据和预测数据
plt.scatter(x, y)
plt.plot(np.sort(x), y_predict2[np.argsort(x)], color='r')
plt.show()


# In[4]:


# 打印求得的跟样本特征空间相对应的系数，是不是跟产生数据时的 （-0.5，1）很接近
coef = lin_reg2.coef_
intercept = lin_reg2.intercept_
print("COEFFICIENT: ", coef, '\n', "INTERCEPT: ",intercept)


# 在上例中，我们给一个二次曲线的拟合数据加上一些噪音来产生一个数据集，然后实例化一个线性回归模型，去拟合出一条直线，结果可想而知，你用一个线性模型去拟合二次数据点准确率肯定不高。接着，我们在原始数据上手动添加了一维，且第二维数据是第一维数据的平方，然后我们再次实例化一个线性回归模型，这次拟合出了一条曲线，就没那么辣眼睛了吧。下面我们使用 scikit-learn 中包装好的多项式回归在试验下。

# ## scikit-learn 中的多项式回归与 Pipeline

# 使用多项式回归时，由于拓展的维度是已给定维度的多项式项，而多项式的高次操作可能会导致数据之间的量级差异加剧，所以，对已经进行过多项式操作的数据进行归一化操作也是十分必要的，然后我们再使用回归模型进行预测，以得到准确度更高的模型。为了简化这个过程，个人墙裂推荐使用 scikit-learn 中的 Pipeline 将这三个模型封装起来串联操作，让模型接口更加简洁，使用起来也更加的优雅。接下来是使用手工制作数据集使用 scikit-learn 中的内封模型进行的代码演示。

# In[5]:


#引入必要的包
import numpy as np
import matplotlib.pyplot as ply
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# 手工制作数据
np.random.seed(17)
x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
y = -0.5*x**2 + x + np.random.normal(0, 1, size=100)


# 为方便调用，使用 Pipeline 封装了一个多项式回归函数，函数 PolynomialRegression() 中传入的超参数 degree 是用来指定所得的多项式回归中所用多项式的阶次。

# In[6]:


# 制作一个多项式回归的 Pipeline
def PolynomialRegression(degree):
    return Pipeline([
    ('poly', PolynomialFeatures(degree=degree)),
    ('std', StandardScaler()),
    ('lin_reg', LinearRegression())
])

# 实例化一个多项式回归并进行训练和预测
poly_reg = PolynomialRegression(degree=2)
poly_reg.fit(X, y)
y_predict = poly_reg.predict(X)

# 绘制原始数据和预测数据
plt.scatter(x, y)
plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')
plt.show()


# In[7]:


# 为了产生过拟合的效果，令 degree=100, 有些夸张
poly_reg2 = PolynomialRegression(degree=100)
poly_reg2.fit(X, y)
y_predict2 = poly_reg2.predict(X)

plt.scatter(x, y)
plt.plot(np.sort(x), y_predict2[np.argsort(x)], color='r')
plt.axis([-3, 3, -10, 3])
plt.show()


# ## 模型泛化

# 一般情况下，我们并不知道我们所研究的数据的分布规律，所以说在使用多项式回归的时候也很难直接给出符合数据潜在规律的次幂，当我们指定的 degree 过低时，当然会直接从预测的准确率直接反映出来，但是当超参数 degree 过高时，就经常会呈现出上图所示的过拟合现象，这时，我们可以通过模型泛化来解决这个问题。
# 
# 常用的用于解决模型过拟合问题的泛化回归模型有岭回归和 LASSO 回归，这两种回归都是在原来回归模型的损失函数中添加上对应特征系数的相关项，岭回归使用了各个系数的平方和：

# ![image.png](attachment:image.png)

# 而 LASSO 回归是各项系数绝对值之和：

# ![image.png](attachment:image.png)

# alpha 是岭回归和 LASSO 回归正则项前一项重要的可调整的超参数，下图显示了 alpha 的变化对训练所得模型权重的影响（注意alpha在下图中的横轴上是逐渐变小的），可以看出超参数 alpha 对训练得到的模型参数的影响还是挺大的，随着 alpha 的增大，模型参数的模值被压缩到一个更小的范围之内。

# ![image.png](attachment:image.png)

# 那么接下来我们就从 scikit-learn 中引入岭回归和 LASSO 回归模型，并将其实例化，看看他们的模型泛化能力到底如何：

# In[13]:


# 引入必要的函数和类
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso

X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=6)

# 为方便调用，分别制做了岭回归和 LASSO 回归的Pipeline
def RidgeRegression(degree, alpha):
    return Pipeline([
    ('poly', PolynomialFeatures(degree=degree)),
    ('std', StandardScaler()),
    ('ridge', Ridge(alpha=alpha))
])
def LassoRegression(degree, alpha):
    return Pipeline([
    ('poly', PolynomialFeatures(degree=degree)),
    ('std', StandardScaler()),
    ('LASSO', Lasso(alpha=alpha))
])
# 为避免代码过度重复，将画图代码重构为一个函数
def plot_model(model):
    X_ = np.linspace(-3, 3, 100).reshape(100, 1)
    y_ = model.predict(X_)
    
    plt.scatter(x, y)
    plt.plot(X_[:,0], y_, color='r')
    plt.axis([-3, 3, -10, 3])
    plt.show()

#实例化一个岭回归，并进行训练，绘制图形
ridge_reg = RidgeRegression(degree=100, alpha=10)
ridge_reg.fit(X_train, y_train)
plot_model(ridge_reg)


# 上面我们已经训练了一个岭回归，并在图中绘出数据点及其拟合的曲线，下面我们在实例化一个LASSO回归模型，并绘出相应的数据点和曲线：

# In[10]:


lasso_reg = LassoRegression(degree=100, alpha=0.1)
lasso_reg.fit(X_train, y_train)
plot_model(lasso_reg)


# 同样是最高次幂为100的多项式回归模型，添加了 L1和L2 正则的岭回归和 LASSO 回归相比线性回归的曲线缓和了不少，极大程度的缓解了模型的过拟合，通过调整超参数 alpha，还能得到更准确地模型。
