# 引入必要的包
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, LinearRegression

# 引入 scikit-learn 中波士顿房价数据集
boston = datasets.load_boston()

X = boston.data
y = boston.target

# 除去异常值点
X = X[y < 50.0]
y = y[y < 50.0]

# 训练数据集和测试数据集分离
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6)

# 实例化一个正则对象
standardScaler = StandardScaler()

# 拟合正则参数
standardScaler.fit(X_train)

# 正则化输入特征
X_train_std = standardScaler.transform(X_train)
X_test_std = standardScaler.transform(X_test)

# 实例化一个线性回归对象，训练模型并求其准确率
lin_reg = LinearRegression()
lin_reg.fit(X_train_std, y_train)
lin_reg.score(X_test_std, y_test)

sgd_reg = SGDRegressor(random_state=6)
sgd_reg.fit(X_train_std, y_train)
sgd_reg.score(X_test_std, y_test)

sgd_reg2 = SGDRegressor(n_iter=15, random_state=6)
sgd_reg2.fit(X_train_std, y_train)
sgd_reg2.score(X_test_std, y_test)

sgd_reg3 = SGDRegressor(n_iter=150, random_state=6)
sgd_reg3.fit(X_train_std, y_train)
sgd_reg3.score(X_test_std, y_test)