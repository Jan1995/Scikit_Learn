#代码分两部分，为方便放到一个py文件
#第一部分为knn的回归
#第二部分为skl库线性回归使用

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor #引入kNN回归类
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

boston = datasets.load_boston() # 直接加载scikit-learn 内部波斯顿房价数据集

X = boston.data
y = boston.target

X = X[y < 50]   # 为消除反常数据，便于拟合，直接抹除y大于50的值
y = y[y < 50]

# 分离训练测试数据集，种下随机种子
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

# 实例化一个kNN回归，并训练
knn_reg = KNeighborsRegressor()
knn_reg.fit(X_train, y_train)

# 检验下模型的 MAE，MSE，和 R squared error
y_predict = knn_reg.predict(X_test)

MAE = mean_absolute_error(y_predict, y_test)
MSE = mean_squared_error(y_predict, y_test)
r2_accuray = r2_score(y_predict, y_test)

print("MAE: %f, MSE: %f, R2 Accuracy: %f" % (MAE, MSE, r2_accuray))

# 检验下模型的准确率
print(knn_reg.score(X_test, y_test))

from sklearn.model_selection import GridSearchCV

# 定义要进行搜索的参数网格
param_grid = [
    {
        "weights": ["uniform"],
        "n_neighbors": [i for i in range(1, 11)]
    },
    {
        "weights": ["distance"],
        "n_neighbors" : [i for i in range(1, 11)],
        "p": [i for i in range(1, 6)]
    }
]

# 实例化一个kNN回归模型
knn_reg2 = KNeighborsRegressor()
# 实例化一个网格搜索模型，并开始训练、搜索
grid_search = GridSearchCV(knn_reg2, param_grid=param_grid, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 打印搜索到的最佳参数
print(grid_search.best_params_)


#第二部分
#skl库的现行回归
from sklearn.linear_model import LinearRegression

# 实例化一个线性回归，使用默认超参数，并训练
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# 打印模型准确度
print(lin_reg.score(X_test, y_test))

print(lin_reg.coef_)

boston.feature_names[np.argsort(lin_reg.coef_)]

