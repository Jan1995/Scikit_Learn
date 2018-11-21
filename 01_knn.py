import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets #sklearn即scikit-learn库
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris() # 方便起见，直接使用sklearn中内置的鸢尾花数据集
X = iris.data[:,:2]   # 为方便可视化，仅取2个特征
y = iris.target

# 展示下数据集中的数据分布
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.scatter(X[y==2,0], X[y==2,1])
plt.show()


# 为了检测模型的准确率，防止模型在训练集中过拟合，将数据集随机分为训练数据集和测试数据集
X_train, X_test, y_train, y_test =train_test_split(X,# 样本值
                       y,#样本对应标签
                       random_state=666#为每次
                       #运行都得到相同的结果，种了颗随机种子
                      )

#实例化一个kNN模型
knn_clf = KNeighborsClassifier()
# 将KNN模型在训练数据集上进行训练
knn_clf.fit(X_train, y_train)
#在测试数据集上检测下模型的准确度
accuracy = knn_clf.score(X_test, y_test)
print("Accuracy: ",accuracy)

# 再实例化一个kNN模型
knn_clf2 = KNeighborsClassifier(n_neighbors=6, weights='distance',p=2)
# 将该KNN模型在训练数据集上进行训练
knn_clf2.fit(X_train, y_train)
#在测试数据集上检测下模型的准确度
accuracy = knn_clf2.score(X_test, y_test)
# 打印准确率
print("Accuracy: ",accuracy)

best_k = -1
best_p = -1
best_accuracy = 0

for k in range(3, 10):
    for p in range(1, 11):
        # 实例化一个kNN模型, 为加快运算速度使n_jobs=-1(使用CPU所有核运算)
        knn_clf2 = KNeighborsClassifier(n_neighbors=k, weights='distance',p=p,n_jobs=-1)
        # 将该KNN模型在训练数据集上进行训练
        knn_clf2.fit(X_train, y_train)
        #在测试数据集上检测下模型的准确度
        accuracy = knn_clf2.score(X_test, y_test)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            best_p = p
            
# 打印最佳参数值
print("最佳k值: %d, 最佳p值 : %d, 最高准确度: %f" %(best_k, best_p, best_accuracy))
