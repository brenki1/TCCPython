from sklearnex import patch_sklearn
patch_sklearn()

import numpy as np
import time
from sklearn.model_selection import train_test_split

try:
    from cuml.svm import SVC as cuSVC
except ImportError:
    print ("Falha ao encontrar cuML, suporte a gpu desabilitado")
    from sklearn.svm import SVC as cuSVC

X = np.random.rand(20000,100)
y = np.random.randint(0,2,20000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

modelo = cuSVC(kernel='rbf', C=1)
inicio = time.time()
modelo.fit(X_train, y_train)
fim = time.time()
print(f"tempo do treino em {fim-inicio} segundos")

previsao = modelo.predict(X_test)
