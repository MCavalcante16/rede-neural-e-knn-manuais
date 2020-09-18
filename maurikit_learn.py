import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix as confusion_matrix
from scipy.spatial import distance

class LogisticRegression_GRAD():
    def __init__(self):
        self._estimator_type = "classifier"
        pass
    
    def fit(self, X, y, epochs=30, learning_rate=0.02):
        #Custo
        self.custos = np.array([])
        
        #Bias
        bias = np.ones((X.shape[0], 1))
        X = np.hstack((bias,X))
        self.w = np.ones(X.shape[1])
        print(self.w)
            
        for i in range(0, epochs):
            #Y predito
            y_pred = np.array([])
            for j in range(0, X.shape[0]):
                y_pred = np.append(y_pred, 1 / (1 + np.exp(np.sum((-1) * np.transpose(self.w) * X[j]))))
            
            #Calculo do Somatório(ei * xi)
            exi = 0     
            for j in range(0, X.shape[0]):
                exi += (y[j] - y_pred[j]) * X[j] #Correção no gradiente
            exi_n = (exi/X.shape[0])
            
            #Atualização dos pesos
            self.w = self.w + (learning_rate * exi_n)
            
            #Calcula e salva custo
            custo = np.sum((-1) * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)) / X.shape[0] * 2
            self.custos = np.append(self.custos, custo)

        print("Coeficientes: " + str(self.w))
                          
    def predict(self, X):
        result = np.array([])
        y_pred = np.array([])
        for j in range(0, X.shape[0]):
            y_pred = np.append(y_pred, 1 / (1 + np.exp((-1) * self.w[0] + np.sum((-1) * np.transpose(self.w[1:]) * X[j]))))
                
        for i in y_pred:
            if i < 0.5:
                result = np.append(result, 0)
            else:
                result = np.append(result, 1)
                      
        return result

class Classification_NBG():
    def __init__(self):
        self._estimator_type = "classifier"
        pass
    
    def fit(self, X, y):
        self.classes_probabilidades = {}
        self.classes_medias = {}
        
        self.E = np.cov(X, rowvar=False)
        self.classes, classes_qtds =  np.unique(y, return_counts=True)
        
        for classe, classe_qtd in zip(self.classes, classes_qtds):
            indices = np.where(y == classe)
            self.classes_probabilidades[classe] = (classe_qtd / X.shape[0])
            x = np.array([X[index, :] for index in indices]).reshape(classe_qtd, X.shape[1])
            self.classes_medias[classe] = x.mean(axis=0)

    def predict(self, X):
        result = np.array([])
        for x in X:
            x_result = np.array([])
            for classe in self.classes:
                aux = 1 / np.sqrt(np.linalg.det(self.E)) * ((2*np.pi) ** (X.shape[1] / 2))
                prob_xc = aux * np.exp(-(1.0/2) * np.transpose(x - self.classes_medias[classe]) @ np.linalg.inv(self.E) @ (x - self.classes_medias[classe]))
                x_result = np.append(x_result, prob_xc * self.classes_probabilidades[classe])
            melhor_classe = self.classes[np.where(x_result == x_result.max())]
            result = np.append(result, melhor_classe) 
        return result
    
class Classification_QD():
    def __init__(self):
        self._estimator_type = "classifier"
        pass
    
    def fit(self, X, y):
        self.classes_probabilidades = {}
        self.classes_medias = {}
        self.Es = {}
        self.classes, classes_qtds =  np.unique(y, return_counts=True)
        
        for classe, classe_qtd in zip(self.classes, classes_qtds):
            indices = np.where(y == classe)
            self.classes_probabilidades[classe] = (classe_qtd / X.shape[0])
            x = np.array([X[index, :] for index in indices]).reshape(classe_qtd, X.shape[1])
            self.classes_medias[classe] = x.mean(axis=0)
            self.Es[classe] = np.cov(x, rowvar=False)

    def predict(self, X):
        result = np.array([])
        for x in X:
            x_result = np.array([])
            for classe in self.classes:
                aux = 1 / np.sqrt(np.linalg.det(self.Es[classe])) * ((2*np.pi) ** (X.shape[1] / 2))
                prob_xc = aux * np.exp(-(1.0/2) * np.transpose(x - self.classes_medias[classe]) @ np.linalg.inv(self.Es[classe]) @ (x - self.classes_medias[classe]))
                x_result = np.append(x_result, prob_xc * self.classes_probabilidades[classe])
            melhor_classe = self.classes[np.where(x_result == x_result.max())]
            result = np.append(result, melhor_classe) 
        return result

    
def acuracia(y_true, y_pred):
    qtdAcertos = 0
    for i in range(0, y_true.shape[0]):
        if y_true[i] == y_pred[i]:
            qtdAcertos += 1

    return qtdAcertos/y_true.shape[0]


def plot_confusion_matrix(classifier, X_test, y_test):
    class_names = np.unique(y_test)
    np.set_printoptions(precision=2)
    title = "Matriz de Confusão"
    disp = confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=None)
    disp.ax_.set_title(title);
    print(title)
    print(disp.confusion_matrix)
    plt.show()


def plot_boundaries(classifier, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('X1')
    plt.ylabel('X2')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()
    
def sigmoid(x):
    return 1/(1 + np.exp(-1 * x))

def sigmoid_derived(x):
    return sigmoid(x) * (1 - sigmoid(x))

class redeMLP():
    def __init__(self, input_layer_size=2, hidden_layer_size=4):
        self.e = {}
        self.w = np.ones((hidden_layer_size, input_layer_size + 1))
        self.m = np.ones((1, hidden_layer_size + 1))
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = 1
        self._estimator_type = "classifier"
        
    def fit(self, X, y, learning_rate=0.01, epochs=50):
        #Bias
        bias = np.full((X.shape[0], 1), -1)
        X = np.hstack((bias,X))
        
        for i in range(0, epochs):
            self.forward(X)
            self.e = y - self.y_pred
            
            for j in range(0, X.shape[0]):
                #Treina camada de Saída
                delta_k = self.e[j] * sigmoid_derived(np.sum(self.z * np.transpose(self.m)))
                self.m = self.m + learning_rate * delta_k * self.z
                
                #Treina camada oculta
                for wIndex in range(0, self.w.shape[0]):
                    delta_i = sigmoid_derived(np.sum(X[j] * self.w[wIndex, :])) * np.sum(delta_k * self.m[0,wIndex])
                    self.w[wIndex,:] = self.w[wIndex, :] + learning_rate * delta_i * X[j]
                    

    def forward(self, X):
        #Saídas
        amostra = 0
        self.z = np.ones((1,self.hidden_layer_size+1))
        self.z[0,0] = -1
        self.y_pred = np.array([])
        
        for x in X:
            this_y_pred = np.array([])
            for i in range(0, self.w.shape[0]):
                u = np.sum(x * self.w[i,:])
                self.z[0,i+1] = sigmoid(u)
                
            for i in range(0, self.m.shape[0]):
                u = np.sum(self.z * self.m)
                this_y_pred = np.append(this_y_pred, sigmoid(u))
            
            if this_y_pred > 0.5:
                this_y_pred = 1
            else: 
                this_y_pred = 0
                
            self.y_pred = np.append(self.y_pred, this_y_pred)
            amostra += 1
        
        return self.y_pred
                
    def predict(self, X):
        #Bias
        bias = np.full((X.shape[0], 1), -1)
        X = np.hstack((bias,X))
        
        #Saídas
        amostra = 0
        self.z = np.ones((1,self.hidden_layer_size+1))
        self.z[0,0] = -1
        self.y_pred = np.array([])
        
        for x in X:
            this_y_pred = np.array([])
            for i in range(0, self.w.shape[0]):
                u = np.sum(x * self.w[i,:])
                self.z[0,i+1] = sigmoid(u)
                
            for i in range(0, self.m.shape[0]):
                u = np.sum(self.z * self.m)
                this_y_pred = np.append(this_y_pred, sigmoid(u))
            
            if this_y_pred > 0.5:
                this_y_pred = 1
            else:
                this_y_pred = 0
            
            self.y_pred = np.append(self.y_pred, this_y_pred)
            amostra += 1
        
        return self.y_pred
            
        
class knn():
    def __init__(self, k=2):
        self._estimator_type = "classifier"
        self.k = k
        pass
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        result = np.array([])
        for x in X:
            #Calculo dos vizinhos mais proximos
            k_nearest_classes = np.full((1,self.k), -1)
            k_nearest_values = np.full((1,self.k), np.inf)
            for x_train_index in range(0, self.X_train.shape[0]):
                this_distance = distance.euclidean(x, self.X_train[x_train_index])
                this_classe = self.y_train[x_train_index]
                
                for k_nearest_index in range(0, self.k):
                    if this_distance < k_nearest_values[0,k_nearest_index] :
                        classe_anterior = k_nearest_classes[0, k_nearest_index]
                        distancia_anterior = k_nearest_values[0,k_nearest_index]
                        
                        k_nearest_classes[0,k_nearest_index] = this_classe
                        k_nearest_values[0,k_nearest_index] = this_distance
                        
                        this_classe = classe_anterior
                        this_distance = distancia_anterior
            
            
            #Calcula a classe de maior ocorrência
            maiorClasse = 0
            maiorQuantidade = 0
            for classe in np.unique(self.y_train):
                classeQtd = np.sum(k_nearest_classes == classe)
                if classeQtd > maiorQuantidade:
                    maiorQuantidade = classeQtd
                    maiorClasse = classe
                

            result = np.append(result, maiorClasse)
            
        return result;
                        
        
def k_fold (X, y, k, metodo):
    e = np.array([])
    porcentagem = 1/k
    qtdPorcentagem = int(X.shape[0] * porcentagem)
    
    for i in range(0,k):
        porcent_init = qtdPorcentagem * i
        porcent_end = qtdPorcentagem * (i+1)
        
        X_train = X[porcent_init:porcent_end, :]
        y_train = y[porcent_init:porcent_end]
        
        X_test = X[:porcent_init, :]
        X_test = np.vstack((X_test, X[porcent_end:, :]))
            
        y_test = y[0:porcent_init]
        y_test = np.append(y_test, y[porcent_end:])
        
        
        metodo.fit(X_train, y_train)
        y_pred = metodo.predict(X_test)
        
        e = np.append(e, acuracia(y_test, y_pred))
                      
    return print("Erros: ", e);
       
        
        
    
    
           
       
 
              
                
            
       
        



    