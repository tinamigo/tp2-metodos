from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA
from time import time
import numpy as np
import pandas as pd
import metnum

#Recibe k, K (k para cantidad de vecinos en kNN y K para cantidad de folds del train data)
#Recibe PCA un bool para ver si aplica PCA antes de kNN o no, alpha importa solo si PCA = true
#Recibe trainpath un string para saber de donde leer los datos
#Recibe opcionalmente un numero para reducir la cantidad de imagenes de training
def tests_KFolds(k, K,has_PCA,alpha,trainpath,*args):  
    #Se toman los datos de training
    df_train = pd.read_csv(trainpath)
    #Se toma menos elementos si es necesario
    if (len(args)>0):
        if(df_train.shape[0] > args[0]):
            df_train = df_train.sample(args[0])
            
    X = df_train[df_train.columns[1:]].values
    y = df_train["label"].values.reshape(-1, 1)
           
            #Se aplica PCA si es necesario
    if has_PCA:             
        PCA_call = metnum.PCA(alpha)
        PCA_call.fit(X)
        X=PCA_call.transform(X)
        #X_train = PCA_call.transform(X_train)
    #Se separan
    kfold = KFold(n_splits=K)
    kfold.get_n_splits(X)  
    
    #Para cada fold se hara un entrenamiento, prediccion y se guardara el accuracity y el tiempo que dio cada uno
    Accuracities = [] 
    Times = []
    cm=np.zeros((10,10))
    Precision=[]
    Recall=[]
    F_Score=[]
    for train_index, test_index in kfold.split(X):
        #Se separan los datos de entrenamiendo y los de validacion de esta iteracion
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index] 
        
            
        #Se hace un clasificador        
        kNN = metnum.KNNClassifier(k)        
        kNNTime_start = time()
        
        #Se entrena con los datos de entrenamiendo      
        kNN.fit(X_train, y_train)

        #Se intenta predecir los datos de validacion        
        y_pred = kNN.predict(X_test)        
        kNNTime_end = time()
        kNNTime = kNNTime_end - kNNTime_start
        Times.append(kNNTime)
        
        #Se ve el accuracity de lo predicho         
        acc = accuracy_score(y_test, y_pred)
        Accuracities.append(acc) 
        cm = cm + confusion_matrix(y_test,y_pred,labels=[u for u in range(10)])     
        other_metrics=precision_recall_fscore_support(y_test,y_pred,labels=[u for u in range(10)])
        Precision.append(other_metrics[0])
        Recall.append(other_metrics[1])
        F_Score.append(other_metrics[2])

    Accuracities = np.asarray(Accuracities)
    Times = np.asarray(Times)
    return [np.mean(Accuracities),np.mean(Times),np.mean(Precision),np.mean(Recall),np.mean(F_Score),cm]