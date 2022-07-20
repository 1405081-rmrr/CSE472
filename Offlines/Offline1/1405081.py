import math as math
import pandas as pd
import numpy as np
import copy as copier
import operator
import collections
import random
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from collections import Counter
matrix=[]
class Preprocessor:
    def __init__(self, fileReader):
        self.fileReader = fileReader
        self.dataset = self.fileReader.readFromCSV()
        self.dataframe=self.dataset
        self.datamatrix=np.array(self.dataset)
        self.datamatrix=np.transpose(self.datamatrix).tolist()
        #print(self.dataframe.describe())
        #print("Length of dataset ",len(self.dataset))
        self.drop_column()
        self.splitDataSet()
        self.calculation()
        #for column in self.dataframe.columns:
            #print("Number of unique value ",column," ",len(pd.unique(self.dataframe[column])))
        #self.count=self.dataframe['Churn'].value_counts()
        #print(self.count)
        #self.splitDataSet()
        #print(self.dataframe.info())
        #print(self.dataframe.loc[:,'Churn'])
        
    def drop_column(self):
        #print(self.dataframe.head(10))
        #self.uniquevalue = len(pd.unique(self.dataframe['Churn']))
        #print("Number of churns : ",self.uniquevalue)
        self.dataset_length=self.dataframe.shape[0]
        for column in list(self.dataframe.columns):
            self.col_length=len(pd.unique(self.dataframe[column]))
            if(self.col_length>=self.dataset_length/2):
                self.dataframe.drop([column],axis=1,inplace=True)
            elif(self.col_length==1):
                self.dataframe.drop([column],axis=1,inplace=True)
        self.drop_row()
        #self.categorical()
        #print(self.dataframe)
        
        #for column in list(self.dataframe.columns):
            #self.uniquevalue=len(pd.unique(self.dataframe[column]))
            #print(column," ",self.uniquevalue," ",self.dataframe[column].dtypes)
        #print(self.dataframe.isnull().sum())
    
    def drop_row(self):
        for column in list(self.dataframe.columns):
            muhaha=self.dataframe[column].tolist()
        for x in muhaha:
            if(x=='?' or x==' '):
                x=np.nan
        self.categorical()
    
    def categorical(self):
        self.dataset_length=self.dataframe.shape[0]
        for column in self.dataframe.columns:
            if(self.dataframe[column].dtypes=='int64'):
                self.dataframe[column]=self.dataframe[column].astype(float)
            self.col_length=len(pd.unique(self.dataframe[column]))
            if(self.col_length==2):
                value_list=pd.unique(self.dataframe[column])
                #print(value_list)
                self.dataframe.loc[self.dataframe[column] ==value_list[0], column] = 0
                self.dataframe.loc[self.dataframe[column] ==value_list[1], column] = 1
                self.dataframe[column]=self.dataframe[column].astype(float)
            elif(self.col_length<self.dataset_length/2 and self.dataframe[column].dtypes=='object' ):
                value_list=pd.unique(self.dataframe[column])
                for x in value_list:
                    self.data_half=self.dataset_length//2
                    self.dataframe.loc[self.dataframe[column] ==x, column] = random.randrange(1.0,len(value_list))
                self.dataframe[column]=self.dataframe[column].astype(float)
            elif(self.col_length<self.dataset_length/2 and self.dataframe[column].dtypes=='float64' ):
                self.dataframe[column] = (self.dataframe[column] - self.dataframe[column].min()) / (self.dataframe[column].max() - self.dataframe[column].min())    
                
            #print("Number of unique value ",column," ",len(pd.unique(self.dataframe[column])))
            #print(column," ",self.dataframe.info())
        #print(self.dataframe.info())
        #print(np.array(self.dataframe))
        #print(self.dataframe)
        
        print(self.dataframe)
        self.datamatrix=(np.array(self.dataframe))
        row, col = self.datamatrix.shape
        self.datamatrix_with_label=self.datamatrix[:,-1]
        self.actual_class=self.datamatrix_with_label
        self.datamatrix_no_label = np.delete(self.datamatrix, -1, axis=1)
        row, col = np.transpose(self.datamatrix_no_label).shape
        #print(row," ",col)
        #print(self.datamatrix_no_label)
        #print(self.datamatrix_with_label)
        self.datamatrix_transpose=np.transpose(self.datamatrix)      
        #print(self.datamatrix_transpose)
        #print(len(self.datamatrix_transpose))
        #print(self.dataframe.iloc[:,17])
        #print(self.datamatrix)
        row, col = self.datamatrix_no_label.shape
        self.w=np.ones((col+1))
    def splitDataSet(self):
        self.datamatrix=np.array(self.dataframe)
        #trans = np.transpose(self.datamatrix).tolist()
        #print(trans)
        train, test = train_test_split(self.dataset, test_size=0.2)
        self.test = test
        self.train = train
        self.train_matrix=np.array(self.train)
        self.test_matrix=np.array(self.test)
        self.train_matrix_x=np.delete(self.train_matrix, -1, axis=1)
        self.train_matrix_y=self.train_matrix[:,-1]
        self.test_matrix_x=np.delete(self.test_matrix, -1, axis=1)
        self.test_matrix_y=self.test_matrix[0:,-1]
        print("Value Count ",collections.Counter(self.train_matrix_y))
        print("Value Count ",collections.Counter(self.test_matrix_y))
        print('\n')
        #print("TestX ",self.test_matrix_x)
        #print("TestY ",self.test_matrix_y)
        #print("TrainX ",self.train_matrix_x)
        #print("TrainY ",self.train_matrix_y)
        #print("Train ",self.train_matrix)
        #print('\n')
        #print("Test ",self.test_matrix)
        #print("length Train ",len(self.train))
        #print("Length test ",len(self.test))
        
        
        
    def tanh(self,z):
        x=(np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
        #print("tanh ",x)
        return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
        #return 1 / (1 + np.exp(-z))
    def hx(self,w,x):
        row, col = self.datamatrix_no_label.shape
        z=w[0]
        result_z=np.zeros((col))
        for i in range(0,col):
            z=z+w[i+1]*x[:,i]
        result_z=np.array(z)
        #norm = np.linalg.norm(result_z)
        #result_z = result_z/norm
        #print("Result ",result_z)
        return self.tanh(result_z)
        #print(result_z)
        #print(self.tanh(result_z))
        
    def cost(self,w, X, Y):
        #w = [element * (-1) for element in w]
        #print("In cost ",w)
        self.y_pred = self.hx(w,X)
        #print(type(self.y_pred))
        #print("Predicted Y value ",self.y_pred)
        #print("Predicted Y From cost Function ",self.y_pred)
        return -1 * sum(Y*np.log(self.y_pred+1.0001) + (1-Y)*np.log(1-self.y_pred+1.0001))
    def grad(self,w, X, Y):
        self.y_pred = self.hx(w,X)
        row, col = self.datamatrix_no_label.shape
        g = list()
        g.append( (-1*sum(Y*(1-self.y_pred)-(1-Y)*self.y_pred)))
        for i in range(0,col):
            g.append(-1 * sum(Y*(1-self.y_pred)*X[:,i] - (1-Y)*self.y_pred*X[:,i]))
        return g
    
    def descent(self,w_prev, lr):
        #print("W Prev ",w_prev)
        #print("Cost Prev \n",self.cost(w_prev, self.datamatrix_no_label, self.datamatrix_with_label))
        j=0
        dummy=0
        row, col = self.datamatrix_no_label.shape
        w_new_list=list()
        w_new_list1=list()
        w_new=np.ones(col+1)
        w_new1=np.ones(col+1)
        while True:
            w_prev = w_new
            w_prev1=w_new1
            w_new_list.clear()
            w_new_list1.clear()
            for i in range(0,col+1):
                dummy=w_prev[i]-lr*self.grad(w_prev,self.train_matrix_x,self.train_matrix_y)[i]
                dummy1=w_prev1[i]-lr*self.grad(w_prev,self.test_matrix_x,self.test_matrix_y)[i]
                w_new_list.append(dummy)
                w_new_list1.append(dummy1)
            w_new=np.array(w_new_list)
            w_new1=np.array(w_new_list1)
            #print("Length of wnew ",len(w_new))
            #print("W-new after ",j,"th iteration ",w_new)
            w_new_first=w_new[0]
            w_new_first1=w_new1[0]
            w_new_t=np.delete(w_new,0)
            w_new_t1=np.delete(w_new1,0)
            #print("W-new after deleting w0 ",w_new_t)
            w_new_2d=np.reshape(w_new_t,(col,1))
            w_new_2d_transpose=np.transpose(w_new_2d)
            w_new_2d1=np.reshape(w_new_t1,(col,1))
            w_new_2d_transpose1=np.transpose(w_new_2d1)
            #print("W in matrix transpose ",w_new_2d_transpose)
            #self.decision=np.dot(w_new_2d_transpose,np.transpose(self.train_matrix_x))
            self.decision=np.dot(w_new_2d_transpose,np.transpose(self.train_matrix_x))
            self.decision1=np.dot(w_new_2d_transpose1,np.transpose(self.test_matrix_x))
            self.decision=w_new_first+self.decision
            self.decision1=w_new_first1+self.decision1
            #print(" Decision ",self.decision)
            #norm = np.linalg.norm(self.decision)
            #self.decision = self.decision/norm
            self.decision_sigmoid=self.tanh(self.decision)
            self.decision_sigmoid1=self.tanh(self.decision1)
            #print("Sigmoid ",self.tanh(self.decision))
            #print("Sigmoid ",self.decision_sigmoid)
            #print("Sigmoid len ",self.decision_sigmoid.shape)
            #print("Min ",np.min(self.decision_sigmoid))
            #print("Max ",np.max(self.decision_sigmoid))
            #self.splitDataSet()
            #print("Cost-new after ",j,"th iteration ",self.cost(w_new, self.datamatrix_no_label, self.datamatrix_with_label))
            #self.cost(w_new, self.datamatrix_no_label,self.datamatrix_with_label)
            if j>9: 
                #return w_new
                break
            j+=1
            
    def calculation(self):
        row, col = self.datamatrix_no_label.shape
        #row_train,col_train=self.train_matrix_x.shape
        self.w=np.ones((col+1))
        self.w1=np.ones((col+1))
        #self.w=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        #self.w=self.descent(self.w,.099)
        self.descent(self.w,0.00099)
        self.descent(self.w1,.00099)
        #print("Predicted Y : ",self.y_pred)
        #print("After multiplication ",self.y_pred)
        #self.y_pred =  [abs(ele) for ele in self.y_pred]
        #print("After abs value ",self.y_pred)
        #print("Min ",np.min(self.decision*10))
        #print("Max ",np.max(self.decision*10))
        #self.decision*=10
        self._count=0
        self.count=0
        row_decision,col_decision=self.decision.shape
        row_decision1,col_decision1=self.decision1.shape
        self.min_sigmoid=np.min(self.decision_sigmoid)
        self.min_sigmoid1=np.min(self.decision_sigmoid1)
        #self.min_sigmoid=round(self.min_sigmoid,6)
        self.max_sigmoid=np.max(self.decision_sigmoid)
        self.max_sigmoid1=np.max(self.decision_sigmoid1)
        #self.max_sigmoid=round(self.max_sigmoid,6)
        self.avg=(self.min_sigmoid+self.max_sigmoid)/2
        self.avg1=(self.min_sigmoid1+self.max_sigmoid1)/2
        self.avg=round(self.avg,15)
        self.avg1=round(self.avg1,15)
        self.predicted_list=list()
        self.predicted_list1=list()
        if(self.avg<0):
            self.avg*=-1
        #print(self.min_sigmoid," ",self.max_sigmoid," ",self.avg)
        for i in range(0,row_decision):
            for j in range(0,col_decision):
                if(self.decision[i][j]<self.avg):
                    self._count+=1
                    self.predicted_list.append(0)
                elif(self.decision[i][j]>=self.avg):
                    self.count+=1
                    self.predicted_list.append(1)
        self.predicted_class=np.array(self.predicted_list)
        self._count=0
        self.count=0
        
        if(self.avg1<0):
            self.avg1*=-1
        #print(self.min_sigmoid," ",self.max_sigmoid," ",self.avg)
        for i in range(0,row_decision1):
            for j in range(0,col_decision1):
                if(self.decision1[i][j]<self.avg1):
                    self._count+=1
                    self.predicted_list1.append(0)
                elif(self.decision1[i][j]>=self.avg1):
                    self.count+=1
                    self.predicted_list1.append(1)
        self.predicted_class1=np.array(self.predicted_list1)
        
            
        print("Train Set Info")
        print('Accuracy: %.3f' % accuracy_score(self.train_matrix_y, self.predicted_class))
        print('Recall: %.3f' % recall_score(self.train_matrix_y, self.predicted_class))
        print('F1 Score: %.3f' % f1_score(self.train_matrix_y, self.predicted_class))
        print('Precision: %.3f' % precision_score(self.train_matrix_y, self.predicted_class))
        tn, fp, fn, tp = confusion_matrix(self.train_matrix_y, self.predicted_class).ravel()
        specificity = tn / (tn+fp)
        false_discovery_rate=fp/(fp+tp)
        print('Specificity : %.3f' % specificity)
        print('False Discovery Rate : %.3f'% false_discovery_rate)
        
        print('\n')
        
        print("Test Set Info")
        print('Accuracy: %.3f' % accuracy_score(self.test_matrix_y, self.predicted_class1))
        print('Recall: %.3f' % recall_score(self.test_matrix_y, self.predicted_class1))
        print('F1 Score: %.3f' % f1_score(self.test_matrix_y, self.predicted_class1))
        print('Precision: %.3f' % precision_score(self.test_matrix_y, self.predicted_class1))
        tn, fp, fn, tp = confusion_matrix(self.test_matrix_y, self.predicted_class1).ravel()
        specificity = tn / (tn+fp)
        false_discovery_rate=fp/(fp+tp)
        print('Specificity : %.3f' % specificity)
        print('False Discovery Rate : %.3f'% false_discovery_rate)
        #print("Weight\n",self.w)
        #print(len(self.w))
        
class FileReader:
    def __init__(self, filename):
        self.filename = filename

    def readFromCSV(self):
        csvParser = pd.read_csv(self.filename)
        return csvParser

    """"def readFromTXT(self):
        testDataFile = open(self.filename, "r")
        lines = [line.strip().split() for line in testDataFile]
        testDataFile.close()
        print (lines)
        return lines
    def readFromData(self):
        testDataFile = open(self.filename, "r")
        lines = [line.strip().split() for line in testDataFile]
        testDataFile.close()
        return lines

    def readFromTest(self):
        testDataFile = open(self.filename, "r")
        lines = [line.strip().split() for line in testDataFile]
        testDataFile.close()
        return lines[1:]
"""
    def getCSV(self):
        return self.csvData
files=["telco_customer.csv"]
#files=["creditcard.csv"]
#files=["adult.csv"]
for file in files:
    preprocessor=Preprocessor(FileReader(file))