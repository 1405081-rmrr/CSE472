import numpy as np
import sys
from numpy.linalg import inv
from random import seed
from random import random
from numpy.lib.function_base import cov
import matplotlib.pyplot as plt
#from hmmlearn.hmm import GaussianHMM
from dataprocess_hmm import GaussianHMM
#import sklearn
class Gaussian_distribution:
    def __init__(self):
        self.datalist=list()
        self.parameterlist=list()
        self.gaussResult=0.0
        self.mean=0.0
        self.standard_deviation=0.0
        self.temp=list()
        self.transitionmatrix=0.0
        self.stationary=0.0
        self.viterbi_output=list()
        self.logQ=0.0
        self.n=0
        self.gaussianresult=list()
        self.gaussianresulttemp=list()
    def fitHMM(self,Q, nSamples):
        model = GaussianHMM(n_components=self.n, n_iter=1000).fit(np.reshape(Q,[len(Q),1]))
        mus = np.array(model.means_)
        _temp=list()
        for i in range(self.n):
            _temp.append(np.diag(model.covars_[i]))
        sigmas=np.array(np.sqrt(np.array(_temp)))
        #sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]),np.diag(model.covars_[1])])))
        P = np.array(model.transmat_)
        samples = model.sample(nSamples)
        if mus[0] > mus[1]:
            mus = np.flipud(mus)
            sigmas = np.flipud(sigmas)
            P = np.fliplr(np.flipud(P))
 
        return mus, sigmas, P
    def readData(self):
        
        with open('/home/roktim/Desktop/ML_Offline/parameters.txt') as parameter:
            for line in parameter:
                x=line.split()
                for item in x:
                    self.parameterlist.append(item)
        #print(self.parameterlist)
        self.n=int(self.parameterlist[0])
        #self.n=3
        self.AnnualQ = np.loadtxt('/home/roktim/Desktop/ML_Offline/data.txt')
        self.logQ = np.log(self.AnnualQ)
        mus, sigmas, P = self.fitHMM(self.logQ, 1000) #mus->mean sigmas->Sd P->TransitionMatrix
        print("Mean ",mus)
        print("Standard Deviation ",sigmas) #eta kintu sigma^2.
        print("Transition Matrix",P)
        self.transitionmatrix=P
        self.stationary=self.stationaryMatrix(self.transitionmatrix)
        """
        self.mean_class1=mus[0][0]
        self.mean_class2=mus[1][0]
        self.standard_deviation1=sigmas[0][0]
        self.standard_deviation2=sigmas[1][0]
        #self.mean_class1=200
        #self.mean_class2=100
        #self.standard_deviation1=10
        #self.standard_deviation2=10
        self.stationary=self.stationaryMatrix(self.transitionmatrix)
        print("Stationary Matrix ",self.stationary)
        print('\n')
        self.gaussResult1=self.normal_distribution_class1(self.logQ,self.mean_class1,self.standard_deviation1)
        #print("Gauss for 1",self.gaussResult1)
        self.gaussResult2=self.normal_distribution_class1(self.logQ,self.mean_class2,self.standard_deviation2)
        self.datapass()
        """
        for iteration in  range(self.n):
            self.mean=mus[iteration][0]
            self.standard_deviation=sigmas[iteration][0]
            self.gaussianresulttemp=self.normal_distribution(self.logQ,self.mean,self.standard_deviation)
            self.gaussianresult.append(self.gaussianresulttemp)
        self.datapass()
    def normal_distribution(self,x,mean,sd):
        prob_density = (1/((np.pi*2*sd)**0.5)) * np.exp(-0.5*((x-mean)**2/(sd))) 
        return prob_density
    def stationaryMatrix(self,transitionmatrix):
        self.temp=list()
        a=np.append(np.transpose(transitionmatrix)-np.identity(self.n),[np.ones(self.n)],axis=0)
        for i in range(self.n):
            self.temp.append(0)
        self.temp.append(1)
        b=np.transpose(np.array(self.temp))
        
        return (np.linalg.solve(np.transpose(a).dot(a), np.transpose(a).dot(b)))

    def initial_viterbi(self,gaussianvalue,stationary):
        previousprob=list()
        """prob_lanina=gaussian_value1*stationary[0]
        previousprob.append(prob_lanina)
        prob_elnino=gaussian_value2*stationary[1]
        previousprob.append(prob_elnino)
        result=max(prob_lanina,prob_elnino)
        if(prob_lanina>prob_elnino):
            self.viterbi_output.append("Lanina")
            #print("Lanina")
        elif(prob_lanina<=prob_elnino):
            self.viterbi_output.append("Elnino")
            #print("Elnino")
        return previousprob
        """
        for i in range(self.n):
            probability=gaussianvalue[i]*stationary[i]
            previousprob.append(probability)
        result=max(previousprob)
        max_prob_index=previousprob.index(result)
        self.viterbi_output.append(max_prob_index)
        return previousprob
        
    def viterbi(self,transitionmatrix,gaussianvalue,prevvalue):
        new_prob_store=list()
        """joint_prob_elnino_given_elnino=gaussianvalue1*transitionmatrix[0][0]
        temp_prob1=joint_prob_elnino_given_elnino*prevvalue[0]
        joint_prob_elnino_given_lanina=gaussianvalue1*transitionmatrix[1][0]
        temp_prob2=joint_prob_elnino_given_lanina*prevvalue[1]
        set_max_joint_lanina=max(temp_prob1,temp_prob2)
        new_prob_store.append(set_max_joint_lanina)
        
        
        joint_prob_lanina_given_elnino=gaussianvalue2*transitionmatrix[0][1]
        temp_prob1=joint_prob_lanina_given_elnino*prevvalue[0]
        joint_prob_lanina_given_lanina=gaussianvalue2*transitionmatrix[1][1]
        temp_prob2=joint_prob_lanina_given_lanina*prevvalue[1]
        set_max_joint_elnino=max(temp_prob1,temp_prob2)
        new_prob_store.append(set_max_joint_elnino)
        """
        for i in range(self.n):
            temp_prob=list()
            for j in range(self.n):
                joint_prob=gaussianvalue[i]*transitionmatrix[j][i]
                joint_prob=joint_prob*prevvalue[j]
                temp_prob.append(joint_prob)
            set_max=max(temp_prob)
            new_prob_store.append(set_max)
        max_class=max(new_prob_store)
        _index=new_prob_store.index(max_class)
        self.viterbi_output.append(_index)
        return new_prob_store
    def datapass(self):
        flag=1
        prev_value=list()
        num_of_iter=len(self.gaussianresult[0])
        """for(gausvalue1,gausvalue2)in zip(self.gaussResult1,self.gaussResult2):
            if(flag==1):
                prev_value=self.initial_viterbi(gausvalue1,gausvalue2,self.stationary)
                #print(prev_value)
                flag=0
            else:
                prev_value=self.viterbi(self.transitionmatrix,gausvalue1,gausvalue2,prev_value)
                #print(prev_value)
        """
        for item in range(num_of_iter):
            gaussvalue=list()
            for _list in self.gaussianresult:
                gaussvalue.append(_list[item])
            if(flag==1):
                prev_value=self.initial_viterbi(gaussvalue,self.stationary)
                flag=0
            else:
                prev_value=self.viterbi(self.transitionmatrix,gaussvalue,prev_value)
        self._count()
    def _count(self):
        frequency_mycode={}
        for item in self.viterbi_output:
            if item in frequency_mycode:
                frequency_mycode[item]+=1
            else:
                frequency_mycode[item]=1
        print("My code Frequency ",frequency_mycode)
        with open('/home/roktim/Desktop/ML_Offline/viterbioutput.txt') as vout:
            templist=list()
            frequency_given={}
            for line in vout:
                line = line.replace('"', '').strip()
                templist.append(line)
        for item in templist:
            if item in frequency_given:
                frequency_given[item]+=1
            else:
                frequency_given[item]=1
        print("Given Frequency ",frequency_given)
        self.result_output()
    def result_output(self):
        for item in self.viterbi_output:
            pass
             
gaussian_distribution=Gaussian_distribution()
#gaussian_distribution.readFile()
gaussian_distribution.readData()
