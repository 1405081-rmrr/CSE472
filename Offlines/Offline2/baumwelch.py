import numpy as np
import sys
from numpy.linalg import inv
from random import seed
from random import random
from numpy.lib.function_base import cov
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
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
        self.baumtransition=list()
        self.baumemission=list()
        self.gaussianresult=list()
        self.gaussianresulttemp=list()
        self.mus=0.0
        self.sigmas=0.0
        self.P=0.0
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
        temp=list()
        temp.append(float(self.parameterlist[1]))
        temp.append(float(self.parameterlist[2]))
        self.baumtransition.append(temp)
        temp=list()
        temp.append(float(self.parameterlist[3]))
        temp.append(float(self.parameterlist[4]))
        self.baumtransition.append(temp)
        self.AnnualQ = np.loadtxt('/home/roktim/Desktop/ML_Offline/data.txt')
        self.logQ = np.log(self.AnnualQ)
        self.mus, self.sigmas, self.P = self.fitHMM(self.logQ, 1000) #mus->mean sigmas->Sd P->TransitionMatrix
        print("Mean ",self.mus)
        print("Standard Deviation ",self.sigmas) #eta kintu sigma^2.
        print("Transition Matrix",self.P)
        self.transitionmatrix=self.P
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
            self.mean=self.mus[iteration][0]
            self.standard_deviation=self.sigmas[iteration][0]
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
        f=open('/home/roktim/Desktop/ML_Offline/resultviterbi.txt','w')
        for item in self.viterbi_output:
            writeitem=str(item)+'\n'
            f.write(writeitem)
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
        
    def baumwelch(self):
        print("TransitionMatrix ",self.baumtransition) 
        self.baumemission=self.stationaryMatrix(self.baumtransition)
        print("emission Matrix ",self.baumemission)
        self.mean=self.mus.ravel()
        self.standard_deviation=self.sigmas.ravel()
        print("Mean ",self.mean)
        self.highest_prob=list()
        self.iteration_prob=list()
        print("SD ",self.standard_deviation)
        #self.call_baum()
        for i in range(30):
            self.call_baum(i)
    def baumwelch_calc(self,gaussvalue1,gaussvalue2):
        temp_list=list()
        for i in range(self.n):
            for j in range(self.n):
                temp_list.append(self.baumemission[j]*self.normal_distribution(gaussvalue1,self.mean[j],self.standard_deviation[j])*self.baumtransition[i][j]*self.normal_distribution(gaussvalue2,self.mean[i],self.standard_deviation[i]))
        self.highest_prob.append(max(temp_list))
        self.iteration_prob.append(temp_list)
        self.highest_probability=sum(self.highest_prob)
    def call_baum(self,iter):
        for i in range(len(self.logQ)-1):
            self.baumwelch_calc(self.logQ[i],self.logQ[i+1])
        print('\n')
        self.new_transition=list()
        self.new_transition_matrix=list()
        for i in range(len(self.iteration_prob[0])):
            _sum=0.0
            for _list in self.iteration_prob:
                _sum+=_list[i]
            self.new_transition.append(_sum)
        for i in self.new_transition:
            self.new_transition_matrix.append(i/self.highest_probability)
        _sum=sum(self.new_transition_matrix)
        _sum1=self.new_transition_matrix[0]+self.new_transition_matrix[2]
        _sum2=self.new_transition_matrix[1]+self.new_transition_matrix[3]
        #print("New matrix before normalize ",self.new_transition_matrix)
        for i in range(len(self.new_transition_matrix)):
            if(i%2==0):
                self.new_transition_matrix[i]=self.new_transition_matrix[i]/_sum1
            else:
                self.new_transition_matrix[i]=self.new_transition_matrix[i]/_sum2
        self.new_transition_matrix=np.array(self.new_transition_matrix)
        self.new_transition_matrix=self.new_transition_matrix.reshape(2,2)
        self.baumtransition=self.new_transition_matrix
        print("New matrix after iteration {} ".format(iter+1),self.baumtransition)
gaussian_distribution=Gaussian_distribution()
#gaussian_distribution.readFile()
gaussian_distribution.readData()
gaussian_distribution.baumwelch()
