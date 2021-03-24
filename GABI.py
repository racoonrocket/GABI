import numpy as np
import sklearn
import time
import multiprocessing as mp
from tqdm import tqdm
#import _pickle as cPickle
from scipy.sparse import issparse



'''
How to use:
# matrix: MxN binary matrix
#labels: M dim vector of profiles cell type membership
gb = GABI(labels)
gb.fit(matrix)
matrixCT = gb.predict(matrix)
'''


class GABI:

    def __init__(self,labels,tol=1e-5,max_iter=200,verbose=False,RandomSampling=False,Nsamp=50000,ID=''):
        '''
            labels: vector of integers representing the membership of the different profiles
            RandomSampling: The matrix is fitted only on Nsamp samples
        '''
        self.labels = labels.astype(np.int32)
        self.D = len(self.labels)
        self.get_all_combination()

        #Matrices
        self.X = np.zeros((self.D))
        self.tau = np.ones(self.K)*1./self.K
        self.Tki = np.zeros(self.K)
        self.logPXZ = np.zeros((self.D,self.K))

        #Parameters Initialization
        self.a01 = np.random.rand(self.D)
        # self.a10 = np.random.rand(self.D)
        self.a10 = np.ones(self.D)*0.01
        # self.a01 = np.ones(self.D)*0.7
        self.a11 = 1-self.a01
        self.a00 = 1-self.a10

        self.lbound = -np.inf
        self.tol = tol
        self.Q = 0
        self.max_iter=max_iter

        self.likelihood_record = []

        self.verbose = verbose

        self.RandomSampling = RandomSampling
        self.Nsamp = Nsamp
        self.ID = ID

    def update_logPXZ(self):
        '''
            Compute the probability of the observed vector i knowing the state k
            (a priori probability)
        '''
        term11 = np.dot(np.multiply(self.Z,np.log(self.a11)[:,None]).T,self.X) # K x I
        term01 = np.dot(np.multiply(self.Z,np.log(self.a01)[:,None]).T,1-self.X) # K x I
        term00 = np.dot(np.multiply(1-self.Z,np.log(self.a00)[:,None]).T,1-self.X) # K x I
        term10 = np.dot(np.multiply(1-self.Z,np.log(self.a10)[:,None]).T,self.X) # K x I

        self.logPXZ = term11 + term01 + term00 + term10 + np.log(self.tau[:,np.newaxis])


    def e_step(self):
        '''
            Expectation step: Compute the probability of a state k knowing
            the observed vector i (a posteriori probability)
        '''
        num = np.exp(self.logPXZ)
        denom = np.sum(num,axis=0)
        denom[denom < 1e-200]=1e-200

        self.Tki = num/denom[None,:]
        self.Tki[self.Tki<1e-200] = 1e-200


    def m_step_init(self):
        self.num10 = np.zeros(self.D)
        self.denom10 = np.zeros(self.D)
        self.num01 = np.zeros(self.D)
        self.denom01 = np.zeros(self.D)
        self.tau_i = np.zeros(self.K)

    def m_step_i(self):
        '''
            Build iteratively the parameters required for the m-step
        '''
        _ZT =  np.dot((1-self.Z),self.Tki) #DxI
        self.num10 += np.diag(np.dot(self.X,_ZT.T))
        self.denom10 += _ZT.sum(axis=1)

        ZT =  np.dot(self.Z,self.Tki) #DxI
        self.num01 += np.diag(np.dot(1-self.X,ZT.T))
        self.denom01 += ZT.sum(axis=1)

        self.tau_i +=  self.Tki.sum(axis=1)

    def m_step(self):
        '''
            Last step of the m-step, the actualisation of the parameters
        '''
        self.denom10[self.denom10<1e-200]=1e-200
        self.denom01[self.denom01<1e-200]=1e-200

        self.a10 = self.num10/self.denom10
        self.a01 = self.num01/self.denom01

        self.tau = self.tau_i/float(self.N)


        self.a11 = 1 - self.a01
        self.a00 = 1 - self.a10

        #Avoid NaNs from log of negative value
        self.a01.setflags(write=True)
        self.a10.setflags(write=True)
        self.a11.setflags(write=True)
        self.a00.setflags(write=True)
        self.tau.setflags(write=True)

        self.a10[self.a10<1e-200] = 1e-200
        self.a01[self.a01<1e-200] = 1e-200
        self.a11[self.a11<1e-200] = 1e-200
        self.a00[self.a00<1e-200] = 1e-200

    def set_parameter(self):
        '''
            Save the optimal parameters
        '''
        self.a10_final = self.a10
        self.a01_final = self.a01
        self.a11_final = self.a11
        self.a00_final = self.a00
        self.tau_final = self.tau


    def get_parameter(self):
        return self.a10_final,self.a01_final,self.a11_final,self.a00_final,self.tau_final


    def predict(self,matrix,GetProba=False):
        '''
            Predict the matrix based on the distance of each genomic site to
            the closest state of reference

            GetProba: Get the probability to have a one for each cell type
        '''
        #Initialization of matrix batches
        self.N = matrix.shape[1]
        idx1 = np.where(matrix.sum(axis=0)>0)[0]
        N1 = len(idx1)

        #Set best estimation
        self.a10 = self.a10_final
        self.a01 = self.a01_final
        self.a11 = self.a11_final
        self.a00 = self.a00_final
        self.tau = self.tau_final

        if GetProba:
            matrixProba1 = np.zeros((self.NCT,N1))
            for i in tqdm(range(N1),disable=not(self.verbose),desc=self.ID):
                self.get_X(matrix,idx1[i])

                self.update_logPXZ()
                self.e_step()
                matrixProba1[:,i] = self.combmat.dot(self.Tki).squeeze()

            #Restore the zeros
            matrixProba = np.zeros((self.NCT,self.N))
            matrixProba[:,idx1] =  matrixProba1
            return matrixProba

        else:
            self.labels_state = np.zeros(N1)
            for i in tqdm(range(N1),disable=not(self.verbose),desc=self.ID):
                # self.X = matrix[:,idx1[i]][:,None]
                self.get_X(matrix,idx1[i])

                self.update_logPXZ()
                self.e_step()
                self.labels_state[i] = self.Tki.argmax(axis=0)
                # PZ_X_max[i] = self.Tki.max(axis=0)

            if self.verbose: print ('Building matrix from states')
            matrixCT1 = np.zeros((self.NCT,N1))
            for l in tqdm(range(self.K),disable=not(self.verbose),desc=self.ID):
                matrixCT1[:,self.labels_state==l] = self.combmat[:,l][:,None]

            #Restore the zeros of the matrix
            labels_stateT = np.zeros(self.N)
            matrixCT = np.zeros((self.NCT,self.N))
            matrixCT[:,idx1] = matrixCT1
            labels_stateT[idx1] = self.labels_state
            self.labels_state = labels_stateT

            return matrixCT

    def lower_bound_init(self):
        self.Q = 0

    def lower_bound_i(self):
        self.Q += np.multiply(self.Tki[None,:,:],self.logPXZ).sum() - np.multiply(self.Tki,np.log(self.Tki)).sum()

    def fit(self,matrix):
        self.issparse = issparse(matrix)

        #Check if the matrix is Binary
        if not check_if_binary_matrix(matrix):
            print ('Matrix must be binary !')
            raise ValueError

        matrix = matrix.astype(np.int32)


        #Initialization of matrix batches
        self.N = matrix.shape[1]
        idx1 = np.where(matrix.sum(axis=0)>0)[0]
        N1 = len(idx1)

        #Begining Fitting
        self.lbound = -np.inf
        for self.n_iter in range(self.max_iter):

            self.m_step_init()
            self.lower_bound_init()

            if self.RandomSampling:
                idxR = np.random.permutation(N1)[:self.Nsamp]
            else:
                idxR = np.arange(N1)

            start_time = time.time()
            for i in tqdm(range(len(idxR)),disable=not(self.verbose),desc=self.ID):
                # self.X = matrix[:,idx1[idxR[i]]][:,None]
                self.get_X(matrix,idx1[idxR[i]])
                self.update_logPXZ()
                self.e_step()

                self.m_step_i()
                self.lower_bound_i()

            #print timer
            # if self.verbose: print 'Timer = ' +  str(time.time() - start_time)

            # The lower bound is computed with the parameters at t-1
            # thus the parameters are attributed with the m-step after
            # the comparison of the lower bound withe the lowest bound
            if self.Q > self.lbound:
                self.set_parameter() #Store the parameters as best
                self.diff = abs(self.Q - self.lbound)
                self.lbound = self.Q
                if self.verbose: print ('Lowest Bound: ' + str(self.lbound) + ' , Diff: ' + str(self.diff))

            #Gradient on the parameters
            self.m_step()

            if self.diff <= self.tol:
                if self.verbose: print('Convergence after {0} Iteration'.format(self.n_iter))
                break

            #Get record of the evolution of the likelihood
            self.set_likelihood_record()

        if self.verbose: print ('Final Likelihood ( Nstates =' + str(self.K) + ' ): ' + str(self.Q))

    def get_X(self,matrix,idx):
        '''
            Get matrix Slice
        '''
        if self.issparse:
            #iloc added
            self.X = matrix.iloc[:,idx].toarray()
            # self.X = self.X[:,None]
        else:
            #iloc added
            self.X = matrix.iloc[:,idx][:,None]

    def set_likelihood_record(self):
        self.likelihood_record.append(self.Q)

    def get_all_combination(self):
        '''
            Defines all the combinaison between the different clusters
            defined by the labels
        '''
        self.NCT = int(np.max(self.labels)+1)   #Number of cell types
        self.K = int(pow(2,self.NCT))           #Number of combination

        self.combmat = np.zeros((self.NCT,self.K))

        for m in range(self.K):
            vect = np.zeros(self.NCT)
            pow2 = np.array([pow(2,i) for i in range(self.NCT)])

            for i in range(self.NCT)[::-1]:
                vect[i]=1
                if m<np.sum(pow2[vect>0]):
                    vect[i]=0

                elif m==np.sum(pow2[vect>0]):
                    break

            self.combmat[:,m]=vect

        self.Z = comb2states(self.labels,self.combmat)

    def reconstruct_matrix(self,idx=[]):
        '''
            Reconstruct the infered matrix with all the replicates

            idx: indexes of the sites that will be computed
        '''
        if len(idx)==0 : idx = np.arange(len(self.labels_state))
        labels_stateC = self.labels_state[idx]
        matrixR = np.zeros((self.D,len(labels_stateC)))
        for l in np.unique(labels_stateC):
            matrixR[:,labels_stateC==l] = self.Z[:,int(l)][:,None]

        return matrixR

    def save(self,datapath):
        """save class"""
        file = open(datapath,'w')
        file.write(cPickle.dumps(self.__dict__))
        file.close()

    def load(self,datapath):
        """ load class"""
        file = open(datapath,'r')
        dataPickle = file.read()
        file.close()
        self.__dict__ = cPickle.loads(dataPickle)

    def get_pickle(self):
        return cPickle.dumps(self.__dict__)

    def set_pickle(self,pickle):
        self.__dict__ = cPickle.loads(pickle)


def check_if_binary_matrix(matrix):
    #Define if the matrix is boolean or Discrete
    uniq = np.unique(matrix)
    if len(uniq)==2:
        Binary = np.all(uniq==np.array([False,True]))
    else:
        Binary = False

    return Binary

def comb2states(labels,combmat):
    '''
        Combination of cell types to combination of profiles.
    '''
    M = len(labels)
    NCT = int(labels.max() + 1)

    Rmat = np.zeros((M,NCT))
    for i in range(NCT):
        print(labels)
        print("lenramta = ")
        print(len(Rmat))
        print("ramta = ")
        print(Rmat)
        print("###################")
        print([labels == i])
        print("test")
        #try and pass added
        #Rmat[labels==i,i] modified to work
        #/!\
        for k in range(len(labels==i)):
            if (labels == i).iat[k,0]:
                Rmat[k,i] = 1
        print("RMAT")
        print(Rmat)
    statemat = Rmat.dot(combmat)

    return statemat
