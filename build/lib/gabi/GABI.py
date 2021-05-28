import numpy as np
#from sklearn.utils.extmath import logsumexp
#from sklearn.exceptions import ConvergenceWarning
import sklearn
import time
import multiprocessing as mp
from tqdm import tqdm
import pickle
from scipy.sparse import issparse, csr_matrix, vstack
import yaml
import pyBigWig as bg


'''
How to use:
# matrix: MxN binary matrix
#labels: M dim vector of profiles cell type membership
gb = GABI(labels)
gb.fit(matrix)
matrixCT = gb.predict(matrix)
'''


class singlecore:

    def __init__(self,labels=None,tol=1e-5,max_iter=200,bw=False,yamfile="sources.yaml",chr_list = [],verbose=False,RandomSampling=False,Nsamp=50000,ID=''):
        '''
            labels: vector of integers representing the membership of the different profiles
            RandomSampling: The matrix is fitted only on Nsamp samples
        '''
        self.bw = bw
        if self.bw:
            self.yamfile = yamfile
            self.chr_list = chr_list
            self.binsize = 200
            self.matrixbw = self.load_bigwig()
            self.labels = self.labels.astype(np.int32)
        elif not self.bw:
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

    def load_bigwig(self):
        """ TAke as input a path to a  .txt containing BigWig file path and return a matrix of concatenated epigenomic profiles of this BigWig
        INPUT ::
        path_to_all_bw : type = string, default = None is mandatory
        bin_size : type = int , default = 500 kb
        list_chr : type = list, defautl = ["chr21"] list of the chr you want to concatenate into you gabi matrix
        OUTPUT ::
        sparsedmatrix :: type = numpy.array ( a sparse matrix containing all epigenomics profile)
        labelmatrix :: type = numpy.array ( array containing all the label of the same or not epigenomic profiles=
        ouput =
        labels = [1,1,2,..]"""
        self.labels = []
        self.general_positions = []
        self.specific_position = []
        compteur = 0
        listaas = []
        self.chrsizes = []
        # pickle.dump(sparsedmatrix, open('sparsed.p', 'wb+'))
        # First step make the labels vector
        with open(self.yamfile) as yaml_file:
            BW_paths = yaml.load(yaml_file)
            BW_paths2 = []
            for key in BW_paths:
                self.labels.extend([key] * len(BW_paths[key]))
                BW_paths2.extend([element for element in BW_paths[key]])
            for OneBWPath in BW_paths2:
                print(OneBWPath)
                # Second step start to work on the BigWig file, binin and creating the matrix
                with bg.open(OneBWPath) as BigWig:  # open the big wig
                    chrom_dict = BigWig.chroms()
                    for element in self.chr_list:
                        # since there is no function to extract a specific bin size along a big wig
                        # We create the maximum number of bin from the choosed bin size and add a litle bin to complete a the end
                        # total nucleotides = number_of_bins * binsize + litle_bin (simple euclidian division)
                        number_of_bins = chrom_dict[element] // self.binsize
                        litle_bin = chrom_dict[element] % self.binsize
                        chromvalues = [BigWig.stats(element, self.binsize * i, self.binsize * (i + 1))[0] for i in
                                       range(number_of_bins)]
                        if litle_bin != 0:
                            chromvalues.extend(
                                BigWig.stats(element, self.binsize * number_of_bins, chrom_dict[element]))
                        chromvalues = [0 if chromvalues[i] is None else chromvalues[i] for i in range(len(chromvalues))]
                        chromvalues = [True if int(chromvalues[i]) > 0.5 else False for i in range(len(chromvalues))]
                        # here comes a lot of variables to keep track of index , start en ending of each bins
                        # This is needed to recreate a bigwig at the end of GABI
                        if BW_paths2.index(OneBWPath) == 0:
                            self.general_positions.append([self.binsize * i + 1 for i in range(number_of_bins)])
                            if litle_bin != 0:
                                self.general_positions[-1].append(number_of_bins * self.binsize + litle_bin + 1)
                            if len(self.specific_position) != 0:
                                self.specific_position.append(
                                    [self.binsize * i + self.specific_position[-1][-1] + 1 for i in
                                     range(number_of_bins)])
                            else:
                                self.specific_position.append([self.binsize * i + 1 for i in range(number_of_bins)])
                            self.savedheader = BigWig.header()
                            self.chrsizes.append(chrom_dict[element])
                            print(self.savedheader)
                    # the matrix was saved on the hard disk to save some RAM
                    # It was more usefull before when the matrix was not sparsed
                    # maybe its not usefull anymore will see
                    # listaas = pickle.load(open('sparsed.p', 'rb'))
                    if len(listaas) == 0:
                        listaas = [csr_matrix(chromvalues)]
                    else:
                        listaas.append(csr_matrix(chromvalues))
                    # pickle.dump(listaas, open('sparsed.p', 'wb+'))
                    # listaas = None
        # listaas = pickle.load(open('sparsed.p', 'rb'))
        self.labels = np.array(self.labels)
        return vstack(listaas)

    def save_as_bigwig(self, matrixCT,prename="consolidated"):
        """
        Take the outpout of GABI predict or GABI MP (matrixCT) and saved it as a one big wig file per consolidate profile
        INPUT::
        GABI object class
        matrixCT from GABI.predict() function
        prename the name you want to choose for you bw
        """
        profile_number = 0
        for profile in matrixCT:
            with bg.open(prename + "_" + str(profile_number) + ".bw", "w") as bw:
                profile_number += 1
                newheader = []
                for k in range(len(self.chr_list)):
                    print("chrsizek", self.chrsizes)
                    newheader.append((self.chr_list[k], self.chrsizes[k]))
                bw.addHeader(newheader)
                # print(self.general_positions,self.chrsizes)
                # print(self.general_positions,self.general_positions[1:]+self.chrsizes,matrixCT)
                print("len", self.general_positions[0], "chrsize", self.chrsizes)
                # faut créer un bigwig part type cellulaire
                for i in range(len(self.chr_list)):
                    print(self.chr_list[i] * len(self.general_positions[i]))
                    print(self.general_positions[i])
                    print(self.general_positions[i][1:] + [self.chrsizes[i]])
                    bw.addEntries([self.chr_list[i]] * len(self.general_positions[i]),
                                  self.general_positions[i],
                                  ends=self.general_positions[i][1:] + [self.chrsizes[i]],
                                  values=[profile[k] for k in range(len(profile))],
                                  validate=False)

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
        #added int
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


    def predict(self,matrix=None,GetProba=False):
        '''
            Predict the matrix based on the distance of each genomic site to
            the closest state of reference

            GetProba: Get the probability to have a one for each cell type
        '''
        if self.bw:
            matrix = self.matrixbw
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

    def fit(self,matrix=None):
        if self.bw:
            matrix = self.matrixbw
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
            self.X = matrix[:,idx].toarray()
            # self.X = self.X[:,None]
        else:
            self.X = matrix[:,idx][:,None]

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
    if not issparse(matrix):
        uniq = np.unique(matrix)
        print(uniq)
        print("UNIQUE IS UNIQUE" + str(uniq))
        if len(uniq)==2:
            Binary = np.all(uniq==np.array([False,True]))
        else:
            Binary = False
    else:
        #if its a sparsed matrix it should only have one value in it if the non sparsed is binary
        if len(np.unique(matrix.data)) ==1:
            Binary = True
        else:
            print(np.unique(matrix.data))
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
        Rmat[labels==i,i] = 1
    statemat = Rmat.dot(combmat)

    return statemat
