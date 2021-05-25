import numpy as np

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import pairwise_distances
# import multiprocessing as mp
#from multiprocessing import Process, Manager, Pipe
import yaml
import pyBigWig as bg
from threading import Thread
import queue as Queue

from scipy.sparse import csr_matrix,coo_matrix,issparse,vstack,hstack
import pickle

from tqdm import tqdm

import GABI as gbi

class GABI:
    def __init__(self,matrix=None,labels=None,distmat=[],bw=False,yamfile="sources.yaml",chr_list = [],verbose=False,NClust=5,tol=1e-2,max_iter=200):

        self.bw = bw
        if self.bw:
            self.yamfile = yamfile
            self.chr_list = chr_list
            self.binsize = 1000
            self.matrixbw = self.load_bigwig()
            self.labels = self.labels.astype(np.int32)
            matrix = self.matrixbw
            self.issparse = issparse(matrix)
        elif not self.bw:
            self.labels = labels.astype(np.int32)
            self.labels = labels
            self.issparse = issparse(matrix)

        self.distmat = distmat
        self.verbose = verbose
        self.NClust = NClust

        #Check if the matrix is Binary
        if not check_if_binary_matrix(matrix):
            print ('Matrix must be binary !')
            raise ValueError

        #Check input variables
        if self.NClust>self.labels.max()+1: self.NClust = self.labels.max()+1

        #Compute the Distance matrix
        if len(distmat)==0:
            print ('Compute the Distance matrix')
            idx1 = np.where(matrix.sum(axis=0)>0)[0]   #We compute only the non zeros columns
            if not issparse(matrix):
                idx1 = np.where(matrix.sum(axis=0) > 0)[0]  # We compute only the non zeros columns
                self.distmat = yule_distance(matrix[:,idx1])
                self.distmat[np.isnan(self.distmat)] = 1 #If a profile is null, Nan are produced
                print("ya",self.distmat)
            else:
                # if the column exist in that sparsed matrix -> this column countain at least one one
                idx1 = np.unique(matrix.indices)
                self.distmat = yule_distance(matrix[:, idx1].toarray())
                self.distmat[np.isnan(self.distmat)] = 1
                print(self.distmat)
        if verbose: print ('Split Labels')
        self.labels_c,self.labels_cS,self.Idxlabels_c = get_labels_split(self.labels,self.distmat,self.NClust)
        if not issparse(matrix):
            self.matrix_c = [matrix[idx,:] for idx in self.Idxlabels_c]
        else:
            self.matrix_c = [matrix[idx, :].toarray() for idx in self.Idxlabels_c]

        self.NP = len(self.labels_c)
        self.queue = Queue.Queue()
        for k in range(self.NP):
            self.queue.put({'gb': gbi.GABI(self.labels_cS[k],verbose=verbose,tol=tol,max_iter=max_iter,ID='Thread: {}'.format(k)),
                            'matrix': self.matrix_c[k]
                            })

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
        #pickle.dump(sparsedmatrix, open('sparsed.p', 'wb+'))
        # First step make the labels vector
        with open(self.yamfile) as yaml_file:
            BW_paths = yaml.load(yaml_file)
            BW_paths2 = []
            for key in BW_paths:
                self.labels.extend([key] * len(BW_paths[key]))
                BW_paths2.extend([element for element in BW_paths[key]])
            for OneBWPath in BW_paths2:
                # Second step start to work on the BigWig file, binin and creating the matrix
                with bg.open(OneBWPath) as BigWig:  # open the big wig
                    chrom_dict = BigWig.chroms()
                    for element in self.chr_list:
                        #since there is no function to extract a specific bin size along a big wig
                        #We create the maximum number of bin from the choosed bin size and add a litle bin to complete a the end
                        #total nucleotides = number_of_bins * binsize + litle_bin (simple euclidian division)
                        number_of_bins = chrom_dict[element] // self.binsize
                        litle_bin = chrom_dict[element] % self.binsize
                        chromvalues = [BigWig.stats(element, self.binsize * i, self.binsize * (i + 1))[0] for i in
                                       range(number_of_bins)]
                        if litle_bin != 0:
                            chromvalues.extend(BigWig.stats(element, self.binsize * number_of_bins, chrom_dict[element]))
                        chromvalues = [True if chromvalues[i] > 0.5 else False for i in range(len(chromvalues))]
                        #here comes a lot of variables to keep track of index , start en ending of each bins
                        #This is needed to recreate a bigwig at the end of GABI
                        if BW_paths2.index(OneBWPath) == 0:
                            self.general_positions.append([self.binsize * i + 1 for i in range(number_of_bins)])
                            if litle_bin != 0:
                                self.general_positions[-1].extend(number_of_bins * self.binsize + litle_bin + 1)
                            if len(self.specific_position) != 0:
                                self.specific_position.append([self.binsize * i + self.specific_position[-1][-1] + 1 for i in range(number_of_bins)])
                            else:
                                self.specific_position.append([self.binsize * i + 1 for i in range(number_of_bins)])
                            self.savedheader = BigWig.header()
                            self.chrsizes.append(chrom_dict[element])
                            print(self.savedheader)
                    #the matrix was saved on the hard disk to save some RAM
                    #It was more usefull before when the matrix was not sparsed
                    #maybe its not usefull anymore will see
                    #listaas = pickle.load(open('sparsed.p', 'rb'))
                    if len(listaas) == 0:
                        listaas = [csr_matrix(chromvalues)]
                    else:
                        listaas.append(csr_matrix(chromvalues))
                    #pickle.dump(listaas, open('sparsed.p', 'wb+'))
                    #listaas = None
        #listaas = pickle.load(open('sparsed.p', 'rb'))
        self.labels = np.array(self.labels)
        return vstack(listaas)

    def save_as_bigwig(self, matrixCT,prename="Results/consolidated"):
        """
        Take the outpout of GABI predict or GABI MP (matrixCT) and saved it as a one big wig file per consolidate profile
        INPUT::
        GABI object class
        matrixCT from GABI.predict() function

        """
        profile_number = 0
        for profile in matrixCT:
            with bg.open(prename + "_" + str(profile_number) + ".bw","w") as bw:
                profile_number +=1
                newheader = []
                for k in range(len(self.chr_list)):
                    print("chrsizek", self.chrsizes)
                    newheader.append((self.chr_list[k], self.chrsizes[k]))
                bw.addHeader(newheader)
                #print(self.general_positions,self.chrsizes)
                #print(self.general_positions,self.general_positions[1:]+self.chrsizes,matrixCT)
                print("len",self.general_positions[0],"chrsize",self.chrsizes)
                    #faut créer un bigwig part type cellulaire
                for i in range(len(self.chr_list)):
                    print(self.chr_list[i]*len(self.general_positions[i]))
                    print(self.general_positions[i])
                    print(self.general_positions[i][1:]+[self.chrsizes[i]])
                    bw.addEntries([self.chr_list[i]]*len(self.general_positions[i]),self.general_positions[i],ends=self.general_positions[i][1:]+[self.chrsizes[i]],values=[profile[k] for k in range(len(profile))],validate=False)


    def fit(self):
        '''
            Fit the different parts with GABI
        '''

        if self.verbose: print('Annotations')
        # recv_end_c, send_end_c = zip(*[Pipe(False) for k in range(self.NP)])
        list_threads = []
        while not self.queue.empty():
            for k in range(self.NP):
                t = Threading(self.queue, k)
                list_threads.append(t)
                t.start()

            for thread in list_threads:
                thread.join()

        self.gb_c = [l.gb for l in list_threads]

        # try:
        #     for thread in list_threads:
        #         thread.join()

        #     # gb = [recv_end.recv() for recv_end in recv_end_c]
        #     self.gb_c = [recv_end.recv() for recv_end in recv_end_c]

        # except KeyboardInterrupt:
        #     self.gb_c = [recv_end.recv() for recv_end in recv_end_c]


    def predict(self,GetProba=False):
        '''
            Predict the output matrix
        '''

        matrixCT_c = [(self.gb_c[k]).predict(self.matrix_c[k]) for k in range(self.NP)]
        if self.verbose: print ('Merge Splitted matrices')
        self.matrixCT = merge_split_matrices(matrixCT_c,self.labels_c)

        if self.verbose: print ('Get all combinations')
        self.combmat,self.labels_states,self.counts = get_all_combination(self.matrixCT)

        #Get combinations with replicates
        self.statemat = comb2states(self.labels,self.combmat)

        if GetProba:
            matrixProba_c = [(self.gb_c[k]).predict(self.matrix_c[k],GetProba=GetProba) for k in range(self.NP)]
            if self.verbose: print ('Merge Splitted matrices')
            self.matrixProba = merge_split_matrices(matrixProba_c,self.labels_c)
            return self.matrixCT,self.matrixProba,self.statemat,self.labels_states

        else:
            return self.matrixCT,self.statemat,self.labels_states


    def get_FP_FN(self):
        '''
            Return FP, FN, TP, TN from GABI optimizations
        '''
        FP = np.zeros(self.distmat.shape[0])
        FN = np.zeros(self.distmat.shape[0])

        for k,idx in enumerate(self.Idxlabels_c):
            FP[idx] = self.gb_c[k].a10_final
            FN[idx] = self.gb_c[k].a01_final

        TN = 1-FP
        TP = 1-FN


        return FP,FN,TP,TN

    def reconstruct_matrix(self,idx=[]):
        '''
            Reconstruct the infered matrix with all the replicates

        '''
        return comb2states(self.labels,self.matrixCT[:,:idx])

    def Optimal_ordering(self):
        '''
            Get an otpimal order for the Annoation matrix and the names
            from a hierarchical clustering leaf index

            return:
                idxOpt: Optimal index for vectors with only cell types
                idxOptRep: Optimal index for vectors with the cell types replicates
        '''
        distAnn = yule_distance(self.matrixCT)
        distAnn = (distAnn + distAnn.T)/2. #Symetrize in case of some numerical error
        distAnn = distAnn - np.diag(np.diag(distAnn))
        Z = linkage(squareform(distAnn), method='average', metric='precomputed')
        self.idxOpt = leaves_list(Z).astype(np.int32)
        self.idxOptRep = np.hstack([np.where(labelsC==idxOpt[l])[0]
                                  for l,idx in enumerate(idxOpt)]).astype(np.int32)

        return self.idxOpt,self.idxOptRep


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

# def GABI_fit_wrapper(matrix,gb,send_end,k):
#     try:
#         print ("Starting thread: ", k)
#         gb.fit(matrix)
#         send_end.send(gb)
#         # send_end.send(gb.get_pickle())
#         print("thread: {} Terminated".format(k))

#     except KeyboardInterrupt:
#         send_end.send(gb)
#         # send_end.send(gb.get_pickle())
#         print ("Keyboard interrupt in process: ", k)


class Threading(Thread):
    def __init__(self, queue, k):
        Thread.__init__(self)
        self.queue = queue
        self.id = k
    def run(self):
        if not self.queue.empty():
            print('Thread %s Started' %self.id)
            item = self.queue.get()
            item['gb'].fit(item['matrix'])

        self.gb = item['gb']
        print('Thread %s Terminated' %self.id)


###########################################################
##           Auxilliary functions
###########################################################

def get_labels_split(labels,distmat,NClust):
    '''
        Split the labels in NG parts. The labels in the parts are selected such that
        their distance is the smallest, in order to maximize the overlap between the profiles,
        which improve the annotation algorithm.

        inputs:
            labels: (int numpy vector) corresponding  to the profiles membership
            distmat: Distance matrix between the profiles (full)
            NClust: Maximum number of different labels per parts (per default=6)

        ouputs:
        labels_c: labels in each parts
        labels_cS: labels in each parts, reindexed in order to be continuous
        Idxlabels_c: indexes of the profiles for each parts

    '''

    LM = (labels.max()+1) #Number of different labels
    # NG = int(np.ceil(LM/float(NClust)))

    #If the number of cell types is smalle than NClust, the matrix is not splitted
    if LM == NClust:
        labels_c = [labels]
        labels_cS = [labels]
        Idxlabels_c = [np.arange(len(labels))]
        return labels_c,labels_cS,Idxlabels_c

    # #Define the number of parts
    # while LM/NClust<2:
    #     NClust-=1

    #Linkage to get a chain of closest point
    Z = linkage(squareform(distmat), method='average', metric='precomputed')

    #From this list counts the number of times clusters are consecutive
    countsmat = np.zeros((LM,LM))
    for i in range(len(labels)-1):
        I = labels[leaves_list(Z)[i]]
        J = labels[leaves_list(Z)[i+1]]
        countsmat[I,J]+=1
    #From count matrix to distance matrix
    countsmat = (countsmat+countsmat.T)/2.
    # countsmat = countsmat/
    countsmat = np.exp(-countsmat)
    countsmat = countsmat - np.diag(np.diag(countsmat))
    #Linkage to get a chain of closest point amonf clusters
    Z = linkage(squareform(countsmat), method='average', metric='precomputed')

    idxNG = np.arange(0,LM,NClust)
    labels_c = [leaves_list(Z)[idxNG[k]:idxNG[k+1]] for k in range(len(idxNG)-1)]
    labels_c.append(leaves_list(Z)[LM-NClust:LM])

    NG = len(labels_c)

    print('labels')
    for i,l in enumerate(labels_c):
        print(i, l)
    #Get indexes and new labels values
    Idxlabels_c = [np.concatenate([np.where(labels==l)[0] for l in labels_c[k]]) for k in range(NG)]

    labels_c = [labels[idx] for idx in Idxlabels_c]

    labels_cS = [labels_c[k].copy() for k in range(NG)]
    #Change the labels to fit the split matrices
    for ng in range(NG):
        u,ind = np.unique(labels_c[ng],return_index=True)
        u = u[np.argsort(ind)]

        for i,l in enumerate(u):
            labels_cS[ng][labels_c[ng]==l] = i

    return labels_c,labels_cS,Idxlabels_c



def merge_split_matrices(matrix_c,labels_c):
    '''
        Return and matrix from a list of labels and matrices
    '''
    def unique_keep_order(labels):
        u, ind = np.unique(labels, return_index=True)
        labels = u[np.argsort(ind)]
        return labels

    labelsCT = np.concatenate([unique_keep_order(labels) for labels in labels_c])
    matrix = np.concatenate(matrix_c,axis=0)

    idx = np.argsort(labelsCT)
    labelsCT = labelsCT[idx]
    matrix = matrix[idx,:]

    #remove duplicates
    idxRm = []
    for i in range(labelsCT.shape[0]-1):
        if labelsCT[i]==labelsCT[i+1]:
            idxRm.append(i+1)

    if len(idxRm)>0:
        matrix = np.delete(matrix, idxRm, 0)
        labelsCT = np.delete(labelsCT, idxRm, 0)

    return matrix

def get_all_combination_low_CT(matrix):
    '''
        Get all combination of profiles present in matrix in the case the number of CT
        is lower than 128 (equivalent to long float)

        output:
            statemat: matrix of all the combination (profiles x combination)
            labels_site: postion of the combination
            counts: number of sites for each combination
    '''
    K_CT,N = matrix.shape

    pow2 = np.array([np.power(2,i,dtype=np.float32) for i in range(K_CT)])

    states2 = (matrix.T).dot(pow2)

    labelsU,idxU,counts = np.unique(states2,return_index=True,return_counts=True)

    statemat = matrix[:,idxU]

    #Build labels_site
    labels_site = np.zeros(matrix.shape[1],dtype=np.int32)
    for i,l in enumerate(labelsU):
        labels_site[states2==l] = i

    return statemat,labels_site,counts

def get_all_combination_binary(matrix):
    '''
        Get all  the combination of a matrix encoded in a bases 2 value
    '''
    pow2 = np.array([np.power(2,i,dtype=np.float32) for i in range(matrix.shape[0])])
    states2 = (matrix.T).dot(pow2)
    return state2


def get_all_combination(matrix):
    '''
        Get all combination of profiles present in matrix

        output:
            statemat: matrix of all the combination (profiles x combination)
            labels_site: postion of the combination
            counts: number of sites for each combination
    '''

    if matrix.shape[0]<128:
        return get_all_combination_low_CT(matrix)

    idx = np.arange(0,matrix.shape[0],127)
    if idx[-1]!=matrix.shape[0]:
        idx = np.append(idx,matrix.shape[0])

    MatComb = np.zeros((len(idx)-1,matrix.shape[1]),dtype=np.float128)
    Statemat_c = []
    for i in range(len(idx)-1):
        statemat_i,labels_site_i,_ = get_all_combination_low_CT(matrix[idx[i]:idx[i+1],:])
        MatComb[i,:] = labels_site_i
        Statemat_c.append(statemat_i)

    #get all the combinations reccursively
    idxComb = []
    Reccursive_comb(mat=MatComb,level=0,idxComb=idxComb,idx0=np.arange(MatComb.shape[1]))
    idxComb = np.array(idxComb)

    #get labels sites
    labels_site = np.ones(len(matrix[1]))*(-1)
    for i,idx in enumerate(idxComb):
        labels_site[idx] = i

    _,idxU,counts = np.unique(labels_site,return_index=True,return_counts=True)
    combmat = matrix[:,idxU]

    return combmat,labels_site,counts


def Reccursive_comb(mat,level,idxComb,idx0):
    '''
        mat: matrix which lines represents the combinasons of a group
        level: level of recursion
        idxComb: list of combinations positions returned by reference (output)
        idx0: Index of the original matrix
    '''
    idx = np.argsort(mat[level,:])
    mat = mat[:,idx]
    idx0 = idx0[idx]
    idxU = np.insert(np.append(np.where(np.diff(mat[level,:])!=0)[0]+1,mat.shape[1]),0,0)
    for i in range(len(idxU)-1):
        if level<(mat.shape[0]-1):
            Reccursive_comb(mat=mat[:,idxU[i]:idxU[i+1]],level=level+1,idxComb=idxComb,idx0=idx0[idxU[i]:idxU[i+1]])
        else:
            idxComb.append(idx0[idxU[i]:idxU[i+1]])

def comb2states(labels,combmat):
    '''
        Combination of cell types to combination of profiles.
    '''
    M = len(labels)
    NCT = int(labels.max() + 1)

    Rmat = np.zeros((M,NCT))
    for i in range(NCT):
        Rmat[labels==i,i] = 1
    print(Rmat)
    print(combmat)
    statemat = Rmat.dot(combmat)

    return statemat

def check_if_binary_matrix(matrix):
    #Define if the matrix is boolean or Discrete
    if not issparse(matrix):
        uniq = np.unique(matrix)
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
            Binary = False

    return Binary

###########################################################
##           Distance functions
###########################################################

def yule_distance(X):
    '''
        Yule  Distance
        X,Y = samples x dimensions
    '''

    c11 = np.dot(X,X.T).astype(np.float32)

    c01 = np.dot(1-X,X.T).astype(np.float32)

    c10 = np.dot(X,1-X.T).astype(np.float32)

    c00 = np.dot(1-X,1-X.T).astype(np.float32)

    yule = 2*(c10*c01)/((c00*c11) + (c10*c01))

    yule[np.isnan(yule)] = 1 #Nan append when there is only zeros in both samples

    return yule

def GetOptimalOrder(matrix):
    '''
        Get an otpimal order for the Annoation matrix and the names
        from a hierarchical clustering leaf index

        return:
            idxOpt: Optimal index
    '''
    dist = yule_distance(matrix)
    dist = (dist+dist.T)/2.
    dist = dist - np.diag(np.diag(dist))
    Z = linkage(squareform(dist), method='average', metric='precomputed')

    return leaves_list(Z).astype(np.int32)

