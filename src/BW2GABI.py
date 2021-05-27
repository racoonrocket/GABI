
import pyBigWig as bg
import re
import numpy as np
from scipy.sparse import csr_matrix
import time
def bigwig2gabi(path_to_all_bw,list_chr,binsize=10000):

    labels = []
    sparsedmatrix = []
    pickle.dump(sparsedmatrix, open('sparsed.p','wb+'))
    with open(path_to_all_bw) as file_to_all_bw:
        list_bw_path = file_to_all_bw.readlines()
    for element in list_bw_path:
        print(str(list_bw_path.index(element)) + "over" + str(len(list_bw_path)))
        beginNum = re.compile("[1-9]+")
        TemplateBWPath = re.compile("[/][^ç]+[\.]bw")
        try:
            matching = beginNum.match(element)
            labels.append(int(matching.group(0))-1)

        except:
            print("warning label is missing on this line")
            pass
        try:
            matchingpath = TemplateBWPath.search(element)
            print(matchingpath)
            OneBWPath = "/zfast1/haschka/data/H3K27me3"+matchingpath.group(0)
        except:
            print("warning path is missing on this line")
            continue
        #start to work on the BigWig file, binin and the sparse
        BigWig = bg.open(OneBWPath)  # open the big wig
        chrom_dict = BigWig.chroms() #open BIgWig chromosome dictionnary
        epigenomic_profile = []
        try:
            for element in list_chr:
                number_of_bins = int(chrom_dict[element]/binsize)
                chromvalues = BigWig.stats(element,1,chrom_dict[element],nBins=number_of_bins)
                chromvalues = [0 if value == None else value for value in chromvalues]
                chromvalues = [int(value) for value in chromvalues]
                epigenomic_profile.extend(chromvalues)
            sparsedmatrix = pickle.load(open('sparsed.p','rb'))
            sparsedmatrix.append(np.array(epigenomic_profile))
            pickle.dump(sparsedmatrix,open('sparsed.p','wb+'))
            sparsedmatrix=None
            print("lengh after bining" + str(len(epigenomic_profile)))
        except:
            labels.pop(37)
            print("PROBLEMO")
            continue
    #sparsedmatrix = csr_matrix(sparsedmatrix)
    sparsedmatrix = pickle.load(open('sparsed.p','rb'))
    return np.array(sparsedmatrix,dtype=np.bool), np.array(labels)






