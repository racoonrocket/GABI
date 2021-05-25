import pyBigWig as bg
import random as rd

def create_bw(name,chr_list,len_list):
    """
    create a bigwiggle with random binary values
    """
    file = bg.open(name+".bw","w")
    header = [(chr_list[i],len_list[i]+1) for i in range(len(chr_list))]
    print(header)
    file.addHeader(header)
    for i in range(len(chr_list)):
        valeurs = [rd.randint(0,1) for i in range(len_list[i])]
        places = [k for k in range(len_list[i]+1)]
        ends = places + [places[-1]+1]
        chrome = [chr_list[i]]*len_list[i]
    file.addEntries(chrome,places,values=valeurs,span=1)

#liste = ['chr1','chr2','chr3']
#longueurs = [10000,20000,300478]
#reate_bw("first",liste,longueurs)

def create_wiggle(name,longueur):
    """
    create a wiggle with random binary values
    """
    with open(name+".wig","w+") as file:
        chaine = 'variableStep chrom=chr3\n'
        #chaine += 'variableStep chrm=chr3 \n'
        for i in range(1,longueur):
            chaine += str(i)  + ' '  + str(rd.randint(0,1)) + '\n'
        print(chaine)
        file.write(chaine)
    with open(name+"size.txt","w+") as file:
        file.write("chr3 " + str(longueur))


#four = ['slip','slap','mic','mac']
#for element in four:
#pypybigwig
# create_wiggle(element,20000)