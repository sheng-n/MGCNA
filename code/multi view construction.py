import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys

"calculated k-mer feature"
def k_mer(seq):
    def get_1mer(seq):
        A_count = seq.count("A")
        T_count = seq.count("T")
        C_count = seq.count("C")
        G_count = seq.count("G")
        return [A_count/len(seq), T_count/len(seq), C_count/len(seq), G_count/len(seq)]

    def get_2mer(seq):
        res_dict = {}
        for x in "ATCG":
            for y in "ATCG":
                k = x + y
                res_dict[k] = 0
                # print(k)
        # print(res_dict)
        i = 0
        while i + 2 < len(seq):
            k = seq[i:i + 2]
            i = i + 1
            res_dict[k] = res_dict[k] + 1

        return [x/len(seq) for x in list(res_dict.values())]

    def get_3mer(seq):
        res_dict = {}
        for x in "ATCG":
            for y in "ATCG":
                for z in "ATCG":
                    k = x + y + z
                    res_dict[k] = 0
        i = 0
        while i + 3 < len(seq):
            k = seq[i:i + 3]
            i = i + 1
            res_dict[k] = res_dict[k] + 1
        return [x/len(seq) for x in list(res_dict.values())]

    def get_4mer(seq):
        res_dict = {}
        for x in "ATCG":
            for y in "ATCG":
                for z in "ATCG":
                    for p in "ATCG":
                        k = x + y + z + p
                        res_dict[k] = 0
        i = 0
        while i + 4 < len(seq):
            k = seq[i:i + 4]
            i = i + 1
            res_dict[k] = res_dict[k] + 1
        return [x/len(seq) for x in list(res_dict.values())]
    # print(get_1mer(seq))
    # return get_1mer(seq) + get_2mer(seq) + get_3mer(seq) + get_4mer(seq)
    return get_1mer(seq) + get_2mer(seq) + get_3mer(seq)


def miRNA_mer():
    df = pd.read_excel('data/miRNA-sequences.xlsx', usecols=["miRNA_name",'Sequence'])
    df['Sequence'] = df['Sequence'].str.replace('U', 'T')
    miRNA_dict = dict(zip(df['miRNA_name'], df['Sequence']))

    result = []
    for name, seq in miRNA_dict.items():
        RNA_mer = k_mer(seq)
        result.append(RNA_mer)
        print(RNA_mer)
    # print(len(result))
    return result


def cosine_similarity(features):

    sim_matrix = np.zeros((len(features), len(features)))

    for i in range(len(features)):
        for j in range(i, len(features)):
            v1 = features[i]
            v2 = features[j]

            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    print(sim_matrix,sim_matrix.shape)
    return sim_matrix


'''calculate gaussianKernel_sim'''
def calculate_kernel_bandwidth(A):
    IP_0 = 0
    for i in range(A.shape[0]):
        IP = np.square(np.linalg.norm(A[i]))
        # print(IP)
        IP_0 += IP
    lambd = 1/((1/A.shape[0]) * IP_0)
    return lambd

def calculate_GaussianKernel_sim(A):
    kernel_bandwidth = calculate_kernel_bandwidth(A)
    gauss_kernel_sim = np.zeros((A.shape[0],A.shape[0]))
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            gaussianKernel = np.exp(-kernel_bandwidth * np.square(np.linalg.norm(A[i] - A[j])))
            gauss_kernel_sim[i][j] = gaussianKernel
            # print("gau",gauss_kernel_sim)

    return gauss_kernel_sim

def calculate_molecular_similarity(excel_path):
    df = pd.read_excel(excel_path)
    mols = []

    for idx, smiles in enumerate(df['smiles']):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            mols.append(mol)

    fingerprints = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]

    n = len(fingerprints)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            # Tanimoto
            similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    return similarity_matrix


def threshold_similarity_matrix(similarity_matrix, threshold):

    binary_matrix = (similarity_matrix > threshold).astype(int)

    np.fill_diagonal(binary_matrix, 1)

    return binary_matrix


if __name__ == '__main__':

    "calculated miRNA sequence similarity"
    miRNA_3mer_features = miRNA_mer()
    mi_seq_sim = cosine_similarity(miRNA_3mer_features)
    mi_seq_sim = threshold_similarity_matrix(mi_seq_sim, threshold = 0.8)
    np.savetxt("data/miRNA_seq_sim.txt",mi_seq_sim)

    # "calculate miRNA gaussianKernel sim"
    # miRNA_gene = pd.read_excel("data/miRNA-gene-matrix.xlsx",header=0,index_col=0)
    # # print(miRNA_gene[0])
    # mi_gau_sim_g = calculate_GaussianKernel_sim(miRNA_gene.values)  #based miRNA-gene
    # mi_gau_sim_g = threshold_similarity_matrix(mi_gau_sim_g, threshold = 0.8)
    # np.savetxt("data/miRNA_gau_sim_g.txt",mi_gau_sim_g)

    "calculate drug smiles sim"
    drug_smiles_sim = calculate_molecular_similarity('data/drug-smiles.xlsx')
    drug_smiles_sim = threshold_similarity_matrix(drug_smiles_sim, threshold = 0.5)
    np.savetxt("data/drug_smiles_sim.txt", drug_smiles_sim)


    "calculate drug gaussianKernel sim"
    drug_gene = pd.read_excel("data/drug-gene-matrix.xlsx",header=0,index_col=0)
    # print(miRNA_gene[0])
    drug_gau_sim_g = calculate_GaussianKernel_sim(drug_gene.values)  #based drug-gene
    drug_gau_sim_g = threshold_similarity_matrix(drug_gau_sim_g, threshold = 0.5)
    np.savetxt("data/drug_gau_sim_g.txt", drug_gau_sim_g)


    "calculate miRNA/drug gaussianKernel sim  based miRNA-drug"
    miRNA_drug = pd.read_excel("data/miRNA-drug-matrix.xlsx",header=0,index_col=0)
    # print(miRNA_gene[0])
    mi_gau_sim_r = calculate_GaussianKernel_sim(miRNA_drug.values)  #based miRNA-drug
    mi_gau_sim_r = threshold_similarity_matrix(mi_gau_sim_r, threshold = 0.6)
    np.savetxt("data/miRNA_gau_sim_r.txt", mi_gau_sim_r)


    drug_gau_sim_m = calculate_GaussianKernel_sim(miRNA_drug.values.T)  #based miRNA-drug
    drug_gau_sim_m = threshold_similarity_matrix(drug_gau_sim_m, threshold = 0.5)
    np.savetxt("data/drug_gau_sim_m.txt", drug_gau_sim_m)






