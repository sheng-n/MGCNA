from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import scipy.sparse as sp


class Data_class(Dataset):

    def __init__(self, triple):
        self.entity1 = triple[:, 0]
        self.entity2 = triple[:, 1]
        self.label = triple[:, 2]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):

        return self.label[index], (self.entity1[index], self.entity2[index])


def load_data(args, test_ratio=0.2):
    """Read data from path, convert data into loader, return features and adjacency"""
    # read data
    print('Loading {0} seed{1} dataset...'.format(args.pos_sample, args.seed))  #
    positive = np.loadtxt(args.pos_sample, dtype=np.int64)   # pos_pair 8720, 2
    # print("postive:",positive.shape)

    #sample postive
    link_size = int(positive.shape[0])
    np.random.seed(args.seed)
    np.random.shuffle(positive)
    positive = positive[:link_size]

    # sample negative
    negative_all = np.loadtxt(args.neg_sample, dtype=np.int64)  # neg_pair 237448, 2
    # print("negative:", negative_all.shape)
    np.random.shuffle(negative_all)
    negative = np.asarray(negative_all[:positive.shape[0]])  # equal negative samples to positive samples
    print("positive examples: %d, negative examples: %d." % (positive.shape[0], negative.shape[0]))  #


    test_size = int(test_ratio * positive.shape[0]) # test number  1744
    print("test_number:",test_size)

    positive = np.concatenate([positive, np.ones(positive.shape[0], dtype=np.int64).reshape(positive.shape[0], 1)], axis=1) # 8720,3
    negative = np.concatenate([negative, np.zeros(negative.shape[0], dtype=np.int64).reshape(negative.shape[0], 1)], axis=1) # 8720,3
    # negative_all = np.concatenate([negative_all, np.zeros(negative_all.shape[0], dtype=np.int64).reshape(negative_all.shape[0], 1)], axis=1) #90537,3
    print("positive_negative: ",positive.shape, negative.shape)  # (8720, 3) (8720, 3)

    train_data = np.vstack((positive[: -test_size], negative[: -test_size])) # 13952, 2
    test_data = np.vstack((positive[-test_size:], negative[-test_size:]))    # 3488, 2
    print("data: ",train_data.shape,test_data.shape)

    # build data loader
    params = {'batch_size': args.batch, 'shuffle': True, 'num_workers': args.workers, 'drop_last': True}

    training_set = Data_class(train_data)
    train_loader = DataLoader(training_set, **params)
    print(train_loader)

    test_set = Data_class(test_data)
    test_loader = DataLoader(test_set, **params)
    print(test_loader)


    # build multiple view
    dataset = dict()
    "miRNA sequence sim"
    mm_s_matrix = np.loadtxt('data/miRNA_seq_sim.txt')
    mm_s_edge_index = mm_s_matrix.nonzero() # construct view non_zero index
    mm_s_edge_index = torch.tensor(np.vstack((mm_s_edge_index[0], mm_s_edge_index[1])), dtype=torch.long) # edge index
    dataset['mm_s'] = {'data_matrix': mm_s_matrix, 'edges': mm_s_edge_index}

    "miRNA functional sim based miRNA-gene"
    # mm_g_matrix = np.loadtxt("data/miRNA_gau_sim_g.txt")
    # mm_g_matrix = np.loadtxt("data/miRNA_binary_gau_sim.txt")
    # mm_g_matrix = np.loadtxt("data/miRNA_binary_gau_sim_g_0.5.txt")
    # mm_g_edge_index = mm_g_matrix.nonzero() # construct view non_zero index
    # mm_g_edge_index = torch.tensor(np.vstack((mm_g_edge_index[0], mm_g_edge_index[1])), dtype=torch.long) # edge index
    # dataset['mm_g'] = {'data_matrix': mm_g_matrix, 'edges': mm_g_edge_index}

    "miRNA functional sim based miRNA-drug"
    mm_r_matrix = np.loadtxt("data/miRNA_gau_sim_r.txt")
    mm_r_edge_index = mm_r_matrix.nonzero() # construct view non_zero index
    mm_r_edge_index = torch.tensor(np.vstack((mm_r_edge_index[0], mm_r_edge_index[1])), dtype=torch.long) # edge index
    dataset['mm_r'] = {'data_matrix': mm_r_matrix, 'edges': mm_r_edge_index}


    "drug fingerprint sim"
    dd_f_matrix = np.loadtxt("data/drug_smiles_sim.txt")
    dd_f_edge_index = dd_f_matrix.nonzero() # construct view non_zero index
    dd_f_edge_index = torch.tensor(np.vstack((dd_f_edge_index[0], dd_f_edge_index[1])), dtype=torch.long) # edge index
    dataset['dd_f'] = {'data_matrix': dd_f_matrix, 'edges': dd_f_edge_index}

    "drug gene sim"
    dd_g_matrix = np.loadtxt("data/drug_gau_sim_g.txt")
    dd_g_edge_index = dd_g_matrix.nonzero() # construct view non_zero index
    dd_g_edge_index = torch.tensor(np.vstack((dd_g_edge_index[0], dd_g_edge_index[1])), dtype=torch.long) # edge index
    dataset['dd_g'] = {'data_matrix': dd_g_matrix, 'edges': dd_g_edge_index}

    "drug miRNA sim"
    dd_m_matrix = np.loadtxt("data/drug_gau_sim_m.txt")
    dd_m_edge_index = dd_m_matrix.nonzero() # construct view non_zero index
    dd_m_edge_index = torch.tensor(np.vstack((dd_m_edge_index[0], dd_m_edge_index[1])), dtype=torch.long) # edge index
    dataset['dd_m'] = {'data_matrix': dd_m_matrix, 'edges': dd_m_edge_index}

    print('Loading finished!')
    return dataset, train_loader, test_loader  # view, train/test loader


# from parms_setting import settings
# args = settings()  #读取parms_setting中的自定义参数
# load_data(args)