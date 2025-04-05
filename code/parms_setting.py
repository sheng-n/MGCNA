import argparse

"You can modify according to your own needs"

def settings():
    parser = argparse.ArgumentParser()

    # public parameters
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed. Default is 0.')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')

    parser.add_argument('--workers', type=int, default=0,
                        help='Number of parallel workers. Default is 0.')

    parser.add_argument('--pos_sample', default="data/pos.edgelist",  # read positive data
                        help='Positive data path.')

    parser.add_argument('--neg_sample', default="data/neg.edgelist",  # read negative data
                        help='Negative data path.')

    # Training settings
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Initial learning rate. Default is 5e-4.')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate. Default is 0.5.')

    parser.add_argument('--weight_decay', default=5e-4,
                        help='Weight decay (L2 loss on parameters) Default is 5e-4.')

    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size. Default is 32.')

    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of epochs to train. Default is 60.')


    # model parameter setting
    parser.add_argument("--miRNA_number", type=int, default=1578,
                        help="miRNA number. Default is 1579.")

    parser.add_argument("--drug_number", type=int, default=156,
                        help="disease number. Default is 156.")

    parser.add_argument('--dimensions', type=int, default=512,
                        help='dimensions of feature d. Default is 512')

    parser.add_argument('--hidden1', default=256,
                        help='Embedding dimension of encoder layer 1 for MGCNA. Default is d/2.')

    parser.add_argument('--hidden2', default=128,
                        help='Embedding dimension of encoder layer 2 for MGCNA. Default is d/4.')

    parser.add_argument('--decoder1', default=256,
                        help='Embedding dimension of decoder layer 1 for MGCNA. Default is 512.')

    args = parser.parse_args()

    return args
