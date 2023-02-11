import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_m',
                        default=0.5,
                        type=float)
    parser.add_argument('--s',
                        default=60,
                        type=float)

    parser.add_argument('--cell_line',
                        dest='cell_line',
                        type=str,
                        default="HUVEC")

    parser.add_argument('--device',
                        default='cuda:0')

    parser.add_argument('--lr',
                        default=5e-6,
                        type=float)


    parser.add_argument('--k',
                        default=3,
                        type=int)

    parser.add_argument('--embed_num',
                        default=16,
                        type=int)

    embed_num = parser.parse_known_args()[0].embed_num
    parser.add_argument('--hidden_dim',
                        default=embed_num * 2,
                        type=int)

    parser.add_argument('--num_layers',
                        default=1,
                        type=int)

    cell_line = parser.parse_known_args()[0].cell_line

    parser.add_argument('--cnn_kernel_size',
                        default=3,
                        type=int)
    parser.add_argument('--dim',
                        default=5568,
                        type=int)

    parser.add_argument('--batchsize',
                        default=64,
                        type=int)

    parser.add_argument('--ImbalancedDatasetSampler',
                        default=False,
                        type=bool)

    parser.add_argument('--epochs',
                        default=100,
                        type=int)



    parser.add_argument('--weight_decay',
                        default=1e-8,
                        type=float)

    parser.add_argument('--train_CNRCI',
                        default='./CNRCI/CNRCI_train_data_source/transcripts_type/' + cell_line + '/lncRNA.csv')
    parser.add_argument('--dev_CNRCI',
                        default='./CNRCI/CNRCI_dev_data_source/transcripts_type/' + cell_line + '/lncRNA.csv')
    parser.add_argument('--test_CNRCI',
                        default='./CNRCI/CNRCI_test_data_source/transcripts_type/' + cell_line + '/lncRNA.csv')

    parser.add_argument('--save',
                        default='checkpoints/')

    parser.add_argument('--train_dataset',
                        default='./feature_fusion/train/' + cell_line + '/set1.csv')
    parser.add_argument('--dev_dataset',
                        default='./feature_fusion/dev/' + cell_line + '/set1.csv')
    parser.add_argument('--test_dataset',
                        default='./feature_fusion/test/' + cell_line + '/set1.csv')

    parser.add_argument('--test_batchsize',
                        default=256,
                        type=int)

    parser.add_argument('--cnn_out_channels',
                        default=16,
                        type=int)
    parser.add_argument('--cnn_stride',
                        default=2,
                        type=int)
    parser.add_argument('--kernel_size',
                        default=12,
                        type=int)

    parser.add_argument('--dropout',
                        default=0.5,
                        type=float)
    parser.add_argument('--dropout_LC',
                        default=0.3,
                        type=float)
    parser.add_argument('--freeze_embed',
                        default=True,
                        type=bool)

    parser.add_argument('--stride',
                        default=3,
                        type=int)




    args = parser.parse_known_args()[0]
    return args
