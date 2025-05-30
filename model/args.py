import argparse

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--n_head',
                        type=int, default=12,  # 改为 12，因为命令行中的默认值是 12
                        help='GPT number of heads')
    parser.add_argument('--fold',
                        type=int, default=0,
                        help='number of folds for fine tuning')
    parser.add_argument('--n_layer',
                        type=int, default=12,  # 12 是命令行中的默认值
                        help='GPT number of layers')
    parser.add_argument('--d_dropout',
                        type=float, default=0.1,  # 默认值 0.1
                        help='Decoder layers dropout')
    parser.add_argument('--n_embd',
                        type=int, default=768,  # 默认值 768
                        help='Latent vector dimensionality')
    parser.add_argument('--fc_h',
                        type=int, default=512,
                        help='Fully connected hidden dimensionality')

    # Train
    parser.add_argument('--n_batch',
                        type=int, default=64,  # 默认值 64
                        help='Batch size')
    parser.add_argument('--from_scratch',
                        action='store_true', default=False,
                        help='train on qm9 from scratch')
    parser.add_argument('--checkpoint_every',
                        type=int, default=100,  # 默认值 100
                        help='save checkpoint every x iterations')
    parser.add_argument('--lr_start',
                        type=float, default=3e-5,  # 默认值 3e-5
                        help='Initial lr value')
    parser.add_argument('--lr_multiplier',
                        type=float, default=1,  # 默认值 1
                        help='lr weight multiplier')
    parser.add_argument('--n_jobs',
                        type=int, default=1,  # 默认值 1
                        help='Number of threads')
    parser.add_argument('--device',
                        type=str, default='cuda',  # 默认值 'cuda'
                        help='Device to run: "cpu" or "cuda:<device number>"')
    parser.add_argument('--seed',
                        type=int, default=12345,  # 默认值 12345
                        help='Seed')

    parser.add_argument('--seed_path',
                        type=str, default='',  # 默认值为空
                        help='path to trainer file to continue training')

    parser.add_argument('--num_feats',
                        type=int, required=False, default=32,  # 默认值 32
                        help='number of random features for FAVOR+')
    parser.add_argument('--max_epochs',
                        type=int, required=False, default=1000,  # 默认值 1000
                        help='max number of epochs')

    # Additional parameters
    parser.add_argument('--mode',
                        type=str, default='avg',  # 默认值 'avg'
                        help='type of pooling to use')
    parser.add_argument('--train_dataset_length', type=int, default=None, required=False)
    parser.add_argument('--eval_dataset_length', type=int, default=None, required=False)
    parser.add_argument('--desc_skip_connection', type=bool, default=False, required=False)
    parser.add_argument('--num_workers', type=int, default=8, required=False)  # 默认值 8
    parser.add_argument('--dropout', type=float, default=0.1, required=False)  # 默认值 0.1
    parser.add_argument('--dims', type=int, nargs="*", default=[768, 768, 768, 1], required=False)  # 默认值 [768, 768, 768, 1]
    parser.add_argument('--smiles_embedding',
                        type=str,
                        default='/dccstor/medscan7/smallmolecule/runs/ba-predictor/small-data/embeddings/protein/ba_embeddings_tanh_512_2986138_2.pt',
                        help='Path to SMILES embeddings')
    parser.add_argument('--aug', type=int, required=False)
    parser.add_argument('--num_classes', type=int, required=False)
    parser.add_argument('--dataset_name', type=str, required=False, default='not_specify')  # 默认值为命令行中的 dataset_name
    parser.add_argument('--measure_name', type=str, required=False, default='ccs')  # 默认值 'Na'
    parser.add_argument('--checkpoints_folder', type=str, required=True, help='Folder to store checkpoints')
    parser.add_argument('--checkpoint_root', type=str, required=False)

    parser.add_argument('--data_root',
                        type=str,
                        required=False,
                        default='../data/my_dataset',  # 默认值为命令行中的 data_root
                        help='Root directory for dataset')
    
    parser.add_argument('--batch_size', type=int, default=64)  # 默认值 64
    parser.add_argument('--project_name', type=str, default='not_specify')  # 默认值 not_specify
    parser.add_argument('--type', type=str, default='attention')  # 默认值 attention
    parser.add_argument('--adduct_num', type=int, default=3)  # 默认值 attention
    parser.add_argument('--ecfp_num', type=int, default=0)  # 默认值 attention
    return parser

def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    return args
