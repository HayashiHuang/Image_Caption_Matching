import argparse
import torch
from dataloader import load_dataset
from net import Net
from trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_dir', type=str, default='./data/',
                        help='path for text dataset')
    parser.add_argument('--img_dir', type=str, default='./data/Flickr30k_images/',
                        help='path for image dataset')
    parser.add_argument('--work_dir', type=str, default='./result/',
                        help='path for training result')
    parser.add_argument('--batch_size', type=int, default=96,
                        help='input batch size(default: 64)')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                        help='split ratio of train and validation from the training set')
    parser.add_argument('--max_size', type=int, default=5000,
                        help='max size of vocab dic(default: 5000)')
    parser.add_argument('--min_freq', type=int, default=2,
                        help='The minimum frequency needed to include a token in the vocabulary(default: 2)')
    parser.add_argument('--lstm_hidden', type=int, default=600,
                        help='size of hidden feature of the LSMT (default: 100)')
    parser.add_argument('--lstm_layer', type=int, default=2,
                        help='layer of the LSMT (default: 1)')
    parser.add_argument('--lstm_dropout', type=float, default=0.3,
                        help='dropout rate of the LSMT (default: 0.3)')
    parser.add_argument('--feat_dim', type=int, default=600,
                        help='size of output feature(default: 1)')
    parser.add_argument('--epoch', type=int, default=120,
                        help='training epoch(default: 10)')
    parser.add_argument('--text_epoch', type=str, default=40,
                        help='text model training epoch(default: 5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='The id of GPU (default: 0)')

    # lr schedule
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--warm_up_epoch', type=int, default=5,
                        help='Warm up epoch (default: 2)')
    parser.add_argument('--init_lr', type=float, default=0.0004,
                        help='Initial learning rate, useful only when warm up epoch > 0 (default: 0.0002)')
    parser.add_argument('--decay_factor', type=float, default=0.95,
                        help='Factor for lr exponential decay (default: 0.95)')

    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    
    print('Preparing data.')
    train_loader, val_img_loader, _, val_caption_loader, _, vocab_table = load_dataset(args)

    print('Constructing model.')
    model = Net(args, vocab_table.vocab)
    # if torch.cuda.device_count() > 1:
    #     print('use {} GPUs for training...'.format(torch.cuda.device_count()))
    #     model = torch.nn.DataParallel(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = NovoGrad(model.parameters(), lr=args.lr)

    print('===== TRAIN =====')
    text_trainer = Trainer(train_loader, model, optimizer, args, mode='text')
    model = text_trainer.train()
    mix_trainer = Trainer(train_loader, model, optimizer, args, mode='mix')
    mix_trainer.train()
