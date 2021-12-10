import os
import sys

pid = os.getpid()
print('pid:%d' % (pid))

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

import torch
from utils import get_parser, setup_seed
from evalution import train, test, output


if __name__ == '__main__':
    print(sys.argv)
    parser = get_parser()
    args = parser.parse_args()
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.gpu:
        device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
        print("use cuda:"+args.gpu if torch.cuda.is_available() else "use cpu, cuda:"+args.gpu+" not available")
    else:
        device = torch.device("cpu")
        print("use cpu")
    
    SEED = args.random_seed
    setup_seed(SEED)

    if args.dataset == 'taobao':
        path = './data/taobao_data/'
        item_count = 1708531
        batch_size = 256
        seq_len = 50
        test_iter = 500
    elif args.dataset == 'book':
        path = './data/book_data/'
        item_count = 367983
        batch_size = 128
        seq_len = 20
        test_iter = 1000
    elif args.dataset == 'familyTV':
        path = './data/familyTV_data/'
        item_count = 867633
        batch_size = 256
        seq_len = 30
        test_iter = 1000
    
    train_file = path + args.dataset + '_train.txt'
    valid_file = path + args.dataset + '_valid.txt'
    test_file = path + args.dataset + '_test.txt'
    cate_file = path + args.dataset + '_item_cate.txt'
    dataset = args.dataset

    print("hidden_size:", args.hidden_size)
    print("interest_num:", args.interest_num)


    if args.p == 'train':
        train(device=device, train_file=train_file, valid_file=valid_file, test_file=test_file, 
                dataset=dataset, model_type=args.model_type, item_count=item_count, batch_size=batch_size, 
                lr=args.learning_rate, seq_len=seq_len, hidden_size=args.hidden_size, 
                interest_num=args.interest_num, topN=args.topN, max_iter=args.max_iter, test_iter=test_iter, 
                decay_step=args.lr_dc_step, lr_decay=args.lr_dc, patience=args.patience)
    elif args.p == 'test':
        test(device=device, test_file=test_file, cate_file=cate_file, dataset=dataset, model_type=args.model_type, item_count=item_count, 
                batch_size=batch_size, lr=args.learning_rate, seq_len=seq_len, hidden_size=args.hidden_size, 
                interest_num=args.interest_num, topN=args.topN, coef=args.coef)
    elif args.p == 'output':
        output(device=device, dataset=dataset, model_type=args.model_type, item_count=item_count, 
                batch_size=batch_size, lr=args.learning_rate, seq_len=seq_len, hidden_size=args.hidden_size, 
                interest_num=args.interest_num, topN=args.topN)
    else:
        print('do nothing...')
    
