import torch
import argparse
from utils import *
from dataset import *
from tqdm import tqdm
from model_utils import get_model
from prepare import *
import os.path as osp
import numpy as np

def train(model, data, optimizer, loss_func, train_mask, C = None):
    model.train()
    edge_index = data.edge_index.cuda()
    x = data.x.cuda()
    y = data.y.cuda()
    if args.model == 'CGNN':
        C = C.cuda()
    train_mask = train_mask.cuda()
    optimizer.zero_grad()
    if args.model == 'CGNN':
        logits = model(x, edge_index, C)
    else:
        logits = model(x, edge_index)
    loss = loss_func(logits[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def evaluate(model, data, mask, loss_func, C = None):
    model.eval()
    edge_index = data.edge_index.cuda()
    x = data.x.cuda()
    y = data.y.cuda()
    if args.model == 'CGNN':
        C = C.cuda()
    mask = mask.cuda()
    if args.model == 'CGNN':
        logits = model(x, edge_index, C)
    else:
        logits = model(x, edge_index)
    eval_loss = loss_func(logits[mask], y[mask])
    acc = accuracy(logits[mask], y[mask])

    return float(eval_loss), acc

def run(args):
    dataset, data = get_dataset(args.dataset, args.dataset_directory, args.self_loops, args.undirected)
    C_sim = None
    if args.model == 'CGNN':
        if not osp.exists(osp.join('R', f'{args.dataset}.pt')):
            C_sim = get_CT(args, data)
        else:
            C_sim = torch.load(osp.join('R', f'{args.dataset}.pt'))
    val_accs, test_accs = [], []
    num_feats = data.num_features
    num_cls   = dataset.num_classes
    for run_id in range(args.num_runs):
        print("\t\t%d th Run" % run_id)
        # if run_id == 6 or run_id == 8:
        #     continue
        data, train_mask, val_mask, test_mask = get_split(args.dataset, data, run_id)


        best_val_acc = 0
        report_test_acc = 0
        best_val_epoch = 0
        patience_cnt = 0

        model = get_model(args, num_feats=num_feats, num_cls= num_cls)
        # optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
        # loss_func = torch.nn.CrossEntropyLoss()
        optimizer, loss_func = prepare_train(args, model)
        pbar = tqdm(range(args.num_epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')


        for epoch in pbar:
            loss = train(model, data, optimizer, loss_func, train_mask, C_sim)
            val_loss, val_acc = evaluate(model, data, val_mask, loss_func, C_sim)
            test_loss, test_acc = evaluate(model, data, test_mask, loss_func, C_sim)
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_val_epoch = epoch
                report_test_acc = test_acc
                patience_cnt = 0
            else:
                patience_cnt        += 1
            if args.patience >0 and patience_cnt >= args.patience:
                break
            postfix_str = "<Epo %d> [Train Loss] %.4f [Val Acc] %.4f <Best Epo %d> [Best Val Acc] %.4f [Test Acc] %.4f ([Report Test Acc] %.4f)" %(
                           epoch,      loss,             val_acc.item(), best_val_epoch, best_val_acc.item(), test_acc.item(), report_test_acc.item()
            )
            pbar.set_postfix_str(postfix_str)
        test_accs.append(report_test_acc.item())
        val_accs.append(best_val_acc.item())
    print(f"Test Acc: {np.mean(test_accs)} +- {np.std(test_accs)}")
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of dataset", default="squirrel") # chameleon, squirrel, photo, citeseer, cora_ml
    parser.add_argument("--dataset_directory", type=str, help="Directory to save datasets", default="dataset")
    parser.add_argument('--gpu_id', type = int, default= 0) 
    parser.add_argument('--num_runs', type = int, default=10)
    parser.add_argument('--model', type = str, default = 'CGNN')
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)

    args = config2args(args, args.dataset)
    run(args)