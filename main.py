import argparse
import torch.nn as nn
import torch
import numpy as np
from torch.backends import cudnn
import random
import time

from utils.metrics import OneError, Coverage, HammingLoss, RankingLoss, AveragePrecision
from utils.models import linear, mlp
from utils.utils_algo import adjust_learning_rate, predict, exel_loss, align_loss, intra_loss
from utils.utils_data import choose

parser = argparse.ArgumentParser(description='PyTorch implementation')
parser.add_argument('--dataset', default='scene', type=str,
                    choices=['scene',  'YeastBP'],help='dataset name')
parser.add_argument('--num-class', default=6, type=int, help='number of classes')
parser.add_argument('--input-dim', default=294, type=int, help='number of features')
parser.add_argument('--fold', default=9, type=int, help='fold-th fold of 10-cross fold')
parser.add_argument('--model', default="linear", type=str, choices=['linear', 'mlp'])
parser.add_argument('--hidden-dim', default=512, type=int, help='hidden dim for mlp')
parser.add_argument('--dropout', default=0.2, type=float, help='dropout for mlp')
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--a', default=1.0, type=float, help='weight alpha for hard pml loss')
parser.add_argument('--b', default=1.0, type=float, help='weight beta for base loss')
parser.add_argument('--wd', default=1e-3, type=float, help='weight decay')
parser.add_argument('--batch_size', default=256, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--schedule', default=[100, 150], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('--the', default=0.8, type=float, help='prediction threshold. ')
parser.add_argument('--nl', default=2.0, type=float, choices=[1.0, 1.5, 2.0, 2.5],
                    help='noisy level for label files')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    print(args)

    cudnn.benchmark = True

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # make data
    train_loader, test_loader, args.num_class, args.input_dim = choose(args)

    # choose model
    if args.model == "linear":
        model = linear(input_dim=args.input_dim, output_dim=args.num_class)
    elif args.model == "mlp":
        model = mlp(input_dim=args.input_dim, output_dim=args.num_class,
                    hidden_dim=args.hidden_dim, dropout=args.dropout)
    else:
        raise ValueError("Unsupported model: {}".format(args.model))

    model = model.to(device)

    # set optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9,
                                weight_decay=args.wd)

    print("start training")

    best_av = 0
    save_table = np.zeros(shape=(args.epochs, 7))
    train_start_ts = time.time()

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        train_loss = train(train_loader, model, optimizer, args)
        t_hamm, t_one_error, t_converage, t_rank, t_av_pre = validate(test_loader, model, args)
        print("Epoch:{ep}, Tr_loss:{tr}, T_hamm:{T_hamm}, T_one_error:{T_one_error}, T_con:{T_con}, "
              "T_rank:{T_rank}, T_av:{T_av}".format(ep=epoch, tr=train_loss, T_hamm=t_hamm, T_one_error=t_one_error,
                                                    T_con=t_converage, T_rank=t_rank, T_av=t_av_pre))
        save_table[epoch, :] = epoch + 1, train_loss, t_hamm, t_one_error, t_converage, t_rank, t_av_pre

        np.savetxt("result/{ds}_nl{nl}_{M}_lr{lr}_wd{wd}_fold{fd}.csv".format(ds=args.dataset,
                                                                                  M=args.model, lr=args.lr, wd=args.wd,
                                                                                  nl=args.nl, fd=args.fold), save_table,
                   delimiter=',', fmt='%1.4f')
        # save model
        if t_av_pre > best_av:
            best_av = t_av_pre
            torch.save(model.state_dict(), "experiment/{ds}_nl{nl}_{M}_lr{lr}_wd{wd}_fold{fd}_best_model.tar".format(
                ds=args.dataset, M=args.model, lr=args.lr, wd=args.wd, nl=args.nl, fd=args.fold))

    train_end_ts = time.time()
    train_seconds = train_end_ts - train_start_ts
    print("Training time: {:.4f} seconds ({:.4f} minutes)".format(train_seconds, train_seconds / 60.0))

    time_file = "result/{ds}_nl{nl}_{M}_lr{lr}_wd{wd}_fold{fd}_time.csv".format(
        ds=args.dataset, M=args.model, lr=args.lr, wd=args.wd, nl=args.nl, fd=args.fold)
    with open(time_file, "w", encoding="utf-8") as f:
        f.write("start_timestamp,end_timestamp,train_seconds,train_minutes\n")
        f.write("{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
            train_start_ts, train_end_ts, train_seconds, train_seconds / 60.0
        ))



def train(train_loader, model, optimizer, args):
    model.train()
    train_loss = 0
    for i, (images, _, com_labels, pml_labels,index) in enumerate(train_loader):
        images, com_labels ,pml_labels= images.to(device), com_labels.to(device),pml_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        noncandidate_loss = exel_loss(outputs, com_labels)
        
        soft_loss, new_pml_labels = intra_loss(outputs, pml_labels)
        hard_loss = align_loss(outputs, new_pml_labels)
        pml_loss = soft_loss + args.a * hard_loss

        loss = args.b * noncandidate_loss + pml_loss
        
        loss.backward()
        optimizer.step()

        train_loss = train_loss + loss.item()

        for j, k in enumerate(index):
            train_loader.dataset.pml_labels[int(k), :] = new_pml_labels[j, :].detach().cpu().numpy()

    return train_loss / len(train_loader)




# test the results
def validate(test_loader, model, args):
    with torch.no_grad():
        model.eval()
        sig = nn.Sigmoid()
        for data, targets, _, _, _ in test_loader:
            images, targets = data.to(device), targets.to(device)
            output = model(images)
            pre_output = sig(output)
            pre_label = predict(output, args.the)

    t_one_error = OneError(pre_output, targets)
    t_converage = Coverage(pre_output, targets)
    t_hamm = HammingLoss(pre_label, targets)
    t_rank = RankingLoss(pre_output, targets)
    t_av_pre = AveragePrecision(pre_output, targets)

    return t_hamm, t_one_error, t_converage, t_rank, t_av_pre


if __name__ == '__main__':
    dataset=["scene","YeastBP"]
    args = parser.parse_args()
    default_lr = args.lr
    default_a = args.a
    default_b = args.b
    dataset_cfg = {
        "scene": {"lr": 0.01, "a": 10, "b": 2},
        "YeastBP": {"lr": 0.1, "a": 10, "b": 0.3}
    }

    for ds in dataset:
        args.dataset = ds
        cfg = dataset_cfg.get(ds, {})
        args.lr = cfg.get("lr", default_lr)
        args.a = cfg.get("a", default_a)
        args.b = cfg.get("b", default_b)

        for fd in range(10):
            args.fold = fd
            print(
                "Data:{ds}, model:{model}, lr:{lr}, a:{a}, b:{b}, wd:{wd}, nl:{nl}, fold:{fd}".format(
                    ds=args.dataset, model=args.model, lr=args.lr, a=args.a, b=args.b, wd=args.wd, nl=args.nl, fd=args.fold))
            main()
