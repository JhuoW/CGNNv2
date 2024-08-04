
import torch

def prepare_train(args, model):
    loss_func = torch.nn.CrossEntropyLoss()
    if args.model == 'GCNII':
        optimizer = torch.optim.Adam([
            dict(params=model.convs.parameters(), weight_decay=args.conv_weight_decay),
            dict(params=model.lins.parameters(), weight_decay=args.lin_weight_decay)
                            ], lr=args.lr)
    else:      
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    return optimizer, loss_func
