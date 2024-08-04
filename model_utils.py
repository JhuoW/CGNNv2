
from model.CGNN import CGNN

def get_model(args, num_feats, num_cls):

    model = CGNN(in_dim = num_feats,
                hid_dim = args.hidden_dim,
                n_cls = num_cls,
                num_layers = args.num_layers,
                dropout = args.dropout,
                jk = args.jk,
                norm = args.normalize).cuda()
    return model