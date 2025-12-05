import argparse
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np


def main(args=None):
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='FEDformer for BTC price prediction')

    # ===== BASIC CONFIG =====
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--task_id', type=str, default='BTC')
    parser.add_argument('--model', type=str, default='FEDformer')

    # ===== FEDformer-specific =====
    parser.add_argument('--version', type=str, default='Fourier')
    parser.add_argument('--mode_select', type=str, default='random')
    parser.add_argument('--modes', type=int, default=64)
    parser.add_argument('--L', type=int, default=3)
    parser.add_argument('--base', type=str, default='legendre')
    parser.add_argument('--cross_activation', type=str, default='tanh')

    # ===== DATA =====
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='/content/')
    parser.add_argument('--data_path', type=str, default='BTC-USD.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='Close')
    parser.add_argument('--freq', type=str, default='d')
    parser.add_argument('--detail_freq', type=str, default='d')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # ===== FORECASTING =====
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=24)

    # ===== MODEL STRUCTURE =====
    parser.add_argument('--enc_in', type=int, default=6)
    parser.add_argument('--dec_in', type=int, default=6)
    parser.add_argument('--c_out', type=int, default=6)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--moving_avg', default=[24])
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true')
    parser.add_argument('--do_predict', action='store_true')

    # ===== TRAINING =====
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='exp')
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--use_amp', action='store_true')

    # ===== GPU SETTINGS =====
    parser.add_argument('--use_gpu', type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument('--devices', type=str, default='0')

    args = parser.parse_args(args)

    # === GPU detection ===
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("\nArgs in experiment:")
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            setting = f"{args.task_id}_{args.model}_{args.mode_select}_modes{args.modes}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_{ii}"

            exp = Exp(args)
            print(f">>>>>>> start training : {setting} >>>>>>>>>")
            exp.train(setting)
            print(f">>>>>>> testing : {setting} <<<<<<<<<<<")
            exp.test(setting)

            if args.do_predict:
                print(f">>>>>>> predicting : {setting} <<<<<<<<<<<")
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        setting = f"{args.task_id}_{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_0"

        exp = Exp(args)
        print(f">>>>>>> testing : {setting} <<<<<<<<<<<")
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
