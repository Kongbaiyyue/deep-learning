import argparse
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from optimizers import Optimizer
from dataset_SMILES import SmilesDataset, SmilesCollator
from Bert import BERTEncoder, LastLine
from sample_loss import SampleLoss


def train_step(x, y, char_weight):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get saved data/model path")
    parser.add_argument('--train_src', '-train_src', type=str, default='data/chembl/chembl_31_smiles.txt')
    parser.add_argument('--test_src', '-test_src', type=str, default='data/USPTO-50k_no_rxn/src-test.txt')
    parser.add_argument('--num_layers', '-num_layers', type=int, default=6)
    parser.add_argument('--d_model', '-d_model', type=int, default=256)
    parser.add_argument('--dff', '-dff', type=int, default=512)
    parser.add_argument('--num_heads', '-num_heads', type=int, default=8)
    parser.add_argument('--vocab_size', '-vocab_size', type=int, default=88)
    parser.add_argument('--max_length', '-max_length', type=int, default=256)
    parser.add_argument('--batch_size', '-batch_size', type=int, default=256)

    parser.add_argument('--adam_beta1', '-adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', '-adam_beta2', type=float, default=0.998)
    parser.add_argument('--learning_rate', '-learning_rate', type=float, default=1.0)
    parser.add_argument('--epochs', '-epochs', type=int, default=10)

    parser.add_argument('--optim', '-optim', default='adam', choices=['sgd', 'adagrad', 'adadelta', 'adam',
                       'sparseadam', 'adafactor', 'fusedadam'])
    parser.add_argument('--learning_rate_decay', '-learning_rate_decay', type=float, default=0.5)
    parser.add_argument('--adagrad_accumulator_init', '-adagrad_accumulator_init', type=float, default=0)
    parser.add_argument('--model_dtype', '-model_dtype', default='fp32', choices=['fp32', 'fp16'])
    parser.add_argument('--loss_scale', '-loss_scale', type=float, default=0)
    parser.add_argument('--decay_method', '-decay_method', type=str, default="noam",
              choices=['noam', 'noamwd', 'rsqrt', 'none'])
    parser.add_argument('--warmup_steps', '-warmup_steps', type=int, default=4000)
    parser.add_argument('--rnn_size', '-rnn_size', type=int, default=256)
    parser.add_argument('--max_grad_norm', '-max_grad_norm', type=float, default=5)
    parser.add_argument('--reset_cycle_steps', '-reset_cycle_steps', type=int, default=0)
    parser.add_argument('--train_from', '-train_from', default='', type=str)
    parser.add_argument('--start_decay_steps', '-start_decay_steps', type=int, default=50000)

    args = parser.parse_args()
    max_length = args.max_length
    batch_size = args.batch_size

    vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 88, 256, 512, 8
    norm_shape, ffn_num_input, num_layers, dropout = [256], 256, 3, 0.2
    model = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, max_len=256, key_size=256, query_size=256, value_size=256)
    model2 = LastLine(num_hiddens, vocab_size)
    # model = BertModel(num_layers=args.num_layers, d_model=args.d_model, dff=args.dff, num_heads=args.num_heads, vocab_size=args.vocab_size)
    # mask_model = MaskLM(args.vocab_size, args.dff, num_inputs=args.d_model, max_length=max_length)
    # model = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, max_len=256, key_size=256, query_size=256, value_size=256)
    # model.load_state_dict(torch.load('Bert_smiles.pt'))

    device = torch.device("cuda")
    model.to(device)
    model2.to(device)

    train_src = 'data/USPTO-50k_no_rxn/USPTO-50k_no_rxn.vocab.txt'
    train_dataset = SmilesDataset(args, args.train_src)
    test_dataset = SmilesDataset(args, args.test_src)

    train_data_collator = SmilesCollator(
        max_length=max_length
    )
    test_data_collator = SmilesCollator(
        max_length=max_length
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=train_data_collator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=test_data_collator,
    )

    # params = [p for p in model.parameters() if p.requires_grad]
    # params2 = [p for p in model2.parameters() if p.requires_grad]
    # para_num = 0
    # for name, param in model.named_parameters():
        # if name == 'encoder':
            # para_num = param.nelement()
    # print(para_num)
    # params.extend(params2)
    # betas = [args.adam_beta1, args.adam_beta2]
    optimizer = Optimizer.from_opt(model, args, checkpoint=None)
    # optimizer = optim.Adam(params, lr=args.lr, betas=betas, eps=1e-9)
    # loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    loss_fn = SampleLoss()

    epochs = args.epochs
    for epoch in range(epochs):
        model.train()
        # mask_model.train()
        start_time = time.time
        for step, batch in enumerate(train_dataloader):
            x = batch['x']
            y = batch['y']
            char_weight = batch['weight']

            optimizer.zero_grad()
            # mask = torch.eq(x, 1).to(torch.float32)
            # mask = mask.unsqueeze(1).unsqueeze(2)
            valid_lens = batch['valid_lens']
            # outputs = model(x, None, mask, training=True)
            outputs = model(x, None, valid_lens)
            outputs = model2(outputs)
            # mlm_Y_hat = model(x, None, mask, training=True)
            # mlm_Y_hat = mask_model(outputs, char_weight)
            # lcoall = torch.nonzero(char_weight).to(device)
            # print(lcoall)

            # weight = torch.nonzero(char_weight)
            # y = y.reshape(-1)
            # y = [y[idx[0]*max_length+idx[1]] for idx in weight]
            # y = torch.tensor(y).to(device)
            loss = loss_fn(outputs.reshape(-1, args.vocab_size), y.reshape(-1), sample_weight=char_weight.reshape(-1))
            optimizer.backward(loss)
            optimizer.step()
            # print(loss)
            if step % 100 == 0:
                print(loss)
    all_test = 0
    prec = 0

    # 验证模型
    for batch in test_dataloader:
        with torch.no_grad():
            model.eval()
            # mask_model.eval()
            x = batch['x']
            y = batch['y']
            char_weight = batch['weight']

            # optimizer.zero_grad()
            valid_lens = batch['valid_lens']
            outputs = model(x, None, valid_lens)
            outputs = model2(outputs)
            # mask = torch.eq(x, 1).to(torch.float32)
            # mask = mask.unsqueeze(1).unsqueeze(2)
            # outputs = model(x, None, mask, training=True)
            outputs_label = torch.max(outputs, dim=-1)
            w = torch.nonzero(char_weight).to(device)

            for idx in w:
                all_test += 1
                if outputs_label[1][idx[0]][idx[1]] == y[idx[0]][idx[1]]:
                    prec += 1
    print(prec/float(all_test))
    # print(loss)
    torch.save(model.state_dict(), 'Bert_smiles_maxlen1000.pt')






