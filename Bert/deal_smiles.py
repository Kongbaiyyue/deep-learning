import re


def smi_tokenizer(smi):
    """Tokenize a SMILES sequence or reaction"""
    pattern = "(\[[^\]]+]|Bi|Br?|Ge|Te|Mo|K|Ti|Zr|Y|Na|125I|Al|Ce|Cr|Cl?|Ni?|O|S|Pd?|Fe?|I|b|c|Mn|n|o|s|<unk>|>>|Li|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    if smi != ''.join(tokens):
        print('ERROR:', smi, ''.join(tokens))
    assert smi == ''.join(tokens)
    return tokens


def get_USPTO_50k():
    with open('data/USPTO-50k_no_rxn/src-train.txt', 'r') as f1,\
            open('data/USPTO-50k_no_rxn/tgt-train.txt', 'r') as f2, \
            open('data/USPTO-50k_no_rxn/src-val.txt', 'r') as f3, \
            open('data/USPTO-50k_no_rxn/tgt-val.txt', 'r') as f4, \
            open('data/USPTO-50k_no_rxn/Bert-train-nosplit.txt', 'w') as f:
        first = True
        for line in f1.readlines():
            line = line.strip()
            if first:
                f.write(line)
                first = False
            else:
                f.write('\n' + line)

        for line in f2.readlines():
            line = line.strip()
            f.write('\n' + line)

        for line in f3.readlines():
            line = line.strip()
            f.write('\n' + line)

        for line in f4.readlines():
            line = line.strip()
            f.write('\n' + line)


def check_smiles_length():
    max = -1
    min = 100000
    with open('data/USPTO-50k_no_rxn/Bert-train-nosplit.txt', 'r') as f:
        for line in f.readlines():
            line = line.replace(' ', '')
            s_len = len(line)
            if s_len > max:
                max = s_len
            if s_len < min:
                min = s_len
    print(max)
    print(min)
# with open('data/chembl/chembl_31_smiles_notoolen.txt', 'r') as f2, open('data/chembl/chembl_31_smiles.txt', 'w') as f:
#     first = True
#     for line in f2.readlines():
#         tokens = smi_tokenizer(line.split('\n')[0])
#         smi = ' '.join(tokens)
#         if first:
#             f.write(smi)
#             first = False
#         else:
#             f.write('\n' + smi)

# get_USPTO_50k()
check_smiles_length()
