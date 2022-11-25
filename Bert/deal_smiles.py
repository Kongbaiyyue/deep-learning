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


with open('data/chembl/chembl_31_smiles_notoolen.txt', 'r') as f2, open('data/chembl/chembl_31_smiles.txt', 'w') as f:
    first = True
    for line in f2.readlines():
        tokens = smi_tokenizer(line.split('\n')[0])
        smi = ' '.join(tokens)
        if first:
            f.write(smi)
            first = False
        else:
            f.write('\n' + smi)


