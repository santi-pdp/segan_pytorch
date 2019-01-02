import torch
import os
import sys

if len(sys.argv) < 2:
    print('ERROR! Not enough input arguments.')
    print('Usage: {} <weights ckpt file> .'.format(sys.argv[0]))

ckpt_file = sys.argv[1]

# Converts old SEGAN-G weights namings
# Encoder: gen_enc.i.conv.weight/bias (i-th layer) --> enc_blocks.i.conv.weight/bias
# Decoder: gen_dec.i.conv.weight/bias (i-th layer) --> dec_blocks.i.deconv.weight/bias

out_file = ckpt_file + '.v2'


st_dict = torch.load(ckpt_file, 
                     map_location=lambda storage, loc: storage)

new_dict = {'state_dict':{}}
# copy first level keys and values, but state_dict (weights per-se)
for k, v in st_dict.items():
    if 'state_dict' in k:
        continue
    new_dict[k] = v

st_dict = st_dict['state_dict']

for k, v in st_dict.items():
    if 'gen_enc' in k:
        nk = k.replace('gen_enc', 'enc_blocks')
        print('{} -> {}'.format(k, nk))
        new_dict['state_dict'][nk] = v
    elif 'gen_dec' in k:
        nk = k.replace('gen_dec', 'dec_blocks')
        nk = nk.replace('conv', 'deconv')
        print('{} -> {}'.format(k, nk))
        new_dict['state_dict'][nk] = v
    else:
        print('Keeping {}'.format(k))
        new_dict['state_dict'][k] = v

torch.save(new_dict, out_file)
