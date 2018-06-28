import os
from random import shuffle
import numpy as np
import operator
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import json



def txt_clean_file(txtfile):
    with open(txtf, 'r') as txt_f:
        txt = txt_f.read().rstrip().lower()
        txt = re.sub(r'[^\w\s]','',txt)
        txt = re.sub(r'\s+',' ',txt)
        return txt
    
VCTK_PATH='/veu/spascual/git/speakagan/data/vctk/raw/VCTK-Corpus/'

# Select test speakers maximizing textual contents, taking
# 14 speakers with minium intersection of contents with 
# others in the 109 available in VCTKA.

spks = [l.rstrip().split(' ') for l in open(os.path.join(VCTK_PATH,
                                                         'speaker-info.txt'))]
spks = spks[1:]
spk2gen = dict(('p' + el[0], el[4]) for el in spks)
# add lost speaker
spk2gen['p280'] = 'F'
assert len(spk2gen) == 109, len(spk2gen)

txtfs = glob.glob(os.path.join(VCTK_PATH, 'txt', '**', '*.txt'), recursive=True)
print(len(txtfs))

if not os.path.exists('txt2spk') or not os.path.exists('spk2txt'):
    spk2txt = {}
    txt2spk = {}
    for ii, txtf in enumerate(txtfs, start=1):
        spk = txtf.split('/')[-2]
        txtname = txtf.split('/')[-1]
        txt = txt_clean_file(txtf)
        if spk not in spk2txt:
            spk2txt[spk] = []
        spk2txt[spk].append(txt)
        if txt not in txt2spk:
            txt2spk[txt] = []
        txt2spk[txt].append(spk)
        print('Processed {}/{}'.format(ii, len(txtfs)))
    with open('txt2spk', 'w') as txt2spk_f:
        txt2spk_f.write(json.dumps(txt2spk))
    with open('spk2txt', 'w') as spk2txt_f:
        spk2txt_f.write(json.dumps(spk2txt))
else:
    with open('txt2spk', 'r') as txt2spk_f:
        txt2spk = json.load(txt2spk_f)
    with open('spk2txt', 'r') as spk2txt_f:
        spk2txt = json.load(spk2txt_f)
txt2count = dict((k, len(v)) for k, v in txt2spk.items())
print(len(txt2count))
#print(txt2count)
plt.hist(list(txt2count.values()), bins=50)
plt.xlabel('# spks per txt')
plt.savefig('txt2count_hist.png', dpi=200)
spk2count = dict((k, len(v)) for k, v in spk2txt.items())
print(spk2count)
print(len(spk2count))
print('**********')

if not os.path.exists('spk2maxcount'):
    # matrix of spkxspk with interection counts of txts
    spkmat = {}
    # store repetition counts for each spk
    spk2maxcount = dict((k, 0) for k in list(spk2txt.keys()))
    spk2mincount = dict((k, np.inf) for k in list(spk2txt.keys()))
    spk2count = dict((k, 0) for k in list(spk2txt.keys()))
    for ii, txtf in enumerate(txtfs, start=1):
        spk = txtf.split('/')[-2]
        txt = txt_clean_file(txtf)
        spk2maxcount[spk] = max(spk2maxcount[spk], len(txt2spk[txt]))
        spk2mincount[spk] = min(spk2mincount[spk], len(txt2spk[txt]))
        spk2count[spk] += len(txt2spk[txt])
        if spk not in spkmat:
            spkmat[spk] = {}
        for intspk in txt2spk[txt]:
            if intspk not in spkmat[spk]:
                spkmat[spk][intspk] = 0
            spkmat[spk][intspk] += 1
        print('Processed {}/{}'.format(ii, len(txtfs)))
    with open('spk2maxcount', 'w') as spk2maxcount_f:
        spk2maxcount_f.write(json.dumps(spk2maxcount))
    with open('spk2mincount', 'w') as spk2mincount_f:
        spk2mincount_f.write(json.dumps(spk2mincount))
    with open('spkmat', 'w') as spkmat_f:
        spkmat_f.write(json.dumps(spkmat))
    with open('spk2count', 'w') as spk2count_f:
        spk2count_f.write(json.dumps(spk2count))
else:
    with open('spk2count', 'r') as spk2count_f:
        spk2count = json.load(spk2count_f)
    with open('spk2maxcount', 'r') as spk2maxcount_f:
        spk2maxcount = json.load(spk2maxcount_f)
    with open('spk2mincount', 'r') as spk2mincount_f:
        spk2mincount = json.load(spk2mincount_f)
    with open('spkmat', 'r') as spkmat_f:
        spkmat = json.load(spkmat_f)
print(sorted(spk2maxcount.items(), key=operator.itemgetter(1)))
print('---------------')
print(sorted(spk2mincount.items(), key=operator.itemgetter(1)))
print('ooooooooooooooo')
sorted_counts = sorted(spk2count.items(), key=operator.itemgetter(1))
print(sorted_counts)
with open('spkmat.txt', 'w') as mattxt_f:
    spks_h = list(spkmat.keys())
    header = ''
    for spk_h in spks_h:
        header += spk_h + ' '
    header = '     ' + header[:-1] + '\n'
    mattxt_f.write(header)
    # print header
    for si, spk in enumerate(spks_h):
        mattxt_f.write(spk + ' ')
        row = spkmat[spk]
        row_txt = ''
        for row_spk in spks_h:
            row_txt += '{:4d} '.format(spkmat[spk][row_spk])
        row_txt = row_txt[:-1] + '\n'
        mattxt_f.write(row_txt)

TEST_FILES = 14
VALID_FILES = 15
test_spks = []
valid_spks = []
train_spks = []
nontest_counts = []
# Now with minimum counts create test set, ensuring 50% 50% in male female
f = 0
m = 0
for spk in sorted_counts:
    if f + m < TEST_FILES:
        gen = spk2gen[spk[0]]
        if gen == 'F': 
            if f <= TEST_FILES // 2:
                print('Adding F spk: ', spk)
                f += 1
            else:
                print('Skipping F spk: ', spk)
                continue
        if gen == 'M':
            if m <= TEST_FILES // 2:
                print('Adding M spk: ', spk)
                m += 1
            else:
                print('Skipping M spk: ', spk)
                continue
        print('f: {}, m: {}'.format(f, m))
        test_spks.append(spk[0])
    else:
        nontest_counts.append(spk)

#re-shuffle counts now to mix valid-train
shuffle(nontest_counts)
print('DOING VALID -------------------------')
# Valid spks 50% 50%
f = 0
m = 0
for spk in nontest_counts:
    gen = spk2gen[spk[0]]
    if spk[0] in test_spks:
        continue
    if gen == 'F': 
        if f <= VALID_FILES // 2:
            print('Adding F spk: ', spk)
            f += 1
        else:
            print('Skipping F spk: ', spk)
            continue
    if gen == 'M':
        if m <= VALID_FILES // 2:
            print('Adding M spk: ', spk)
            m += 1
        else:
            print('Skipping M spk: ', spk)
            continue
    print('f: {}, m: {}'.format(f, m))
    valid_spks.append(spk[0])
    if f + m >= VALID_FILES:
        print('Out of valid')
        break

for spk in spk2gen.keys():
    if spk in (test_spks + valid_spks):
        continue
    train_spks.append(spk)

print('train spks: ', len(train_spks))
print('valid spks: ', len(valid_spks))
print('test spks: ', len(test_spks))

with open('train_split.txt', 'w') as train_f:
    for tr_spk in train_spks:
        train_f.write(tr_spk[1:] + '\n')

with open('valid_split.txt', 'w') as valid_f:
    for va_spk in valid_spks:
        valid_f.write(va_spk[1:] + '\n')

with open('test_split.txt', 'w') as test_f:
    for te_spk in test_spks:
        test_f.write(te_spk[1:] + '\n')
