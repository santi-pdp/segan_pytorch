import argparse
import json
import glob
import os


def clean(opts):
    logs = glob.glob(os.path.join(opts.ckpt_dir, '*checkpoint*'))
    print(logs)
    for log in logs:
        with open(log, 'r') as log_f:
            log_ = json.load(log_f)
            # first assertive check that all files exist, no mismatch
            # b/w log and filenames existence
            for fname in log_['latest']:
                fpath = os.path.join(opts.ckpt_dir, 'weights_' + fname)
                assert os.path.exists(fpath), fpath
            to_rm = [l for l in log_['latest'][:-1] if l != log_['current']]
            to_kp = log_['latest'][-1]
            for fname in to_rm:
                fpath = os.path.join(opts.ckpt_dir, 'weights_' + fname)
                os.unlink(fpath)
                print('Removed file ', fpath)
            print('Kept file ', os.path.join(opts.ckpt_dir, 'weights_' + \
                                             to_kp))
        # re-write log
        with open(log, 'w') as log_f:
            log_['latest'] = [log_['latest'][-1]]
            log_f.write(json.dumps(log_, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_dir', type=str, default=None)
    opts = parser.parse_args()

    clean(opts)
