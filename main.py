"""
This is our entry file to run all experiments
"""
import argparse
import time

from runner import BaseHandler, SAHandler, SATransferHandler
from utils.io import load_config_from_yaml, print_config
from utils.func import args_grid
from utils.func_config import is_valid_run_cfg
from utils.func_config import convert_to_abbr, ignore_it_in_save_path


def get_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-f', required=True, type=str, help='Path to the config file.')
    parser.add_argument('--handler', '-d', type=str, choices=['CLF', 'SA', 'SAT'], default='SA', help='Handler for running experiments.')
    parser.add_argument('--multi_run', action='store_true', help='If have multiple runs.')
    parser.add_argument('--sleep', type=int, default=0, help='If sleep X seconds between two runs, only valid in a multi_run mode.')
    args = vars(parser.parse_args())
    return args

def main(handler, config):
    if not is_valid_run_cfg(config):
        print("[Warning] skipped this run with config:", config)
        return

    model = handler(config)
    if config['test']:
        metrics = model.exec_test()
    else:
        metrics = model.exec()
    print('[INFO] Metrics:', metrics)

def multi_run_main(handler, config, sleep=0):
    hyperparams = []
    for k, v in config.items():
        if isinstance(v, list):
            hyperparams.append(k)

    if config['data_split_fold'] is None:
        configs = args_grid(config, loop_preference=['dataset_name'])
    else:
        configs = args_grid(config, loop_preference=['data_split_fold', 'dataset_name'])

    for cur_cfg in configs:
        print('\n')
        for k in hyperparams:
            abbr_key, abbr_value = convert_to_abbr(k), convert_to_abbr(cur_cfg[k])
            
            if ignore_it_in_save_path(k, cur_cfg[k]):
                print(f"[INFO] `{k}` is ignored and will not be added to `save_path`.")
                continue

            cur_cfg['save_path'] += '-{}_{}'.format(abbr_key, abbr_value)
            if cur_cfg['test']:
                cur_cfg['test_save_path'] += '-{}_{}'.format(abbr_key, abbr_value)

        if not is_valid_run_cfg(cur_cfg):
            print("[Warning] skipped this run with config:", cur_cfg)
            continue

        model = handler(cur_cfg)
        if cur_cfg['test']:
            print(cur_cfg['test_save_path'])
            metrics = model.exec_test()
        else:
            print(cur_cfg['save_path'])
            metrics = model.exec()

        print('[INFO] Metrics:', metrics)

        time.sleep(sleep)

if __name__ == '__main__':
    cfg = get_cmd_args()
    config = load_config_from_yaml(cfg['config'])
    print_config(config)
    
    if cfg['handler'] == 'CLF':
        handler = BaseHandler
    elif cfg['handler'] == 'SA':
        handler = SAHandler
    elif cfg['handler'] == 'SAT':
        handler = SATransferHandler
    else:
        raise RuntimeError(f"Expected `CLF`, `SA`, or `SAT`, but got {cfg['handler']}")

    if cfg['multi_run']:
        multi_run_main(handler, config, sleep=cfg['sleep'])
    else:
        main(handler, config)
