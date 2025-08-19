import os.path as osp
import numpy as np
from tqdm import tqdm
import wandb
import torch

from .base_handler import BaseHandler
from model.utils import load_model, general_init_weight
from model.deepmil import DeepAttnMISL
from utils.func import parse_str_dims, fetch_kws, rename_keys
from loss.utils import load_loss
from eval.utils import load_evaluator
from dataset.utils import prepare_surv_dataset
from dataset.label_converter import MetaSurvData
from utils.io import save_prediction_surv, save_wsi_representation


class SAHandler(BaseHandler):
    """
    This class handles the initialization, training, and testing 
    of SA (Survival Analysis) models for WSIs.
    """
    def __init__(self, cfg):
        assert cfg['task'] == 'sa', f"Expected task = `sa` but got {cfg['task']}."
        # run setup of cuda, seed, path, model, loss, optimizer
        # LR scheduler, evaluator, and evaluation metrics with 
        # the functions written to override those base ones. 
        super().__init__(cfg)

    def _check_arguments(self, cfg):
        print("[setup] start checking all arguments...")
        
        if 'SurvMLE' in cfg['loss_type']:
            assert cfg['net_output_converter'] == 'sigmoid'
            assert cfg['evaluator'] == 'NLL'
        elif 'SurvIFMLE' in cfg['loss_type']:
            assert cfg['net_output_converter'] == 'softmax'
            assert cfg['evaluator'] == 'NLL-IF'
        elif 'SurvPLE' in cfg['loss_type']:
            assert cfg['net_output_converter'] is None
            assert cfg['evaluator'] == 'Cox'

        print("[setup] argument checking passed.")

    @staticmethod
    def func_load_meta_data(cfg, data_split=None, **kws):
        path_table = cfg['path_table']
        time_format = cfg['time_format']
        time_bins = cfg['time_bins']
        max_event_time = cfg['time_event_max'] if 'time_event_max' in cfg else None

        assert time_format in ['origin', 'ratio', 'interval', 'quantile']
        use_discrete_label = time_format in ['interval', 'quantile']

        meta_data = MetaSurvData(
            path_label=path_table, data_split=data_split, 
            dataset_name=cfg['dataset_name'], dataset_id=0
        ) # dataset_id = 0 as there is only one dataset used for training
        label_column = ['y_t', 'y_e']
        if use_discrete_label:
            meta_data.generate_discrete_label(
                num_bins=time_bins, 
                max_event_time=max_event_time, 
                new_column_t=label_column[0], 
                new_column_e=label_column[1],
                use_quantiles=time_format=='quantile'
            )
        else:
            meta_data.generate_continuous_label(
                new_column_t=label_column[0], 
                new_column_e=label_column[1],
                normalize=time_format=='ratio'
            )

        # correct those `time_bins`-related variables for discrete survival models
        if use_discrete_label:
            if time_bins is None:
                cfg['time_bins'] = meta_data.num_bins
                print(f"[setup] `time_bins` has been changed from {time_bins} to {cfg['time_bins']}.")
            assert cfg['time_bins'] == meta_data.num_bins

            key_num_cls = cfg['arch'].lower() + "_num_cls"
            if meta_data.num_bins != cfg[key_num_cls]:
                print(f"[setup] `{key_num_cls}` will be changed from {cfg[key_num_cls]} to {meta_data.num_bins}.")
                cfg[key_num_cls] = meta_data.num_bins

        print("[setup] meta data has been generated.")
        return meta_data

    @staticmethod
    def func_load_evaluator(cfg, meta_data=None):
        assert cfg['evaluator'] in ['Reg', 'NLL', 'NLL-IF', 'Cox']
        evaluator_backend = 'default' if 'evaluator_backend' not in cfg else cfg['evaluator_backend']
        evaluator = load_evaluator(
            cfg['task'], cfg['evaluator'], 
            backend=evaluator_backend,
            meta_data=meta_data,
        )
        metrics_list = evaluator.valid_metrics
        ret_metrics = ['c_index', 'loss']
        return evaluator, metrics_list, ret_metrics

    @staticmethod
    def func_prepare_dataset(patient_ids, set_name, cfg, meta_data=None):
        dataset = prepare_surv_dataset(patient_ids, cfg, meta_data=meta_data)
        return dataset

    def save_prediction_results(self, data_cltor, path_to_save, **kws):
        save_prediction_surv(data_cltor['uid'], data_cltor['y'], data_cltor['y_hat'], path_to_save, **kws)
        if self.cfg['test'] and self.cfg['test_save_wsi_representation'] and 'rep_y_hat' in data_cltor:
            cur_save_path = self.cfg['test_save_wsi_representation_path']
            save_wsi_representation(data_cltor['uid'], data_cltor['rep_y_hat'], cur_save_path)

    def _train_each_epoch(self, epoch, train_loader, name_loader):
        self.net.train()
        bp_every_batch = self.cfg['bp_every_batch']
        all_raw_pred, all_gt, all_idx = [], [], []

        idx_collector, x_collector, y_collector = [], [], []
        i_batch = 0
        loop = tqdm(train_loader, desc=name_loader)
        for data_idx, data_x, data_y in loop:
            # data_x = (feats, coords) | data_y = label_slide
            i_batch += 1

            # 1. read data (mini-batch)
            data_input, data_input_ext = data_x
            data_label = data_y.cuda()

            x_collector.append((data_input, data_input_ext))
            y_collector.append(data_label)
            idx_collector.append(data_idx)

            # in a mini-batch
            if i_batch % bp_every_batch == 0: # drop_last_batch = True
                # 2. update network
                batch_loss, batch_pred = self._update_network(x_collector, y_collector)
                all_raw_pred.append(batch_pred)
                all_gt.append(torch.cat(y_collector, dim=0).detach().cpu())
                all_idx.append(torch.cat(idx_collector, dim=0).detach().cpu())

                # 3. reset mini-batch
                idx_collector, x_collector, y_collector = [], [], []
                torch.cuda.empty_cache()

                # 4. log and print
                wandb.log({'train/batch_loss': batch_loss})
                loop.set_description(f"Epoch [{epoch}/{self.cfg['epochs']}]")
                loop.set_postfix(loss=batch_loss)
                # print current LR
                current_lr = self.optimizer.param_groups[0]['lr']
                wandb.log({'train/batch_lr': current_lr})

        all_raw_pred = torch.cat(all_raw_pred, dim=0) # [B, num_out]
        all_gt = torch.cat(all_gt, dim=0) # [B, 2]: (t, e)
        all_idx = torch.cat(all_idx, dim=0).squeeze(-1) # [B, ]

        train_cltor = dict()
        # As it will be used for evaluation
        all_pred = self.output_converter(all_raw_pred)
        all_uids = self._get_unique_id('train', all_idx)
        train_cltor['pred'] = {'y': all_gt, 'raw_y_hat': all_raw_pred, 'y_hat': all_pred, 'uid': all_uids, 'name': 'train'}

        return train_cltor

    def calc_objective_loss(self, pred, label):
        batch_loss = .0
        # Herein we explicitly convert the network's raw outputs,
        # because the loss function cannot handle the raw predictions
        converted_pred = self.output_converter(pred) # e.g., sigmoid / softmax / identity
        for loss_name, loss_func in self.loss.items():
            t, e = label[:, [0]], label[:, [1]]
            batch_loss += self.loss_weight[loss_name] * loss_func(converted_pred, t, e)
        return batch_loss

    def _update_network(self, xs, ys):
        """
        Update network using one batch data
        """
        n_sample = len(xs)
        y_hat = []

        for i in range(n_sample):
            X, ext_data = xs[i]
            if isinstance(self.net, DeepAttnMISL):
                X = X.cuda()
                pred = self.net(X, ext_data.cuda())
            else:
                X = X.cuda()
                pred = self.net(X)
            y_hat.append(pred)

        # 3.1 zero gradients buffer
        self.optimizer.zero_grad()

        # 3.2 loss
        bag_preds = torch.cat(y_hat, dim=0) # [B, num_cls]
        bag_label = torch.cat(ys, dim=0) # [B, 2]
        pred_loss = self.calc_objective_loss(bag_preds, bag_label)

        # 3.3 backward gradients and update networks
        if isinstance(pred_loss, torch.Tensor) and pred_loss.requires_grad:
            pred_loss.backward()
            self.optimizer.step()
            self.steplr.step()
            val_loss = pred_loss.item()
        else:
            print("[batch train] warning: loss is not evaluated; skipped this batch training.")
            val_loss = 0

        val_preds = bag_preds.detach().cpu()
        return val_loss, val_preds

    def _eval_and_print(self, cltor, name='', ret_metrics=None, at_epoch=None):
        if ret_metrics is None:
            ret_metrics = self.ret_metrics
        if at_epoch is None:
            at_epoch = 'NA'
        eval_metrics = self.metrics_list

        ## NEW BEGIN: evaluate each loss item
        eval_results = self.evaluator.compute(
            cltor, eval_metrics, 
            kws_ext_loss=self.loss,
            loss_weight=self.loss_weight
        )
        ## NEW END
        
        eval_results = rename_keys(eval_results, name, sep='/')

        print("[{}] At epoch {}:".format(name, at_epoch), end=' ')
        print(' '.join(['{}={:.6f},'.format(k, v) for k, v in eval_results.items()]))
        wandb.log(eval_results)

        return [eval_results[name+'/'+k] for k in ret_metrics]

    def test_model(self, model, loader, loader_name, ckpt_path=None, **kws):
        if ckpt_path is not None:
            net_ckpt = torch.load(ckpt_path)
            model.load_state_dict(net_ckpt['model'], strict=False)
        model.eval()

        if 'test_save_wsi_representation' in self.cfg and self.cfg['test_save_wsi_representation']:
            save_wsi_rep = True
        else:
            save_wsi_rep = False
        all_reps = []

        all_idx, all_raw_pred, all_pred, all_gt = [], [], [], []
        for data_idx, data_x, data_y in loader:
            # data_x = (feats, coords) | data_y = label_slide
            X, ext_data = data_x
            data_label = data_y
            with torch.no_grad():
                if isinstance(model, DeepAttnMISL):
                    X = X.cuda()
                    raw_pred = model(X, ext_data.cuda())
                else:
                    X = X.cuda()
                    if save_wsi_rep:
                        raw_pred, wsi_rep = model(X, ret_bag_feat=True)
                        all_reps.append(wsi_rep.cpu())
                    else:
                        raw_pred = model(X)
                # To convert raw predictions for 
                # evaluation and prediction saving
                pred = self.output_converter(raw_pred)
            all_gt.append(data_label)
            all_raw_pred.append(raw_pred.detach().cpu())
            all_pred.append(pred.detach().cpu())
            all_idx.append(data_idx)
        
        all_raw_pred = torch.cat(all_raw_pred, dim=0) # [B, num_cls]
        all_pred = torch.cat(all_pred, dim=0) # [B, num_cls]
        all_gt = torch.cat(all_gt, dim=0) # [B, 2]
        all_idx = torch.cat(all_idx, dim=0).squeeze() # [B, ]

        cltor = dict()
        all_uids = self._get_unique_id(loader_name, all_idx)
        cltor['pred'] = {'y': all_gt, 'raw_y_hat': all_raw_pred, 'y_hat': all_pred, 'uid': all_uids, 'name': loader_name}

        if save_wsi_rep and len(all_reps) > 0:
            cltor['pred']['rep_y_hat'] = torch.cat(all_reps, dim=0) # [B, num_cls]

        return cltor
