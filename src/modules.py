import copy
import datetime
import numpy as np
import sys
import time
import torch
import torch.nn.functional as F
import models
from itertools import compress
from config import cfg
from data import make_data_loader, make_batchnorm_stats, FixTransform, MixDataset
from utils import to_device, make_optimizer, collate, to_device
from metrics import Accuracy, Metric


def FedAVG_comm(server_model, models, client_weights_samples):
  """
  Performs the fedAVG optimization algos
  input: server_model, client_models
  output: new server_model, updated
  """
  with torch.no_grad():
    for key in server_model.state_dict().keys():
        # num_batches_tracked is a non trainable LongTensor and
        # num_batches_tracked are the same for all clients for the given datasets
        if 'num_batches_tracked' in key:
              #server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
              server_model.state_dict()[key].data.copy_(models[0].model_state_dict[key])
        else:
              temp = torch.zeros_like(server_model.state_dict()[key])
              for client_idx in range(len(client_weights_samples)):
                  #temp += client_weights_samples[client_idx] * models[client_idx].state_dict()[key]
                  temp += client_weights_samples[client_idx] *  models[client_idx].model_state_dict[key]

              server_model.state_dict()[key].data.copy_(temp)
              #for client_idx in range(len(client_weights_samples)):
              #    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
  return server_model #, models


class FedYogi():

    def __init__(self):

        self.delta_t_prv = None
        self.v_prv = None

        self.learning_rate = 0.01
        self.beta1 = 0.9 # 0
        self.beta2 = 0.99 # 0.5
        self.tao = 1e-3

    def aggregate(self, server_model, all_clients, target_client_ids, client_weights_samples) :
        """
        :param server_model:
        :param all_clients:
        :param target_client_ids:
        :return: the caller model can use .load_state_dict() to update the model
        """

        with torch.no_grad():

            x_t = copy.deepcopy(server_model.state_dict())

            ### CREATE DIFF DICTIONNARY
            dT = [0] * len(client_weights_samples)
            for client_id in target_client_ids:
               dT[client_id] = copy.deepcopy(server_model.state_dict())
               for name, param in server_model.named_parameters():
                  #dT[client_id][name] = all_clients[client_id].state_dict()[name] - server_model.state_dict()[name]
                  dT[client_id][name] = all_clients[client_id].model_state_dict[name] - server_model.state_dict()[name]
                  

            ###

            delta_t_cur = copy.deepcopy(x_t)
            for name, param in server_model.named_parameters():
                delta_t_cur[name].zero_()

            for client_id in target_client_ids:
                weight_i = client_weights_samples[client_id]

                #sd_local_delta = all_clients[client_id].get_model_delta_state_dict()
                sd_local_delta = dT[client_id]


                for name, param in server_model.named_parameters():
                    delta_t_cur[name] += sd_local_delta[name] * weight_i

            delta_t = copy.deepcopy(delta_t_cur)

            if self.delta_t_prv is None:
                self.delta_t_prv = copy.deepcopy(delta_t)
                for name, param in server_model.named_parameters():
                    self.delta_t_prv[name] = 0.0

            for name, param in server_model.named_parameters():
                delta_t[name] = self.beta1 * self.delta_t_prv[name] + (1 - self.beta1) * delta_t_cur[name]
            self.delta_t_prv = delta_t

            if self.v_prv is None:
                self.v_prv = copy.deepcopy(delta_t)
                for name, param in server_model.named_parameters():
                    self.v_prv[name] = self.tao**2

            v_t = copy.deepcopy(self.v_prv)
            for name, param in server_model.named_parameters():
                v_t[name] = self.v_prv[name] \
                            - (1 - self.beta2) * delta_t[name] * delta_t[name] \
                            * torch.sign(self.v_prv[name] - delta_t[name] * delta_t[name])
            self.v_prv = v_t

            x_t_nxt = copy.deepcopy(x_t)
            for name, param in server_model.named_parameters():
                x_t_nxt[name] = x_t[name] + self.learning_rate * delta_t[name]/(torch.sqrt(v_t[name]) + self.tao)
                ########################################################################################
                #for client_id in target_client_ids:
                #    all_clients[client_id].state_dict()[name].data.copy_(x_t_nxt[name])
                ##########################################################################################
                server_model.state_dict()[name].data.copy_(x_t_nxt[name])

        return server_model #, all_clients



class Server:
    def __init__(self, model):
        self.model_state_dict = save_model_state_dict(model.state_dict())
        if 'fmatch' in cfg['loss_mode']:
            optimizer = make_optimizer(model.make_sigma_parameters(), 'local')
            global_optimizer = make_optimizer(model.make_phi_parameters(), 'global')
        else:
            optimizer = make_optimizer(model.parameters(), 'local')
            global_optimizer = make_optimizer(model.parameters(), 'global')
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())

    def distribute(self, client, batchnorm_dataset=None, BN = False):
        opt_mode = cfg['optimization_mode']
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(self.model_state_dict)

       
        if batchnorm_dataset is not None:
              model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        model_state_dict = save_model_state_dict(model.state_dict())
       
        for m in range(len(client)):
            if client[m].active:
                client[m].model_state_dict = copy.deepcopy(model_state_dict)
                
                   
        return

    def update(self, client):
        opt_mode = cfg['optimization_mode']
        if (opt_mode =='FedYogi'):
            yogi = FedYogi()
        if 'fmatch' not in cfg['loss_mode']:
            with torch.no_grad():
                valid_client = [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    weight = torch.ones(len(valid_client))
                    weight = weight / weight.sum()

                    if opt_mode == 'FedSGD':
                        print('Doing FedSGD ')
                        global_optimizer = make_optimizer(model.parameters(), 'global')
                        global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                        global_optimizer.zero_grad()
                        for k, v in model.named_parameters():
                            parameter_type = k.split('.')[-1]
                            if 'weight' in parameter_type or 'bias' in parameter_type:
                                tmp_v = v.data.new_zeros(v.size())
                                for m in range(len(valid_client)):
                                    tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                                v.grad = (v.data - tmp_v).detach()
                        global_optimizer.step()
                        self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                        self.model_state_dict = save_model_state_dict(model.state_dict())
                    elif opt_mode == 'FedAVG':
                        print('Doing Fed AVG')
                        model = FedAVG_comm(server_model = model, models = valid_client, 
                                            client_weights_samples = weight)
                        self.model_state_dict = save_model_state_dict(model.state_dict())
                    elif opt_mode == 'FedYogi':
                        print('Doing FedYogi')
                        model = yogi.aggregate(server_model = model, all_clients = valid_client,
                                              target_client_ids= [m for m in range(len(valid_client))],
                                               client_weights_samples = weight)
                        self.model_state_dict = save_model_state_dict(model.state_dict())
                    else:
                        raise ValueError('Dip Optimization not implemented')
        elif 'fmatch' in cfg['loss_mode']:
            with torch.no_grad():
                valid_client = [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    weight = torch.ones(len(valid_client))
                    weight = weight / weight.sum()

                    if opt_mode == 'FedSGD':
                        print('Doing FedSGD ')
                        global_optimizer = make_optimizer(model.make_phi_parameters(), 'global')
                        global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                        global_optimizer.zero_grad()
                        for k, v in model.named_parameters():
                            parameter_type = k.split('.')[-1]
                            if 'weight' in parameter_type or 'bias' in parameter_type:
                                tmp_v = v.data.new_zeros(v.size())
                                for m in range(len(valid_client)):
                                    tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                                v.grad = (v.data - tmp_v).detach()
                        global_optimizer.step()
                        self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                        self.model_state_dict = save_model_state_dict(model.state_dict())
                    elif opt_mode == 'FedYogi':
                        print('Doing FedYogi')
                        model = yogi.aggregate(server_model = model, all_clients = valid_client,
                                              target_client_ids= [m for m in range(len(valid_client))],
                                               client_weights_samples = weight)
                        self.model_state_dict = save_model_state_dict(model.state_dict())
                    else:
                        raise ValueError('Dip Optimization not implemented')
        else:
            raise ValueError('Not valid loss mode')
        for i in range(len(client)):
            client[i].active = False
        return

    def update_parallel(self, client):
        opt_mode = cfg['optimization_mode']
        if (opt_mode =='FedYogi'):
            yogi = FedYogi()
        if 'frgd' not in cfg['loss_mode']:
            with torch.no_grad():
                valid_client_server = [self] + [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client_server) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    weight = torch.ones(len(valid_client_server))
                    weight = weight / (2 * (weight.sum() - 1))
                    weight[0] = 1 / 2 if len(valid_client_server) > 1 else 1
                    

                    if opt_mode == 'FedSGD':
                        print('Doing FedSGD ')
                        global_optimizer = make_optimizer(model.parameters(), 'global')
                        global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                        global_optimizer.zero_grad()
                        for k, v in model.named_parameters():
                            parameter_type = k.split('.')[-1]
                            if 'weight' in parameter_type or 'bias' in parameter_type:
                                tmp_v = v.data.new_zeros(v.size())
                                for m in range(len(valid_client_server)):
                                    tmp_v += weight[m] * valid_client_server[m].model_state_dict[k]
                                v.grad = (v.data - tmp_v).detach()
                        global_optimizer.step()
                        self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                        self.model_state_dict = save_model_state_dict(model.state_dict())
                    elif opt_mode == 'FedYogi':
                        print('Doing FedYogi')
                        model = yogi.aggregate(server_model = model, all_clients = valid_client_server,
                                              target_client_ids= [m for m in range(len(valid_client_server))],
                                               client_weights_samples = weight)
                        self.model_state_dict = save_model_state_dict(model.state_dict())
                  

                    else:
                        raise ValueError('Dip Optimization not implemented')
        elif 'frgd' in cfg['loss_mode']:
            with torch.no_grad():
                valid_client_server = [self] + [client[i] for i in range(len(client)) if client[i].active]
                num_valid_client = len(valid_client_server) - 1
                if len(valid_client_server) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    weight = torch.ones(len(valid_client_server)) / (num_valid_client // 2 + 1)
                    
                    if opt_mode == 'FedSGD':
                        print('Doing FedSGD ')
                        global_optimizer = make_optimizer(model.parameters(), 'global')
                        global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                        global_optimizer.zero_grad()
                        for k, v in model.named_parameters():
                            parameter_type = k.split('.')[-1]
                            if 'weight' in parameter_type or 'bias' in parameter_type:
                                tmp_v_1 = v.data.new_zeros(v.size())
                                tmp_v_1 += weight[0] * valid_client_server[0].model_state_dict[k]
                                for m in range(1, num_valid_client // 2 + 1):
                                    tmp_v_1 += weight[m] * valid_client_server[m].model_state_dict[k]
                                tmp_v_2 = v.data.new_zeros(v.size())
                                tmp_v_2 += weight[0] * valid_client_server[0].model_state_dict[k]
                                for m in range(num_valid_client // 2 + 1, len(valid_client_server)):
                                    tmp_v_2 += weight[m] * valid_client_server[m].model_state_dict[k]
                                tmp_v = (tmp_v_1 + tmp_v_2) / 2
                                v.grad = (v.data - tmp_v).detach()
                        global_optimizer.step()
                        self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                        self.model_state_dict = save_model_state_dict(model.state_dict())
                    elif opt_mode == 'FedYogi':
                        print('Doing FedYogi')
                        model = yogi.aggregate(server_model = model, all_clients = valid_client_server,
                                              target_client_ids= [m for m in range(len(valid_client_server))],
                                               client_weights_samples = weight)
                        self.model_state_dict = save_model_state_dict(model.state_dict())
                    else:
                        raise ValueError('Dip Optimization not implemented')
        else:
            raise ValueError('Not valid loss mode')
        for i in range(len(client)):
            client[i].active = False
        return

    def train(self, dataset, lr, metric, logger):
        if 'fmatch' not in cfg['loss_mode']:
            data_loader = make_data_loader({'train': dataset}, 'server')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
          
            model.load_state_dict(self.model_state_dict)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['server']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][1])))
            else:
                num_batches = None
            for epoch in range(1, cfg['server']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        else:
            data_loader = make_data_loader({'train': dataset}, 'server')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.make_sigma_parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['server']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][1])))
            else:
                num_batches = None
            for epoch in range(1, cfg['server']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            v.grad[(v.grad.size(0) // 2):] = 0
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.model_state_dict = save_model_state_dict(model.state_dict())
        return

class Client:
    def __init__(self, client_id, model, data_split):
        self.client_id = client_id
        self.data_split = data_split
        self.model_state_dict = save_model_state_dict(model.state_dict())
        if 'fmatch' in cfg['loss_mode']:
            optimizer = make_optimizer(model.make_phi_parameters(), 'local')
        else:
            optimizer = make_optimizer(model.parameters(), 'local')
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.active = False
        self.beta = torch.distributions.beta.Beta(torch.tensor([cfg['alpha']]), torch.tensor([cfg['alpha']]))
        self.verbose = cfg['verbose']

    def make_hard_pseudo_label(self, soft_pseudo_label):
        max_p, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_p.ge(cfg['threshold'])
        return hard_pseudo_label, mask

    def make_dataset(self, dataset, metric, logger):
        if 'sup' in cfg['loss_mode']:
            return dataset
        elif 'fix' in cfg['loss_mode']:
            with torch.no_grad():
                data_loader = make_data_loader({'train': dataset}, 'global', shuffle={'train': False})['train']
                model = eval('models.{}(track=True).to(cfg["device"])'.format(cfg['model_name']))
                model.load_state_dict(self.model_state_dict)
                
                model.train(False)
                output = []
                target = []
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input = to_device(input, cfg['device'])
                    output_ = model(input)
                    output_i = output_['target']
                    target_i = input['target']
                    output.append(output_i.cpu())
                    target.append(target_i.cpu())
                output_, input_ = {}, {}
                output_['target'] = torch.cat(output, dim=0)
                input_['target'] = torch.cat(target, dim=0)
                output_['target'] = F.softmax(output_['target'], dim=-1)
                new_target, mask = self.make_hard_pseudo_label(output_['target'])
                output_['mask'] = mask
                evaluation = metric.evaluate(['PAccuracy', 'MAccuracy', 'LabelRatio'], input_, output_)
                logger.append(evaluation, 'train', n=len(input_['target']))
                if torch.any(mask):
                    fix_dataset = copy.deepcopy(dataset)
                    fix_dataset.target = new_target.tolist()
                    mask = mask.tolist()
                    fix_dataset.data = list(compress(fix_dataset.data, mask))
                    fix_dataset.target = list(compress(fix_dataset.target, mask))
                    fix_dataset.other = {'id': list(range(len(fix_dataset.data)))}
                    if 'mix' in cfg['loss_mode']:
                        mix_dataset = copy.deepcopy(dataset)
                        mix_dataset.target = new_target.tolist()
                        mix_dataset = MixDataset(len(fix_dataset), mix_dataset)
                    else:
                        mix_dataset = None
                    return fix_dataset, mix_dataset
                else:
                    return None
        else:
            raise ValueError('Not valid client loss mode')

        return hard_pseudo_label, mask


    def train(self, dataset, lr, metric, logger, dataset_test):
        if cfg['loss_mode'] == 'sup':
            data_loader = make_data_loader({'train': dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        elif 'fix' in cfg['loss_mode'] and 'mix' not in cfg['loss_mode'] and 'batch' not in cfg[
            'loss_mode'] and 'frgd' not in cfg['loss_mode'] and 'fmatch' not in cfg['loss_mode']:
            fix_dataset, _ = dataset
            fix_data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(fix_data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, input in enumerate(fix_data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        elif 'fix' in cfg['loss_mode'] and 'mix' in cfg['loss_mode'] and 'batch' not in cfg[
            'loss_mode'] and 'frgd' not in cfg['loss_mode'] and 'fmatch' not in cfg['loss_mode']:
            fix_dataset, mix_dataset = dataset
            fix_data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
            mix_data_loader = make_data_loader({'train': mix_dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(fix_data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, (fix_input, mix_input) in enumerate(zip(fix_data_loader, mix_data_loader)):
                    input = {'data': fix_input['data'], 'target': fix_input['target'], 'aug': fix_input['aug'],
                             'mix_data': mix_input['data'], 'mix_target': mix_input['target']}
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['lam'] = self.beta.sample()[0]
                    input['mix_data'] = (input['lam'] * input['data'] + (1 - input['lam']) * input['mix_data']).detach()
                    input['mix_target'] = torch.stack([input['target'], input['mix_target']], dim=-1)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        elif 'batch' in cfg['loss_mode'] or 'frgd' in cfg['loss_mode'] or 'fmatch' in cfg['loss_mode']:
            data_loader = make_data_loader({'train': dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            if 'fmatch' in cfg['loss_mode']:
                optimizer = make_optimizer(model.make_phi_parameters(), 'local')
            else:
                optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    with torch.no_grad():
                        model.train(False)
                        input_ = collate(input)
                        input_ = to_device(input_, cfg['device'])
                        output_ = model(input_)
                        output_i = output_['target']
                        output_['target'] = F.softmax(output_i, dim=-1)
                        new_target, mask = self.make_hard_pseudo_label(output_['target'])
                        output_['mask'] = mask
                        evaluation = metric.evaluate(['PAccuracy', 'MAccuracy', 'LabelRatio'], input_, output_)
                        logger.append(evaluation, 'train', n=len(input_['target']))
                    if torch.all(~mask):
                        continue
                    model.train(True)
                    input = {'data': input['data'][mask], 'aug': input['aug'][mask], 'target': new_target[mask]}
                    input = to_device(input, cfg['device'])
                    input_size = input['data'].size(0)
                    input['loss_mode'] = 'fix'
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        else:
            raise ValueError('Not valid client loss mode')

        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.model_state_dict = save_model_state_dict(model.state_dict())
        ### ADDED BY DIP TO TEST ON FINE-TUNED DATA
        data_loader_test = make_data_loader({'test': dataset_test}, 'client')['test']
        if cfg['loss_mode'] != 'sup':
            metric = Metric({'train': ['Loss', 'Accuracy', 'PAccuracy', 'MAccuracy', 'LabelRatio'],
                         'test': ['Loss', 'Accuracy']})
        else:
            metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
        # Compute the test_loss and test_accuracy
        curr_loss, curr_acc = self.test_client_dip(data_loader_test, model, metric)

        return curr_loss, curr_acc


    def test_client_dip(self, data_loader, model, metric):
        losses = []
        accuracies = []
        with torch.no_grad():
            model.train(False)
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                output = model(input)
                output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                losses.append(evaluation['Loss'])
                accuracies.append(evaluation['Accuracy'])
        loss_avg = sum(losses)/len(losses)
        acc_avg = sum(accuracies)/len(accuracies)
        #return
        return loss_avg, acc_avg


def save_model_state_dict(model_state_dict):
    return {k: v.cpu() for k, v in model_state_dict.items()}


def save_optimizer_state_dict(optimizer_state_dict):
    optimizer_state_dict_ = {}
    for k, v in optimizer_state_dict.items():
        if k == 'state':
            optimizer_state_dict_[k] = to_device(optimizer_state_dict[k], 'cpu')
        else:
            optimizer_state_dict_[k] = copy.deepcopy(optimizer_state_dict[k])
    return optimizer_state_dict_
