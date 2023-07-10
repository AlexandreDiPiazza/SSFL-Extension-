import argparse
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd 
from config import cfg, process_args
from data import fetch_dataset, split_dataset, make_data_loader, separate_dataset, separate_dataset_su, \
    make_batchnorm_dataset_su, make_batchnorm_stats
from metrics import Metric
from modules import Server, Client
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import make_logger

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    best_loss = float('inf')
    patience = 50
    epochs_without_improvement = 0
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    # Download the dataset, with dataset['train'] containing the train, and dataset['test] containing the test data
    server_dataset = fetch_dataset(cfg['data_name'])
    client_dataset = fetch_dataset(cfg['data_name'])

    # Just add the len of the dataset for both train and test, and the nbr of target classes to the config file
    process_dataset(server_dataset)

    # Server_dataset contains now only 4000 samples (the nbr of supervised we want in config)   
    # Client_dataset now contains only 46000 samples
    server_dataset['train'], client_dataset['train'], supervised_idx = separate_dataset_su(server_dataset['train'],
                                                                                           client_dataset['train'])
    
    # Create the dataloader on the server_dataset for both the test and the train, data_loader['train'] and data_loader['test]
    data_loader = make_data_loader(server_dataset, 'global')
    print(len(data_loader['train'].dataset), len(data_loader['test'].dataset))

    
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
  
    optimizer = make_optimizer(model.parameters(), 'local')
    scheduler = make_scheduler(optimizer, 'global')
   
    if cfg['sbn'] == 1:
        # Ca j'ai pas trop compris ce que ça faisait vrm 
        batchnorm_dataset = make_batchnorm_dataset_su(server_dataset['train'], client_dataset['train'])
    elif cfg['sbn'] == 0:
        batchnorm_dataset = server_dataset['train']
    else:
        raise ValueError('Not valid sbn')
    # split with iid or non iid, output is data_split['train'], data_split['test'] for the clients.
    # Here, data_split['train'][k] is the indices of the training set of client k 
    # Here, data_split['test'][k] is the indices testing set of client k, with the testing set coming from original 
    # test CIFAR-10
    data_split = split_dataset(client_dataset, cfg['num_clients'], cfg['data_split_mode'])
    #print(type(data_split['test'][0]))
    #print(data_split['test'][0])
    #print(len(data_split['train'][0]))
    #print(len(data_split['test'][0]))


    if cfg['loss_mode'] != 'sup':
        metric = Metric({'train': ['Loss', 'Accuracy', 'PAccuracy', 'MAccuracy', 'LabelRatio'],
                         'test': ['Loss', 'Accuracy']})
    else:
        metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        print('Resuming from last_epoch: ', last_epoch)
        if last_epoch > 1:
            data_split = result['data_split']
            supervised_idx = result['supervised_idx']
            server = result['server']
            client = result['client']
            optimizer.load_state_dict(result['optimizer_state_dict'])
            scheduler.load_state_dict(result['scheduler_state_dict'])
            logger = result['logger']
        else:
            server = make_server(model)
            # Give each client its indices for both test and train 
            client = make_client(model, data_split)
            logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_epoch = 1
        server = make_server(model)
        # Creates, for each clien, its model and its train indices/ test indices
        client = make_client(model, data_split)
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    
    Loss = [] ; Acc = []
    print('Nbr of Clients: {} ; % of Clients participating at each round: {}'.format(cfg['num_clients'], cfg['active_rate']) )
    print('Total nbr of Rounds: {} ; Epochs per round: {}'.format(cfg['global']['num_epochs'],
                                                    cfg['client']['num_epochs'] ))
    start_time = time.time()
    for epoch in range(last_epoch, cfg['global']['num_epochs'] + 1):
        # client_dataset['train] contains the whole training set 
        # client contains the indexes of each client for the whole dataset
        # training the clients; with client_dataset['train']
        test_loss, test_acc = train_client(batchnorm_dataset, client_dataset['train'], server, client, optimizer, metric, logger, epoch,
                                           client_dataset['test'])
        Loss.append(test_loss) ; Acc.append(test_acc)
        
        if 'ft' in cfg and cfg['ft'] == 0:
            train_server(server_dataset['train'], server, optimizer, metric, logger, epoch)
            logger.reset()
            server.update_parallel(client)
        else:
            logger.reset()
            # give new weights to the server
            server.update(client)
            train_server(server_dataset['train'], server, optimizer, metric, logger, epoch)
        scheduler.step()
        # load in the model for testing on the server 
        model.load_state_dict(server.model_state_dict)
        print('')
        print('ROUND {} Test Loss: {} ; Test Acc: {}'.format(epoch, test_loss, test_acc))
        print("--- %s seconds ---" % (time.time() - start_time))
        print('')
        # Create a dictionary from the lists
        data = {'Test Loss': Loss, 'Test Acc': Acc}
        df = pd.DataFrame(data)
        # Save the dataframe to a CSV file
        df.to_csv('./output/result_metrics{}-{}.csv'.format(cfg['model_tag'], cfg['optimization_mode']),
                   index=False)

        if (test_loss < best_loss) and (epoch >  5) : #first epoch loss is 0 with mix training set
            best_loss = test_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Check if training should be stopped
        if epochs_without_improvement == patience:
            print(f"No improvement in validation loss for {patience} epochs. Stopping training.")
            break
        #
        ## commented by dip bc we don't do test in the server in this setting
        """
        test_model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        test(data_loader['test'], test_model, metric, logger, epoch)
        """
        result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict(),
                  'supervised_idx': supervised_idx, 'data_split': data_split, 'logger': logger}
      
        save(result, './output/model/{}-{}_checkpoint.pt'.format(cfg['model_tag'], 
                                                        cfg['optimization_mode']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}-{}_checkpoint.pt'.format(cfg['model_tag'], 
                                                        cfg['optimization_mode']),
                        './output/model/{}-{}_best.pt'.format(cfg['model_tag'], 
                                                        cfg['optimization_mode']))
        logger.reset()
    return


def make_server(model):
    server = Server(model)
    return server


def make_client(model, data_split):
    client_id = torch.arange(cfg['num_clients'])
    client = [None for _ in range(cfg['num_clients'])]
    for m in range(len(client)):
        client[m] = Client(client_id[m], model, {'train': data_split['train'][m], 'test': data_split['test'][m]})
    return client


def train_client(batchnorm_dataset, client_dataset, server, client, optimizer, metric, logger, epoch,
                 test_client_dataset):
    logger.safe(True)
  
    # nbr of active clients for this round 
    num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
    client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
    for i in range(num_active_clients):
        client[client_id[i]].active = True
    # Distribute the server weights to the active clients
    
    server.distribute(client, batchnorm_dataset)
  
    num_active_clients = len(client_id)
    #print('nbr of active clients: ', num_active_clients)
    start_time = time.time()
    lr = optimizer.param_groups[0]['lr']
    

    te_loss = [] ; te_acc = [] ; c_weights = []
    for i in range(num_active_clients):
        m = client_id[i]
        # Create the dataset with the whole training set and the corresponding indexes of client m 
        dataset_m = separate_dataset(client_dataset, client[m].data_split['train'])
        ## ADDED BY DIP to create the dataset_m_test
        dataset_m_test = separate_dataset(test_client_dataset, client[m].data_split['test'])
        #print('Client on this round nbr of train/test samples:', len(dataset_m), len(dataset_m_test))
        
        #Check that on each client, the non-iid distribution is the same on train and test
        show = False
        if show == True: 
            print('Client on this round nbr of train/test samples:', len(dataset_m), len(dataset_m_test))
            print(type(dataset_m_test))
            
            category_counts1 = [0] * 10
            category_counts2 = [0] * 10
            dl1 = make_data_loader({'train': dataset_m}, 'client')['train']
            dl2 = make_data_loader({'test': dataset_m_test}, 'client')['test']
            for i, input in enumerate(dl1):
                input = collate(input)['target']
                for elem in (list(input)):
                  category_counts1[int(elem.item())] += 1
            tot = sum(category_counts1)
            category_counts1 = [x / tot for x in category_counts1]
            print(category_counts1)

            for i, input in enumerate(dl2):
                input = collate(input)['target']
                for elem in (list(input)):
                  category_counts2[int(elem.item())] += 1
            tot = sum(category_counts2)
            category_counts2 = [x / tot for x in category_counts2]
            print(category_counts2)
            print("######")
        
        

        ########################################################################""
        if 'batch' not in cfg['loss_mode'] and 'frgd' not in cfg['loss_mode'] and 'fmatch' not in cfg['loss_mode']:
            # turn into the fix and mix datasets.
            dataset_m = client[m].make_dataset(dataset_m, metric, logger)
                                                         
        if dataset_m is not None:
            # Take the len of each dataset
            c_weights.append(len(dataset_m_test))
            client[m].active = True
            client_te_loss, client_te_acc = client[m].train(dataset_m, lr, metric, logger, dataset_m_test)
            #print('Client test Loss/Acc', client_te_loss, client_te_acc)
            te_loss.append(client_te_loss) ; te_acc.append(client_te_acc)
        else:
            client[m].active = False
        if i % int((num_active_clients * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            epoch_finished_time = datetime.timedelta(seconds=_time * (num_active_clients - i - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['global']['num_epochs'] - epoch) * _time * num_active_clients))
            exp_progress = 100. * i / num_active_clients
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch (C): {}({:.0f}%)'.format(epoch, exp_progress),
                             'Learning rate: {:.6f}'.format(lr),
                             'ID: {}({}/{})'.format(client_id[i], i + 1, num_active_clients),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    print('Test Loss of Clients: ', te_loss)
    print('Test Acc of Clients: ', te_acc)
    # Weight of each sample
    N = sum(c_weights)
    weights = [x / N for x in c_weights]
    print('Weights of  Clients: ', weights)
    weighted_loss = sum(w * loss for w, loss in zip(weights, te_loss))
    weighted_acc = sum(w * acc for w, acc in zip(weights, te_acc))
    logger.safe(False)
    return weighted_loss, weighted_acc


def train_server(dataset, server, optimizer, metric, logger, epoch):
    logger.safe(True)
    start_time = time.time()
    lr = optimizer.param_groups[0]['lr']
    server.train(dataset, lr, metric, logger)
    _time = (time.time() - start_time)
    epoch_finished_time = datetime.timedelta(seconds=round((cfg['global']['num_epochs'] - epoch) * _time))
    info = {'info': ['Model: {}'.format(cfg['model_tag']),
                     'Train Epoch (S): {}({:.0f}%)'.format(epoch, 100.),
                     'Learning rate: {:.6f}'.format(lr),
                     'Epoch Finished Time: {}'.format(epoch_finished_time)]}
    logger.append(info, 'train', mean=False)
    print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def test(data_loader, model, metric, logger, epoch):
    logger.safe(True)
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    logger.safe(False)
    return


if __name__ == "__main__":
    main()
