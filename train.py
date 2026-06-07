import warnings
warnings.filterwarnings('ignore')
from config import get_args
from utils import *
from datasets import *
from time import time
from multi_client import *


def main():
    start_time = time()
    args = get_args()
    SEED_EVERYTHING(args)

    if args.dataset.lower() == 'mnist':
        data = None
        args.num_classes = 10
    else:
        data = eval(args.dataset)(root=args.dataset_folder, test_envs=[args.test_env], img_size=args.img_size)
        args.num_classes = data.num_classes
    
    print(f'Number of classes: {args.num_classes}')

    project_name = f'{args.dataset}_server_{args.server_domain}_test_{args.test_env}_multi_client_try_another'
    args_dict = vars(args)
    
    
    args_dict['server_domain_name'] = data.ENVIRONMENTS[args.server_domain] if data else args.server_domain
    args_dict['test_domain_name'] = data.ENVIRONMENTS[args.test_env] if data else args.test_env

    if args.wandb:
        initialize_wandb(args, project_name, args_dict)

    server = load_model(args)
    if args.multi_client:
        server_trainloader, client_trainloader, testloader = get_dataloaders_multi_client(args, data)
        print(f'Using multi-client setting with {len(client_trainloader)} clients')
    else:
        server_trainloader, client_trainloader, testloader = get_dataloaders(args, data)
        print(f'Using single-client setting with {len(client_trainloader)} clients')
    clients = set_client_models(args, client_trainloader)
    set_scheduler(args, server, clients, server_trainloader, client_trainloader)
    
    if server.scheduler:
        print(f'Using {args.scheduler_type} scheduler')

    tgt_results = []

    for round in range(1, args.rounds + 1):
        start_time_round = time()
        server.train_supervised(server_trainloader)

        # query server and clients to record batchnorm statistics before communicating server model to clients
        server.make_batchnorm_stats(server_trainloader, client_trainloader, testloader.dataset.transform)

        for client in clients:
            client.model.load_state_dict(server.test_model.state_dict(), strict=True)

        for client, train_loader in zip(clients, client_trainloader):
            client.train_unsupervised(train_loader)

        client_weights = calculate_client_weights(client_trainloader)
        fedavg_clients(server, clients, skip_bn=args.skip_bn, client_weights=client_weights, 
                       uniform_weights=args.uniform_weights)
        
        # query server and clients to record batchnorm statistics before testing global model
        server.make_batchnorm_stats(server_trainloader, client_trainloader, testloader.dataset.transform)
        target_loss, target_acc = server.evaluate(testloader, test_model=True)
        tgt_results.append(target_acc)

        end_time_round = time()
        remaining_estimated_time = (end_time_round - start_time_round) * (args.rounds - round)
        log_wandb_metrics(args, target_acc, remaining_estimated_time)
        print(f'Round {round} Target loss: {target_loss:.4f}, Target acc: {target_acc:.4f}')

    write_results(args, tgt_results)
    hist = {'target_acc': target_acc, 'target_loss': target_loss}
    save_checkpoint(args, server, clients, hist)

    end_time = time()
    print(f'Total time taken: {(end_time - start_time) / 3600:.2f} hours!')

if __name__ == "__main__":
    main()
