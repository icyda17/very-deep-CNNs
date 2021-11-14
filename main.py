from configs.config_read import ReadConfig
import torch

if __name__ == '__main__':

    print('Loading Config...')
    config = ReadConfig()
    
    model_name = config.model_name
    num_epochs = config.epochs
    shuffle = config.shuffle
    data_path = config.datapath
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Data...')
    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )



