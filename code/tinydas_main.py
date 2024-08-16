def get_data(devices: List[str], **config) -> DataLoader:
    dataset = Dataset(
        n=config["data"]["nfiles"],
        normalize=Normalization.MINMAX,
        dtype=dtypes.float16 if config["data"]["half_prec"] else dtypes.float32  
    )
    return DataLoader(
        dataset, 
        batch_size=config["data"]["batch_size"], 
        devices=devices, 
        num_workers=config["data"]["num_workers"])

def train_mode(args):
    config = get_config(args.model)
    seed_all(config["data"]["seed"])

    devices = get_gpus(args.gpus)
    for x in devices: Device[x]
    
    model = select_model(args.model, devices, **config)
    if args.load: load_model(model)

    dataloader = get_data(devices, **config)

    if config["data"]["half_prec"]:
        for x in nn.state.get_state_dict(model).values():
            x = x.float().half()

    if len(devices) > 1:
        for x in nn.state.get_state_dict(model).values():
            x.realize().to_(devices)

    params = nn.state.get_parameters(model)
    optim = select_optimizer(Opti.ADAM, params, **config["opt"])

    trainer = Trainer(model, dataloader, optim, **config)
    trainer.train()