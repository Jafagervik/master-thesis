def train_mode(args):
    config = get_config(args.model)
    seed_all(config["data"]["seed"])
    devices = get_gpus(args.gpus) 

    dtypes.default_float = dtypes.half if config["data"]["half_prec"] else dtypes.float32

    tl, vl = get_data(devices, **config)

    model = select_model(args.model, **config)
    if config["data"]["half_prec"]: model.half()
    if args.load: model.load()
    if len(devices) > 1: model.send_copy(devices)

    optim = select_optimizer(Opti.ADAM, model.parameters(), **config["opt"])
    schedule = select_lr_scheduler(LRScheduler.REDUCE, optim, **config)

    trainer = Trainer(model, tl, vl, optim, schedule, **config)
    trainer.train()