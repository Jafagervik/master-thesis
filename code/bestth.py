def find_best_treshold(dl: DataLoader, args, devices, **config) -> float:
    reconstruction_errors = []
    model  = select_model(args.model, devices, **config)
    load_model(model)
    
    for data in dl:
        out = model.predict(data)
        rec = mse(data, out).item()
        reconstruction_errors.append(rec)
    
    reconstruction_errors = np.array(reconstruction_errors)
    true_anomalies = get_true_anomalies()
    
    precisions, recalls, thresholds = precision_recall_curve(true_anomalies, reconstruction_errors)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_treshold