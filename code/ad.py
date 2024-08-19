def ad_report(dl: DataLoader, args, devices, **config) -> float:
    errors = []
    model  = select_model(args.model, devices, **config)
    load_model(model)
    
    for data in dl:
        out = model.predict(data)
        rec = mse(data, out).item()
        errors.append(rec)
    
    errors = np.array(errors)
    th = np.percentile(errors, 95)
    
    predicted_anomalies = errors > th
    true_anomalies = get_true_anomalies()
    
    das_confmat(confusion_matrix(true_anomalies, predicted_anomalies))
    print(classification_report(true_anomalies, predicted_anomalies))
    
    precisions, recalls, thresholds = precision_recall_curve(true_anomalies, reconstruction_errors)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"Best threshold: {best_threshold}")
    
    final_predictions = errors > best_threshold
    print(classification_report(true_anomalies, final_predictions))
    
    das_prcurve(recalls, precisions)
    fpr, tpr, _ = roc_curve(true_anomalies, errors)
    roc_auc = auc(fpr, tpr)
    das_roccurve(fpr, tpr, show=True)
    return best_treshold