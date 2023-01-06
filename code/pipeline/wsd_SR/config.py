class parameters():

    # Add data args.
    processor_path = r"C:\Users\pingu\FinQA_replication\code\pipeline\wsd_SR\model\processor_config.json"
    model_path = r"C:\Users\pingu\FinQA_replication\code\pipeline\wsd_SR\model\best_checkpoint_val_f1=0.7626_epoch=018.ckpt"

    # cuda or cpu
    device = "cpu"

    batch_size = 32
    num_workers = 4

    # data paths
    train_path = r"C:\Users\pingu\FinQA_replication\dataset\train.json"
    model_output = r"C:\Users\pingu\FinQA_replication\dataset\train_WSD_SR_augmented.json"