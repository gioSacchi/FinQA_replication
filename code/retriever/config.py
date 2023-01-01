class parameters():

    prog_name = "retriever"

    # set up your own path here
    root_path = "../"
    output_path = "/content/gdrive/MyDrive/FinQAModels"
    cache_dir = "../cache"

    # the name of your result folder.
    model_save_name = "retriever-reberta-large"
    gdrive_save_predictions = "/content/gdrive/MyDrive/FinQAModels"

    train_file = root_path + "../dataset/train.json"
    valid_file = root_path + "../dataset/dev.json"

    test_file = root_path + "../dataset/test.json"

    op_list_file = "operation_list.txt"
    const_list_file = "constant_list.txt"

    # model choice: bert, roberta
    # pretrained_model = "bert"
    # model_size = "bert-base-uncased"

    pretrained_model = "roberta"
    model_size = "roberta-large"

    # train, test, or private
    # private: for testing private test data
    device = "cuda"
    mode = "train"
    resume_model_path = ""

    ### to load the trained model in test time
    saved_model_path = "/content/gdrive/MyDrive/FinQAModels/" + "chosen_checkpoint.pt"
    build_summary = False

    option = "rand"
    neg_rate = 3
    topn = 5

    sep_attention = True
    layer_norm = True
    num_decoder_layers = 1

    max_seq_length = 512
    max_program_length = 100
    n_best_size = 20
    dropout_rate = 0.1

    batch_size = 10
    batch_size_test = 16
    epoch = 100
    learning_rate = 2e-5

    report = 900
    report_loss = 100