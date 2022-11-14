class parameters():

    prog_name = "generator"

    # set up your own path here
    root_path = "../"
    output_path = "../output"
    cache_dir = "../cache"

    model_save_name = "bert-base"

    # train_file = root_path + "dataset/train.json"
    # valid_file = root_path + "dataset/dev.json"
    # test_file = root_path + "dataset/test.json"

    ### files from the retriever results
    # train_file = root_path + "dataset/train_retrieve.json"
    # valid_file = root_path + "dataset/dev_retrieve.json"
    # test_file = root_path + "dataset/test_retrieve.json"
    train_file = "/content/gdrive/MyDrive/Colab Notebooks/FinQA/generator_files/converted_train_predictions.json"
    valid_file = "/content/gdrive/MyDrive/Colab Notebooks/FinQA/generator_files/converted_dev_predictions.json"
    test_file = "/content/gdrive/MyDrive/Colab Notebooks/FinQA/generator_files/converted_test_predictions.json"

    # infer table-only text-only
    # test_file = root_path + "dataset/test_retrieve_7k_text_only.json"

    op_list_file = "operation_list.txt"
    const_list_file = "constant_list.txt"

    # # model choice: bert, roberta, albert
    pretrained_model = "bert"
    model_size = "bert-base-uncased"

    # model choice: bert, roberta, albert
    # pretrained_model = "roberta"
    # model_size = "roberta-large"

    # # finbert
    # pretrained_model = "finbert"
    # model_size = root_path + "pre-trained-models/finbert/"

    # pretrained_model = "longformer"
    # model_size = "allenai/longformer-base-4096"

    # single sent or sliding window
    # single, slide, gold, none
    retrieve_mode = "single"

    # use seq program or nested program
    program_mode = "seq"

    # train, test, or private
    # private: for testing private test data
    device = "cuda"
    mode = "train"
    saved_model_path = "/content/gdrive/MyDrive/Colab Notebooks/FinQA/" + "version_gen_model.pt"
    build_summary = False

    sep_attention = True
    layer_norm = True
    num_decoder_layers = 1

    max_seq_length = 512 # 2k for longformer, 512 for others
    max_program_length = 30
    n_best_size = 20
    dropout_rate = 0.1

    batch_size = 8
    batch_size_test = 16
    epoch = 300
    learning_rate = 2e-5

    report = 300
    report_loss = 100

    max_step_ind = 11