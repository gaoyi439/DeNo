from utils import dataset

def choose(args):
    nl_tag = '{:g}'.format(args.nl)
    if args.dataset == 'scene':
        print('Data Preparation of scene')
        file_name = ["./data/scene_data.csv", "./data/scene_label.csv", "./data/scene_{}_com_label.csv".format(nl_tag),"./data/scene_{}_pml_label.csv".format(nl_tag)]
        train_loader, test_loader, _ = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 6
        input_dim = 294
    elif args.dataset == "yeast":
        print('Data Preparation of yeast')
        file_name = ["./data/yeast_data.csv", "./data/yeast_label.csv", "./data/yeast_{}_com_label.csv".format(nl_tag),"./data/yeast_{}_pml_label.csv".format(nl_tag)]
        train_loader, test_loader, _ = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 14
        input_dim = 103
    elif args.dataset == "bookmark":
        print('Data Preparation of bookmark')
        file_name = ["./data/bookmark_data.csv", "./data/bookmark_label.csv", "./data/bookmark_{}_com_label.csv".format(nl_tag), "./data/bookmark_{}_pml_label.csv".format(nl_tag)]
        train_loader, test_loader, _ = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 15
        input_dim = 2150
    elif args.dataset == "mediamill":
        print('Data Preparation of mediamill')
        file_name = ["./data/mediamill_data.csv", "./data/mediamill_label.csv", "./data/mediamill_{}_com_label.csv".format(nl_tag), "./data/mediamill_{}_pml_label.csv".format(nl_tag)]
        train_loader, test_loader, _ = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 15
        input_dim = 120
    elif args.dataset == "delicious":
        print('Data Preparation of delicious')
        file_name = ["./data/delicious_data.csv", "./data/delicious_label.csv", "./data/delicious_{}_com_label.csv".format(nl_tag), "./data/delicious_{}_pml_label.csv".format(nl_tag)]
        train_loader, test_loader, _ = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 15
        input_dim = 500
    elif args.dataset == "Corel16k":
        print('Data Preparation of Corel16k')
        file_name = ["./data/Corel16k_data.csv", "./data/Corel16k_label.csv", "./data/Corel16k_{}_com_label.csv".format(nl_tag), "./data/Corel16k_{}_pml_label.csv".format(nl_tag)]
        train_loader, test_loader, _ = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 15
        input_dim = 500
    elif args.dataset == "tmc2007":
        print('Data Preparation of tmc2007')
        file_name = ["./data/tmc2007_data.csv", "./data/tmc2007_label.csv", "./data/tmc2007_{}_com_label.csv".format(nl_tag), "./data/tmc2007_{}_pml_label.csv".format(nl_tag)]
        train_loader, test_loader, _ = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 22
        input_dim = 981
    elif args.dataset == "rcv1subset1":
        print('Data Preparation of rcv1subset1')
        file_name = ["./data/rcv1subset1_data.csv", "./data/rcv1subset1_label.csv", "./data/rcv1subset1_{}_com_label.csv".format(nl_tag), "./data/rcv1subset1_{}_pml_label.csv".format(nl_tag)]
        train_loader, test_loader, _ = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 15
        input_dim = 944
    elif args.dataset == "rcv1subset2":
        print('Data Preparation of rcv1subset2')
        file_name = ["./data/rcv1subset2_data.csv", "./data/rcv1subset2_label.csv", "./data/rcv1subset2_{}_com_label.csv".format(nl_tag), "./data/rcv1subset2_{}_pml_label.csv".format(nl_tag)]
        train_loader, test_loader, _ = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 15
        input_dim = 944
    elif args.dataset == "rcv1subset3":
        print('Data Preparation of rcv1subset3')
        file_name = ["./data/rcv1subset3_data.csv", "./data/rcv1subset3_label.csv", "./data/rcv1subset3_{}_com_label.csv".format(nl_tag), "./data/rcv1subset3_{}_pml_label.csv".format(nl_tag)]
        train_loader, test_loader, _ = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 15
        input_dim = 944
    elif args.dataset == "rcv1subset4":
        print('Data Preparation of rcv1subset4')
        file_name = ["./data/rcv1subset4_data.csv", "./data/rcv1subset4_label.csv", "./data/rcv1subset4_{}_com_label.csv".format(nl_tag), "./data/rcv1subset4_{}_pml_label.csv".format(nl_tag)]
        train_loader, test_loader, _ = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 15
        input_dim = 944
    elif args.dataset == "rcv1subset5":
        print('Data Preparation of rcv1subset5')
        file_name = ["./data/rcv1subset5_data.csv", "./data/rcv1subset5_label.csv", "./data/rcv1subset5_{}_com_label.csv".format(nl_tag), "./data/rcv1subset5_{}_pml_label.csv".format(nl_tag)]
        train_loader, test_loader, _ = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 15
        input_dim = 944
    elif args.dataset == "YeastCC":
        print('Data Preparation of YeastCC')
        file_name = ["./data/YeastCC_data.csv", "./data/YeastCC_label.csv", "./data/YeastCC_com_label.csv", "./data/YeastCC_pml_label.csv"]
        train_loader, test_loader, _ = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 50
        input_dim = 6139
    elif args.dataset == "YeastBP":
        print('Data Preparation of YeastBP')
        file_name = ["./data/YeastBP_data.csv", "./data/YeastBP_label.csv", "./data/YeastBP_com_label.csv", "./data/YeastBP_pml_label.csv"]
        train_loader, test_loader, _ = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 217
        input_dim = 6139
    elif args.dataset == "Music_emotion":
        print('Data Preparation of Music_emotion')
        file_name = ["./data/music_emotion_data.csv", "./data/music_emotion_label.csv", "./data/music_emotion_com_label.csv", "./data/music_emotion_pml_label.csv"]
        train_loader, test_loader, _ = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 11
        input_dim = 98
    elif args.dataset == "Music_style":
        print('Data Preparation of Music_style')
        file_name = ["./data/music_style_data.csv", "./data/music_style_label.csv", "./data/music_style_com_label.csv", "./data/music_style_pml_label.csv"]
        train_loader, test_loader, _ = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 10
        input_dim = 98
    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    return train_loader, test_loader, num_class, input_dim