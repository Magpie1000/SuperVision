
from Utils.utils import does_not_exists

def invalid_args():
    print('Invalid System Arguments')
    exit(1)

def parseArgs_data_processing(args):
    #default path
    train_path = '../images/train'
    valid_path = '../images/valid'

    i = 1
    while i < len(args):
        if args[i] == '-t': # Train Image folder
            if i+1 < len(args) and not args[i+1].startswith('-'):
                train_path = args[i+1]
            else:
                invalid_args()
            i += 2
        elif args[i] == '-e': # Valid Image folder
            if i+1 < len(args) and not args[i+1].startswith('-'):
                valid_path = args[i+1]
            else:
                invalid_args()
            i += 2
        else:
            invalid_args()

    def check(path): # path check
        if does_not_exists(path):
            print(path)
            print('Image folder path not found')
            exit(1)

    check(train_path)
    check(valid_path)
    return train_path, valid_path

def parseArgs_main(args):
    dict = {}
    to_load_saved_model = False
    to_train_model = False
    to_evaluate = False
    to_test = False
    model_path = 'Model/model_0'
    test_file = '../images/valid/002660.jpg'
    train_data_path = 'Data/DataSet_train.tfrecord'
    eval_data_path = 'Data/DataSet_valid.tfrecord'
    epochs = 101
    batch_size = 64

    i = 1
    while i < len(args):
        if args[i] == '-lm': # Load Model?
            to_load_saved_model = True
            i += 1
        elif args[i] == '-m': # Model folder
            if i+1 < len(args) and not args[i+1].startswith('-'):
                model_path = args[i+1]
            else:
                invalid_args()
            i += 2
        elif args[i] == '-t': # Train
            to_train_model = True
            if i+1 < len(args) and not args[i+1].startswith('-'):
                train_data_path = args[i+1]
                i += 1
            i += 1
        elif args[i] == '-e': # Evaluate
            to_evaluate = True
            if i+1 < len(args) and not args[i+1].startswith('-'):
                eval_data_path = args[i+1]
                i += 1
            i += 1
        elif args[i] == '-o': # Get Output on given file 
            to_test = True
            if i+1 < len(args) and not args[i+1].startswith('-'):
                test_file = args[i+1]
                i += 1
            i += 1
        elif args[i] == '-epochs':
            if i+1 < len(args) and not args[i+1].startswith('-'):
                epochs = int(args[i+1])
            else:
                invalid_args()
            i += 2
        elif args[i] == '-batch':
            if i+1 < len(args) and not args[i+1].startswith('-'):
                batch_size = int(args[i+1])
            else:
                invalid_args()
            i += 2
        else:
            invalid_args()

    dict['model_path'] = model_path
    if to_load_saved_model and not does_not_exists(model_path):
        dict['load_model'] = True
    else:
        dict['load_model'] = False

    dict['train'] = to_train_model
    if to_train_model:
        dict['train_data_path'] = train_data_path
        dict['epochs'] = epochs

    dict['eval'] = to_evaluate
    if to_evaluate:
        dict['valid_data_path'] = eval_data_path

    if to_train_model or to_evaluate:
        dict['batch'] = batch_size

    dict['test'] = to_test
    if to_test:
        dict['test_file'] = test_file

    return dict