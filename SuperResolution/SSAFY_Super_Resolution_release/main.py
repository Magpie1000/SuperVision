import sys
from dataloader import load_dataset
from datasaver import save_data, analyze_output
from trainer import get_trainer
from data_processing import load_jpg
from Utils.options import parseArgs_main


def evaluate(trainer, feature, target, batch_size=1, save_output=True):
    if save_output:
        outputs = trainer.evaluate(feature, target, batch_size, True)
        save_data(feature, outputs, target)
    else:
        trainer.evaluate(feature, target, batch_size, False)


def test(trainer, input_file):
    inp_lr = load_jpg(input_file, True)
    out = trainer.output(inp_lr)
    analyze_output(inp_lr, out[0])


def main(dict):
    print(dict)
    # Model Vars
    h, w = 480, 640

    # Make/Load Trainer
    trainer = get_trainer(w // 2, h // 2, dict['model_path'], dict['load_model'])

    if dict['train']:
        # Load Train Data
        train_feature, train_target, _ = load_dataset(dict['train_data_path'])
        # Train
        trainer.train(train_feature, train_target, dict['epochs'], dict['batch'])

    if dict['eval']:
        # Load Valid Data
        valid_feature, valid_target, _ = load_dataset(dict['valid_data_path'])
        # Evaluate
        evaluate(trainer, valid_feature, valid_target, dict['batch'])

    if dict['test']:
        test(trainer, dict['test_file'])


if __name__ == '__main__':
    main(parseArgs_main(sys.argv))
