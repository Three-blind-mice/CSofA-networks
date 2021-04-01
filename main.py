from data_loaders.conv_data_generator import DataGenerator
from models.conv_model import ConvModel
from trainers.conv_model_trainer import ConvModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args


def main():

    try:
        args = get_args()
        config = process_config(args.config)
        file = args.model_file
    except Exception as err:
        print("missing or invalid arguments")
        exit(0)
    create_dirs([config.callbacks.tensor_board.log_dir, config.callbacks.checkpoint.dir, config.graphics.dir])
    data_generator = DataGenerator(config)
    print('Created the data generator.')
    model = ConvModel(config)
    if file != "None":
        model.load(file)
    print('Created the model.')
    trainer = ConvModelTrainer(config, model.model)
    print('Created the trainer')
    print('Start training the model.')
    train = data_generator.get_train_data()
    valid = data_generator.get_valid_data()
    test = data_generator.get_test_data()
    trainer.train(train, valid)
    trainer.evaluate(test)


if __name__ == '__main__':
    main()