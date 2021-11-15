import configparser
import tensorflow as tf
from utils import load_data_outer, load_lists, plot_history, compare_embeddings
from dimensionality_reduction.autoencoder import AutoEncoder


def main(config):
    exp_name = config['experiment']

    if config['embeddings'] == 'Laser_embeddings':
        full = 1024
    else:
        full = 768

    bottleneck = int(load_lists(config['encoder_dimensions'])[-1])

    ratio = bottleneck / full * 100

    exp_name = f'{exp_name}_ratio_{ratio}'

    # load training data set
    train_sents, train_embs, _ = load_data_outer(config['training_data'],
                                                 config['embeddings'],
                                                 config['training_samples_per_cat'])

    # load validation data set
    dev_sents, dev_embs, _ = load_data_outer(config['training_data'],
                                             config['embeddings'],
                                             config['dev_samples_per_cat'])

    print(exp_name, len(train_embs), len(dev_embs))

    # perform dimensionality reduction with the autoencoder
    autoencoder = AutoEncoder(
        encoder_dimensions=load_lists(config['encoder_dimensions']),
        decoder_dimensions=load_lists(config['decoder_dimensions']))

    history = autoencoder.train(train_embs, dev_embs, epochs=int(config['epochs']),
                                batch_size=int(config['batch_size']))

    # plot training and validation loss
    plot_history(history, 'loss', f'{exp_name}', config['plots_dir'])
    # plot training and validation accuracy
    plot_history(history, 'accuracy', f'{exp_name}', config['plots_dir'])

    # reconstruct development embeddings
    reconstructed_dev_embs = autoencoder.full_model.predict(dev_embs)

    # cast tensor to float16
    reconstructed_dev_embs = tf.cast(reconstructed_dev_embs, tf.float16)

    average_cosine_similarity = compare_embeddings(dev_embs, reconstructed_dev_embs)

    # try:
    with open(f'{config["results_dir"]}/log.txt', 'a') as outfile:
        outfile.write(f'exp name: {exp_name} ---- avg cosine similarity: {round(average_cosine_similarity, 2)}\n')
    # except:
    #     with open(f'{config["results_dir"]}/log.txt', 'w') as outfile:
    #         outfile.write(f'exp name: {exp_name} ---- avg cosine similarity: {round(average_cosine_similarity, 2)}')

    print()
    print(average_cosine_similarity)


if __name__ == '__main__':
    cfg = configparser.ConfigParser()
    cfg.read_file(open('src/config.ini'))
    main(cfg['default'])
