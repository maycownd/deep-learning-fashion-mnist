# -*- coding: utf-8 -*-
import os
import click
import logging
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
from PIL import Image

PATH_TRAIN = "data/raw/fashion-mnist_train.csv"
PATH_TEST = "data/raw/fashion-mnist_test.csv"
DATA_PATH = "data/raw/"

dict_fashion = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


def csv2img(csv, path, is_train=True):
    """
    Convert pixel values from .csv to .png image
    Source: https://www.kaggle.com/alexanch/image-classification-w-fastai-fashion-mnist
    """
    # define the name of the directory to be created
    if is_train:
        image_path = "working/train/"
    else:
        image_path = "working/test/"

    full_path = os.path.join(path, image_path)
    if os.path.isdir(full_path):
        return None
    try:
        os.makedirs(full_path)
    except OSError:
        print("Creation of the directory %s failed" % full_path)
    else:
        print("Successfully created the directory %s" % full_path)
        for i in range(len(csv)):
            # csv.iloc[i, 1:].to_numpy() returns pixel values array
            # for i'th imag excluding the label
            # next step: reshape the array to original shape(28, 28)
            # and add missing color channels
            result = Image.fromarray(np.uint8(
                np.stack(
                    np.rot90(
                        csv.iloc[i, 1:].to_numpy().
                        reshape((28, 28)))*3, axis=-1)))
            # save the image:
            result.save(f'{full_path}{str(i)}.png')

        print(f'{len(csv)} images were created.')


def create_train_test(csv_train, csv_test, data_path=DATA_PATH):
    """Create images on `data_path` from data provided by csvs.
    This is just a wrapper of csv2img to create the images provided
    by many csvs at once.

    Args:
        csv_list ([type]): [description]
        data_path (str, optional): [description]. Defaults to "../../Data/raw".
    """
    csv2img(csv_train, data_path, True)
    csv2img(csv_test, data_path, False)


def import_xy(
        path_train=PATH_TRAIN,
        path_test=PATH_TEST,
        label_name="label"):
    """Import data from specified path.

    Args:
        path_train (str, optional): [description]. Defaults to PATH_TRAIN.
        path_test ([type], optional): [description]. Defaults to PATH_TEST.
        label_name (str, optional): [description]. Defaults to "label".

    Returns:
        [type]: [description]
    """
    # importng the data from the paths which are there by default
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)

    # creating images from csv data
    create_train_test(df_train, df_test)

    # creating labels
    df_train['label_text'] = df_train['label'].apply(lambda x: dict_fashion[x])
    df_test['label_text'] = df_test['label'].apply(lambda x: dict_fashion[x])

    # add image names:
    df_train['img'] = pd.Series([str(i)+'.png' for i in range(len(df_train))])
    df_test['img'] = pd.Series([str(i)+'.png' for i in range(len(df_test))])
    X_train, y_train = df_train.drop("label", axis=1), df_train["label"]
    X_test, y_test = df_test.drop("label", axis=1), df_test["label"]

    # save corresponding labels and image names to .csv file:
    df_train[['img', 'label_text']].to_csv(
        os.path.join(DATA_PATH,
                     'working/train_image_labels.csv'), index=False)

    df_test[['img', 'label_text']].to_csv(
        os.path.join(DATA_PATH,
                     'working/test_image_labels.csv'), index=False)

    return X_train, y_train, X_test, y_test



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True),
                default="data/raw/")
@click.argument('output_filepath', type=click.Path(),
                default="data/interim/")
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    if not input_filepath:
        input_filepath="data/raw/fashion-mnist_train.csv"
    if not output_filepath:
        output_filepath="data/interim/mnist_train.csv"
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    X_train, y_train, X_test, y_test = import_xy()
    print("imported data")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
