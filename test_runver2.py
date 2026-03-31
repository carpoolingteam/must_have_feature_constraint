#her bir sınıfın berlirli yuzdesi yerin tum veri setinin belirli bir syuzdesi kadar noktanın constraintini 1 yap
""" https://github.com/INFORMSJoC/2023.0419/tree/main/data yararlanılmıştır. Yayın :
@misc{bauhoc2024pccc,
  author =        {P. Baumann and D.S. Hochbaum},
  publisher =     {INFORMS Journal on Computing},
  title =         {{An algorithm for clustering with confidence-based must-link and cannot-link constraints}},
  year =          {2024},
  doi =           {10.1287/ijoc.2023.0419.cd},
  url =           {https://github.com/INFORMSJoC/2023.0419},
  note =          {Available for download at https://github.com/INFORMSJoC/2023.0419},
}  """



import shutil
import requests
import zipfile
import tarfile
import gzip
import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
# Store urls of data sets (valid as of April 2022)
links_to_data_sets = {
    'appendicitis': "https://sci2s.ugr.es/keel/dataset/data/classification/appendicitis.zip",
    'banana': "https://sci2s.ugr.es/keel/dataset/data/classification/banana.zip",
    'breast_cancer':
        "https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset",
    'bupa': "https://sci2s.ugr.es/keel/dataset/data/classification/bupa.zip",
    'cifar-10': "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    'cifar-100': "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
    'ecoli': "https://sci2s.ugr.es/keel/dataset/data/classification/ecoli.zip",
    'glass': "https://sci2s.ugr.es/keel/dataset/data/classification/glass.zip",
    'haberman': "https://sci2s.ugr.es/keel/dataset/data/classification/haberman.zip",
    'hayesroth': "https://sci2s.ugr.es/keel/dataset/data/classification/hayes-roth.zip",
    'heart': "https://sci2s.ugr.es/keel/dataset/data/classification/heart.zip",
    'ionosphere': "https://sci2s.ugr.es/keel/dataset/data/classification/ionosphere.zip",
    'iris': "https://sci2s.ugr.es/keel/dataset/data/classification/iris.zip",
    'led7digit': "https://sci2s.ugr.es/keel/dataset/data/classification/led7digit.zip",
    'letter': "https://sci2s.ugr.es/keel/dataset/data/classification/letter.zip",
    'mnist': ["https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
              "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
              "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
              "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"],
    'monk2': "https://sci2s.ugr.es/keel/dataset/data/classification/monk-2.zip",
    'movement_libras': "https://sci2s.ugr.es/keel/dataset/data/classification/movement_libras.zip",
    'newthyroid': "https://sci2s.ugr.es/keel/dataset/data/classification/newthyroid.zip",
    'saheart': "https://sci2s.ugr.es/keel/dataset/data/classification/saheart.zip",
    'shuttle': "https://sci2s.ugr.es/keel/dataset/data/classification/shuttle.zip",
    'sonar': "https://sci2s.ugr.es/keel/dataset/data/classification/sonar.zip",
    'soybean': "https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-small.data",
    'spectfheart': "https://sci2s.ugr.es/keel/dataset/data/classification/spectfheart.zip",
    'tae': "https://sci2s.ugr.es/keel/dataset/data/classification/tae.zip",
    'vehicle': "https://sci2s.ugr.es/keel/dataset/data/classification/vehicle.zip",
    'wine': "https://sci2s.ugr.es/keel/dataset/data/classification/wine.zip",
    'zoo': "https://sci2s.ugr.es/keel/dataset/data/classification/zoo.zip",
   }

import time
def download_raw_data_of_collection_1():
    path = 'raw data/'

    zip_files = ['appendicitis', 'bupa', 'ecoli', 'glass', 'haberman', 'hayesroth', 'heart', 'ionosphere', 'iris',
                 'led7digit', 'monk2', 'movement_libras', 'newthyroid', 'saheart', 'sonar', 'spectfheart', 'tae',
                 'vehicle', 'wine', 'zoo',]

    for name in zip_files:
        # Download file
        print('Data set', name, 'starting downloaded.')
        response = requests.get(links_to_data_sets[name])
        time.sleep(5)

        print(response)
        file_name = path + name + '_data.zip'
        open(file_name, 'wb').write(response.content)

        # Extract zip file
        zipfile.ZipFile(file_name, 'r').extractall(path)

        # Delete zip file
        os.remove(file_name)

        # Print progress
        print('Data set', name, 'successfully downloaded.')



    data_files = ['soybean']

    for name in data_files:
        # Download file
        print('Data set', name, 'starting downloaded.')
        response = requests.get(links_to_data_sets[name])
        #time.sleep(5)
        print(response)


        file_name = path + name + '-small.data'

        open(file_name, 'wb').write(response.content)

        # Print progress
        print('Data set', name, 'successfully downloaded.')


def download_raw_data_of_collection_4():
    path = 'raw data'

    zip_files = ['banana', 'letter', 'shuttle']

    for name in zip_files:
        # Download file
        response = requests.get(links_to_data_sets[name])
        file_name = path + name + '_data.zip'
        open(file_name, 'wb').write(response.content)

        # Extract zip file
        zipfile.ZipFile(file_name, 'r').extractall(path)

        # Delete zip file
        os.remove(file_name)

        # Print progress
        print('Data set', name, 'successfully downloaded.')

    cifar_files = ['cifar-10', 'cifar-100']

    for name in cifar_files:
        # Download file
        response = requests.get(links_to_data_sets[name])
        file_name = path + name + '-python.tar.gz'
        open(file_name, 'wb').write(response.content)

        # Open file
        file = tarfile.open(file_name)

        # Extract file
        file.extractall(path)

        file.close()

        # Delete tar file
        os.remove(file_name)

        # Print progress
        print('Data set', name, 'successfully downloaded.')

    # Download mnist data
    links = links_to_data_sets['mnist']
    for link in links:
        response = requests.get(link)
        file_name = path + link.split('/')[-1]
        open(file_name, 'wb').write(response.content)

        # Extract file
        with gzip.open(file_name, 'rb') as f_in:
            with open(file_name.split('.')[0], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Delete gz file
        os.remove(file_name)

    # Print progress
    print('Data set mnist successfully downloaded.')
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_data_from_dat_file(dataset, folder):
    # Read raw data
    f = open('raw data/' + dataset + '.dat')
    content = f.readlines()

    # Extract number of objects and number of features
    while content[0][0] == '@':
        content.pop(0)
    n = len(content)
    d = len(content[0].split(','))

    # Extract feature values
    ls = []
    for i in range(n):
        if content[i] == '\n':
            continue
        ls.append([])
        for j in range(d):
            entry = content[i].split(',')[j]
            if '\n' in entry:
                entry = entry.replace('\n', '')
            ls[i].append(entry)
    # Create DataFrame
    df = pd.DataFrame(ls)

    return df


def preprocess_and_export_old(df, dataset,folder):

    # Convert all columns except last one to float
    for col in df.columns[:-1]:
        df[col] = df[col].astype('float64')

    # Adjust column names
    df.columns = ['x' + str(i) for i in range(len(df.columns) - 1)] + ['class']

    # Encode class column as categorical
    categories = df['class'].unique()
    df['class'] = pd.Categorical(df['class'], categories).codes

    # Standardize features
    scaler = StandardScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])



    df['constraint'] = 0

    # Her bir sınıftan rastgele bir satır seç ve 'constraint' değerini 1 yap
    random_indices = df.groupby('class', group_keys=False).apply(lambda x: x.sample(n=1, random_state=42)).index
    df.loc[random_indices, 'constraint'] = 1

    # Sonucu göster
    print(df)

    # Export processed data set
    df.to_csv(folder+'/' + dataset + '_data.csv', index=False)




def preprocess_and_export(df, dataset,folder):
    print("preprocess_and_export", dataset)

    # Convert all columns except last one to float
    for col in df.columns[:-1]:
        df[col] = df[col].astype('float64')

    # Adjust column names
    df.columns = ['x' + str(i) for i in range(len(df.columns) - 1)] + ['class']

    # Encode class column as categorical
    categories = df['class'].unique()
    df['class'] = pd.Categorical(df['class'], categories).codes

    # Standardize features
    scaler = StandardScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])



    df['constraint'] = 0

    percentages = [0.05, 0.10, 0.15,0.2]
    df_list = {}

    for pct in percentages:
        df_copy = df.copy()
        total_rows = len(df_copy)  # Tüm veri noktalarının sayısı
        num_selected = max(1, round(total_rows * pct))  # En az 1 olmasını sağla
        class_count = df_copy['class'].nunique()  # Sınıf sayısı
        if num_selected < class_count:
            num_selected=class_count

        # Tüm veri üzerinden rastgele num_selected kadar satır seç
        selected_indices = np.random.choice(df_copy.index, num_selected, replace=False)
        df_copy.loc[selected_indices, 'constraint'] = 1


        # Oluşan dataframe'i kaydet
        df_list[pct] = df_copy
        df_copy.to_csv(f"{folder}/{dataset}_data_{int(pct*100)}.csv", index=False)

    # Sonuçları göstermek için
    for pct, df_res in df_list.items():
        print(f"\nDataFrame with {int(pct*100)}% constraint:\n")
        print(df_res.head())  # İlk birkaç satırı göster








def process_dataset(dataset, folder):
    # Read raw data
    df = get_data_from_dat_file(dataset, folder)

    # Perform preprocessing
    preprocess_and_export(df, dataset, folder)

def process_dataset_breast_cancer():
    # Read raw data
    X, y = load_breast_cancer(return_X_y=True)

    # Create DataFrame
    df = pd.DataFrame(np.concatenate((X, y.reshape(-1, 1)), axis=1))

    # Perform preprocessing
    preprocess_and_export(df, 'breast_cancer', folder = 'modified data')






def process_dataset_saheart():
    # Read raw data
    df = get_data_from_dat_file('saheart', 'COL1')

    # Add column names to encode one of the features as a binary feature
    df.columns = ['Sbp', 'Tobacco', 'Ldl', 'Adiposity', 'Famhist', 'Typea', 'Obesity', 'Alcohol', 'Age', 'class']
    df = df.replace({'Famhist': {'Present': '1', 'Absent': '0'}})

    # Perform preprocessing
    preprocess_and_export(df, 'saheart', folder = 'modified data')


def process_dataset_soybean():
    # Read raw data
    df = pd.read_table('raw data/soybean-small.data', sep=',', header=None)

    # Perform preprocessing
    preprocess_and_export(df, 'soybean', folder = 'modified data')


def test_character(datasetnamelist,folder):
    test_characters=[]
    test_characters.append(["Test Name","nfeature","nrow","ncluster","cluster_capacity"])
    for i in datasetnamelist:
        df=pd.read_csv(folder+'/'+i+'_data.csv')
        ncluster = len(df["class"].unique())
        cluster_capacity = df["class"].value_counts().max()
        nrow, nfeature=df.shape[0],df.shape[1]-2
        test_characters.append([i,nfeature,nrow,ncluster,cluster_capacity])

    test_characters_df=pd.DataFrame(test_characters)
    test_characters_df.to_excel("Real_ConsCluster_Test_Character.xlsx")


# %% Process data sets of collection COL1
def prepare_collection_1():
    folder = 'modified data'
    process_dataset('appendicitis', folder)
    process_dataset_breast_cancer()
    process_dataset('bupa', folder)
    process_dataset('ecoli', folder)
    process_dataset('glass', folder)
    process_dataset('haberman', folder)
    process_dataset('hayes-roth', folder)
    process_dataset('heart', folder)
    process_dataset('ionosphere', folder)
    process_dataset('iris', folder)
    process_dataset('led7digit', folder)
    process_dataset('monk-2', folder)

    process_dataset('movement_libras', folder)
    process_dataset('newthyroid', folder)
    process_dataset_saheart()
    process_dataset('sonar', folder)
    process_dataset_soybean()
    process_dataset('spectfheart', folder)
    process_dataset('tae', folder)
    process_dataset('vehicle', folder)
    process_dataset('wine', folder)
    process_dataset('zoo', folder)





    print('Data sets from collection COL1 sucessfully processed.')
#download_raw_data_of_collection_1()
prepare_collection_1()

dataset_list = ['appendicitis',
                'breast_cancer',
                'bupa',
                'ecoli',
                'glass',
                'haberman',
                'hayes-roth',
                'heart',
                'ionosphere',
                'iris',
                'led7digit',
                'monk-2',
                'movement_libras',
                'newthyroid',
                'saheart',
                'sonar',
                'soybean',
                'spectfheart',
                'tae',
                'vehicle',
                'zoo']
#test_character(dataset_list,folder = 'modified data')