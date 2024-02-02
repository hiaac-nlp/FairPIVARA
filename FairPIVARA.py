import torch
import json
import open_clip
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import itertools
import time
import pandas as pd
from scipy import ndimage
import itertools as it
import numpy as np
import scipy.special
import scipy.stats
from sklearn.metrics.pairwise import cosine_similarity
import argparse

GPU = 4
MAIN_PATH = '/hadatasets/MMBias'
DATASET_PATH = '/hadatasets/MMBias/data'
LANGUAGE_PATH = 'data'
LANGUAGE = 'en'
ft_open_clip = 'False'
adapter = 'False'
CONCEPTS='Disability/Mental|Disability,Disability/Non-Disabled,Disability/Physical|Disability,Nationality/American,Nationality/Arab,Nationality/Chinese,Nationality/Mexican,Religion/Buddhist,Religion/Christian,Religion/Hindu,Religion/Jewish,Religion/Muslim,Sexual|Orientation/Heterosexual,Sexual|Orientation/LGBT'
weighted_list='False'
add_signal = 'True'
sorted_df_similarities = 'True'
top_similar = 15

# device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# with open(f'{MAIN_PATH}/{LANGUAGE_PATH}/{LANGUAGE}_textual_phrases.txt') as f:
with open(f'{MAIN_PATH}/{LANGUAGE_PATH}/{LANGUAGE}_textual_phrases.txt') as f:
    text_dataset = json.load(f)

labels = {}
labels['unpleasant_phrases'] = text_dataset['unpleasant_phrases']
labels['pleasant_phrases'] = text_dataset['pleasant_phrases']
del text_dataset['unpleasant_phrases'], text_dataset['pleasant_phrases']

number_concepts = len(labels['unpleasant_phrases']) + len(labels['pleasant_phrases'])

print(">>>>>>> Loading model")
if ft_open_clip == 'True':
    if adapter is None:
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:hiaac-nlp/CAPIVARA')
        tokenizer = open_clip.get_tokenizer('hf-hub:hiaac-nlp/CAPIVARA')
    else:
        model = OpenCLIPAdapter(inference=True, devices=device)
        model.load_adapters(pretrained_adapter=args.adapter)
else:
    print('Using Baseline Model')
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

if ft_open_clip == 'True':
    vision_processor = preprocess_val
    text_tokenizer = tokenizer
else:
    vision_processor = preprocess
    text_tokenizer = tokenizer

EPS = np.finfo(float).eps

def mutual_information_2d(x, y, sigma=1, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    nmi: float
        the computed similariy measure
    """
    bins = (64, 64)
    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant', output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2))) / np.sum(
            jh * np.log(jh))) - 1
    else:
        mi = (np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1)) -
              np.sum(s2 * np.log(s2)))

    return mi

class MMBiasDataset(Dataset):
    def __init__(self, root_dir, image_preprocessor):
        self.root_dir = root_dir
        self.image_path = [os.path.join(root_dir, filename) for filename in os.listdir(root_dir) if filename.endswith('.jpg') and filename != '.DS_Store.jpg']
        self.transform = image_preprocessor

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image_path = self.image_path[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        return image
    
def all_features(concepts, dataset_path, vision_processor, model, labels, text_tokenizer, device, language, number_concepts, weighted_list, add_signal, sorted_df_similarities,top_similar):
    # Create the file sistem
    concepts = concepts.replace('|', ' ')
    # List thought all the concepts
    bias_list = [item for item in concepts.split(',')]
    all_features = {}
    # Calc the bias for each concept
    for bias in bias_list:
        folder1= bias.split('/')[0]
        folder2= bias.split('/')[1]
        # Load the imagens for the bias select in the loop
        custom_dataset = MMBiasDataset(f'{dataset_path}/Images/{bias}', image_preprocessor=vision_processor)
        dataloader = DataLoader(custom_dataset, batch_size=len(custom_dataset), shuffle=False)

        if language == 'en':
            template = "[CLASS] person"
        elif language == 'pt-br':
            template = "humano [CLASS]"

        my_labels = {}
        my_labels['unpleasant_phrases'] = [template.replace("[CLASS]", label) for label in labels['unpleasant_phrases']]
        my_labels['pleasant_phrases'] = [template.replace("[CLASS]", label) for label in labels['pleasant_phrases']]
        batch_texts = []
        batch_texts = my_labels['unpleasant_phrases'] + my_labels['pleasant_phrases']

        model.to(device)
        model.eval()

        # tokenize all texts in the batch
        # batch_texts_tok = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(device)
        batch_texts_tok_un = tokenizer(my_labels['unpleasant_phrases']).to(device)
        batch_texts_tok_ple = tokenizer(my_labels['pleasant_phrases']).to(device)

        all_images = []
        for image_input in dataloader:
            image_input = image_input.to(device)
            # compute the embedding of images and texts
            with torch.no_grad():
                image_features = F.normalize(model.encode_image(image_input), dim=-1).cpu()
            if folder1 not in all_features:
                all_features[folder1] = {}
            all_features[folder1][folder2] = image_features
            all_images.append(image_input)

        text_features_un = F.normalize(model.encode_text(batch_texts_tok_un), dim=-1).cpu()
        text_features_ple = F.normalize(model.encode_text(batch_texts_tok_ple), dim=-1).cpu()

    all_features['unpleasant_phrases']=text_features_un
    all_features['pleasant_phrases']=text_features_ple

    # batch_text and all_images used only in classification pipeline
    return all_features, batch_texts, all_images

all_features_values, batch_texts, all_images = all_features(CONCEPTS, DATASET_PATH, vision_processor, model, labels, text_tokenizer, device, LANGUAGE, number_concepts, weighted_list, add_signal, sorted_df_similarities,top_similar)
text_features = torch.cat((all_features_values['unpleasant_phrases'], all_features_values['pleasant_phrases']), 0)

combination_list = {}
for global_concept in all_features_values:
    try:
        micro_concept = list(all_features_values[global_concept].keys())
        combination_list[global_concept] = (list(itertools.combinations(micro_concept, 2)))
    except:
        pass

def image_to_text_retrieval(image_features, text_features, all_images, all_texts, sorted_df_similarities,dimensions=None):
    # images_selected = []
    df_list = []
    # all_images = [item for sublist in all_images for item in sublist]
    for image in range(image_features.shape[0]):
        # images_selected.append(all_images[image])
        similarities = []
        for i in range(len(text_features)):  #tensor[:, :500]
            if dimensions!= None:
                for dim in dimensions:
                    new_image_features = torch.cat([image_features[image][:dim], image_features[image][dim+1:]])
                    new_text_features = torch.cat([text_features[i][:dim], text_features[i][dim+1:]])
                scores = new_text_features @ new_image_features.t()  # shape: [batch_size, batch_size]
            else:
                new_text_features = text_features[i]
                new_image_features = image_features[image]
                scores = new_text_features @ new_image_features.t()
            item = {
                'score': scores.cpu(),
                'id': i,
                'text': all_texts[i]
                }
            similarities.append(item)
        similarities_df = pd.DataFrame(similarities)
        if sorted_df_similarities == 'True':
            sorted_df = similarities_df.sort_values(by='score', ascending=False)
            df_list.append(sorted_df)
        else:
            df_list.append(similarities_df)
    # return df_list, images_selected
    return df_list

def Convert(tup, di):
    di = dict(tup)
    return di

def add_list_signal_in_ziplist(ziplist, language):
    #check the score signal
    for i, item in enumerate(ziplist):
        if language == 'en':
            concept_compare = item[1].split(' ')[0]
        elif language == 'pt-br':
            concept_compare = item[1].split(' ')[1]
        if concept_compare in labels['unpleasant_phrases']:
            if type(ziplist[i][0]) != float and type(ziplist[i][0]) != int:
                ziplist[i] = ((-ziplist[i][0][0], -ziplist[i][0][1]), item[1])
            else:
                ziplist[i] = (-ziplist[i][0], ziplist[i][1])
        else:
            if type(ziplist[i][0]) != float and type(ziplist[i][0]) != int:
                ziplist[i] = ((ziplist[i][0][0], ziplist[i][0][1]), item[1])
            else:
                ziplist[i] = (ziplist[i][0], ziplist[i][1])
    return ziplist

def add_list_signal(temp_list, language):
    #check the score signal
    for item in temp_list:
        if language == 'en':
            concept_compare = item.split(' ')[0]
        elif language == 'pt-br':
            concept_compare = item.split(' ')[1]
        if concept_compare in labels['unpleasant_phrases']:
            temp_list[item] = (-temp_list[item][0], -temp_list[item][1])
        else:
            temp_list[item] = (temp_list[item][0], temp_list[item][1])
    return temp_list

def show_mean_result(all_bias, print_type, score_or_quant, language,top_similar,add_signal):
    for i in all_bias:
        print(i)
        for j in all_bias[i]:
            print(f'-- {j}')
            my_list_keys_to_print = list(all_bias[i][j].keys())
            my_list_values_to_print = list(all_bias[i][j].values())

            if score_or_quant == 'quant':
                item_v_list = []
                for item_v in my_list_values_to_print:
                    item_v_list.append(item_v[0])
            elif score_or_quant == 'score':
                item_v_list = []
                for item_v in my_list_values_to_print:
                    item_v_list.append(item_v[1])
            else:
                item_v_list = []
                for item_v in my_list_values_to_print:
                    item_v_list.append((item_v[0], item_v[1]))

            #if top similar not defined, get all values and keys
            top_similar = len(item_v_list)
            vk = sorted(zip(item_v_list,my_list_keys_to_print))[(len(item_v_list)-top_similar):]

            if score_or_quant!='both':
                vk = sorted(vk)

            for _,k in vk:
                print(k, end=',')
            print('')
            if score_or_quant == 'both':
                for v,_ in vk:
                    print(v[0], end=',')
                print('')
                for v,_ in vk:
                    print(v[1], end=',')
                print('')
            elif score_or_quant == 'both_operation':
                for v,_ in vk:
                    if v[0] < 0:
                        print(v[1]/-v[0], end=',')
                    else:
                        print(v[1]/v[0], end=',')
                print('')
            else:
                mean = 0
                for v,_ in vk:
                    print(v, end=',')
                    mean += v
                print(mean)
                print('')

def simple_bias_from_df(df_list):
    parcial_bias = {}
    list_of_concepts = []
    top_one_concepts = []
    # Add the firt "number of concepts" (default = 15) in the dict
    for df in df_list:
        if df.iloc[0].text.split(' ')[0] in labels['unpleasant_phrases']:
            top_one_concepts.append(0)
        else:
            top_one_concepts.append(1)

        for nc in range(top_similar):
            list_of_concepts.append((df.iloc[nc].text,df.iloc[nc].score))

    # Calc the concepts with sum of total itens
    for item in list_of_concepts:
        if item[0] not in parcial_bias:
            parcial_bias[item[0]] = (0,0)
        parcial_bias[item[0]] = (parcial_bias[item[0]][0] + 1, parcial_bias[item[0]][1] + item[1].item())
    # Calc the concepts with mean score
    if weighted_list == 'True':
        for item in parcial_bias:
            weight_parcial_bias[item] = (parcial_bias[item][0], parcial_bias[item][1]/parcial_bias[item][0])
        list_weight_parcial_bias = sorted(weight_parcial_bias.items(), key=lambda x: x[1][1], reverse=True)
        weight_parcial_bias = {}
        weight_parcial_bias = Convert(list_weight_parcial_bias, weight_parcial_bias)

    if weighted_list == 'False':
        temp_list = parcial_bias
    else:
        temp_list = weight_parcial_bias

    if add_signal == 'True':
        temp_list = add_list_signal(temp_list, language=LANGUAGE)
    return temp_list, top_one_concepts


def unique_bias_mean(X,dimensions=None):
    df_list = image_to_text_retrieval(X, text_features, all_images, batch_texts, sorted_df_similarities,dimensions=dimensions)
    bias,label_list = simple_bias_from_df(df_list)
    mean = 0
    for text in bias:
        mean += bias[text][1]
    return mean, label_list

def single_bias_mitigation_algorithm(X, n, theta):
    x = set()
    psi,_ = unique_bias_mean(X)  # Substitua captions_vi e captions_vt com suas legendas
    for d in range(X.size(1)):
        
        X_temp = X.clone()

        # Calcular o MI
        d_bias, labels = unique_bias_mean(X_temp,[d])

        mi = mutual_information_2d(X_temp[:, d],labels)
        
        if mi < theta:
            # Se MI for menor que o limiar, calcular o novo viés
            # psi_d = compute_bias(X_temp, Y_temp, A_temp, B_temp)
            if abs(d_bias) < abs(psi):
                x.add((d, d_bias))

    # Ordenar e selecionar as dimensões a serem removidas
    # z = sorted(x, key=lambda item: item[1])[:n]
    z = sorted(x, key=lambda item: item[1],reverse=True)[:n]
    # print(f'Valores de Z: {z}')

    # Remover as dimensões selecionadas
    best_dimension_bias = (99999,[])
    removed_dimensions = []
    for dim, _ in z:
        # X = torch.cat([X[:, :dim], X[:, dim+1:]], dim=1)
        removed_dimensions.append(dim)
        psi,_ = unique_bias_mean(X,removed_dimensions)
        if abs(psi) < abs(best_dimension_bias[0]):
            best_dimension_bias = (psi,removed_dimensions)
    return best_dimension_bias

theta = [0, 0.25, 0.5, 0.75, 1]
concepts = CONCEPTS.replace('|', ' ')
print(f'Theta, Concept, O-Bias, D-Bias, R-Dimensions')
for t in theta: 
    # List thought all the concepts
    bias_list = [item for item in concepts.split(',')]
    for bias in bias_list:
        folder1= bias.split('/')[0]
        folder2= bias.split('/')[1]
        best_dimension_bias = single_bias_mitigation_algorithm(all_features_values[folder1][folder2],54,t)
        print(f'{t}, {bias}, {unique_bias_mean(all_features_values[folder1][folder2])[0]}, {best_dimension_bias[0]}, {best_dimension_bias[1]}')
