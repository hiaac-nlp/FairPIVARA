import argparse
import sys
import os
import json
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import tqdm
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import pandas as pd
import open_clip
import itertools
import numpy as np
from utils import mutual_information_2d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", help="Path to model checkpoint", )
    parser.add_argument("--dataset-path", help="Path to validation/test dataset")
    parser.add_argument("--translation", choices=["english", "marian", "google"], required=False)
    parser.add_argument("--language", default="en", choices=["en", "pt-br", "xh", "hi"], required=False)
    parser.add_argument("--batch", type=int, help="Batch size", )
    parser.add_argument("--ft-open-clip", type=str, default="False", required=False,
                        help="Indicates whether model is fine-tuned (True) or is the original OpenCLIP (False)")
    parser.add_argument("--gpu", help="GPU", )
    parser.add_argument("--adapter", default=None, required=False, help="Load the adapter weights")
    parser.add_argument("--concepts", type=str, required=False)
    parser.add_argument("--number-concepts", default=None ,type=int, required=False, help="Number of atributes returned by CLIP for target group")
    parser.add_argument("--top-similar", default=None ,type=int, required=False, help="Top atributes returned by CLIP for target group")
    parser.add_argument("--task", type=str, choices=["classification","comparison"], help="Task to be done" )
    parser.add_argument("--weighted-list", default=None)
    parser.add_argument("--print", default="json", choices=["json", "exel", "file"])
    parser.add_argument("--score-or-quant", default="score", choices=["quant", "score", "both","both_operation"])
    parser.add_argument("--add-signal", default=None, choices=["True", "False"])
    parser.add_argument("--sorted-df-similarities", default=None, choices=["True", "False"])
    parser.add_argument("--sort-order", default=None, choices=["signal", "module"])

    return parser.parse_args()

# class MMBiasDataset(Dataset) used to create a custom dataset for the images in the classification task
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

def extract_bias(concepts, dataset_path, vision_processor, model, labels, text_tokenizer, device, language, number_concepts, weighted_list, add_signal, sorted_df_similarities,top_similar):
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

        model.to(device)
        model.eval()

        # tokenize all texts in the batch
        # batch_texts_tok = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(device)
        batch_texts_tok_un = tokenizer(my_labels['unpleasant_phrases']).to(device)
        batch_texts_tok_ple = tokenizer(my_labels['pleasant_phrases']).to(device)

        for image_input in dataloader:
            image_input = image_input.to(device)
            # compute the embedding of images and texts
            with torch.no_grad():
                image_features = F.normalize(model.encode_image(image_input), dim=-1).cpu()
            if folder1 not in all_features:
                all_features[folder1] = {}
            all_features[folder1][folder2] = image_features
        
        text_features_un = F.normalize(model.encode_text(batch_texts_tok_un), dim=-1).cpu()
        text_features_ple = F.normalize(model.encode_text(batch_texts_tok_ple), dim=-1).cpu()

    all_features['unpleasant_phrases']=text_features_un
    all_features['pleasant_phrases']=text_features_ple
    
    combination_list = {}
    for global_concept in all_features:
        try:
            micro_concept = list(all_features[global_concept].keys())
            combination_list[global_concept] = (list(itertools.combinations(micro_concept, 2)))
        except:
            pass
    for conj_combinations in combination_list:
        for combination in combination_list[conj_combinations]:
            print(f'----------------- {combination[0]} vs {combination[1]} -----------------')
            #print(f'Original Bias: {compute_bias(all_features[conj_combinations][combination[0]],all_features[conj_combinations][combination[1]],all_features['unpleasant_phrases'],all_features['pleasant_phrases'])}')
            print(f'Original Bias: {compute_bias(all_features[conj_combinations][combination[0]],all_features[conj_combinations][combination[1]],all_features["unpleasant_phrases"],all_features["pleasant_phrases"])}')
            #print(compute_bias(all_features['Sexual Orientation']['Heterosexual'],all_features['Sexual Orientation']['LGBT'],all_features['unpleasant_phrases'],all_features['pleasant_phrases']))
            #newX, newY, newA, newB = bias_mitigation_algorithm(all_features['Sexual Orientation']['Heterosexual'],all_features['Sexual Orientation']['LGBT'],all_features['unpleasant_phrases'],all_features['pleasant_phrases'],54,0.5)
            newX, newY, newA, newB = bias_mitigation_algorithm(all_features[conj_combinations][combination[0]],all_features[conj_combinations][combination[1]],all_features['unpleasant_phrases'],all_features['pleasant_phrases'],54,0.5)
            print('Best Result: {compute_bias(newX,newY,newA,newB)}')



def compute_bias(X, Y, A, B):
    def phi(w, A, B):
        phi_w = torch.mean(torch.stack([torch.mean(torch.cosine_similarity(w[None,:], a[None,:])) for a in A]), dim=0) - torch.mean(torch.stack([torch.mean(torch.cosine_similarity(w[None,:], b[None,:])) for b in B]), dim=0)
        return phi_w

    # Implementação do cálculo do viés conforme a fórmula (1) em PyTorch
    def bias(X, Y, A, B):
        mean_X = torch.mean(torch.stack([phi(w, A, B) for w in X]), dim=0)
        mean_Y = torch.mean(torch.stack([phi(w, A, B) for w in Y]), dim=0)
        std_dev_XY = torch.std(torch.stack([phi(w, A, B) for w in torch.cat((X, Y))]), dim=0)

        return (mean_X - mean_Y) / std_dev_XY

    # Substitua os argumentos relevantes com seus tensores PyTorch de imagem e texto
    computed_bias = bias(X, Y, A, B)

    return computed_bias

def bias_mitigation_algorithm(X, Y, A, B, n, theta):
    x = set()
    psi = compute_bias(X, Y, A, B)  # Substitua captions_vi e captions_vt com suas legendas
    # print(f'Initial Bias: {psi}')
    # print(f'temps dimension: {X.size()},{Y.size()},{A.size()},{B.size()}')
    for d in range(X.size(1)):
        X_temp = X.clone()
        Y_temp = Y.clone()
        A_temp = A.clone()
        B_temp = B.clone()
        
        # Remover a dimensão d
        # print(f'X_temp[:, :d]: {X_temp[:, :d].size()}, X_temp[:, d+1:]]: {X_temp[:, d+1:].size()}')
        X_temp = torch.cat([X_temp[:, :d], X_temp[:, d+1:]], dim=1)
        Y_temp = torch.cat([Y_temp[:, :d], Y_temp[:, d+1:]], dim=1)
        A_temp = torch.cat([A_temp[:, :d], A_temp[:, d+1:]], dim=1)
        B_temp = torch.cat([B_temp[:, :d], B_temp[:, d+1:]], dim=1)
        # print(f'dimension {d} removed')
        # print(f'New temps dimensions: {X_temp.size()},{Y_temp.size()},{A_temp.size()},{B_temp.size()}')
        # Calcular o MI
        mi = compute_bias(X_temp, Y_temp, A_temp, B_temp)
        # print('New bias')
        # print(f'mi: {mi}, theta: {theta}')
        if mi < theta:
            # Se MI for menor que o limiar, calcular o novo viés
            # psi_d = compute_bias(X_temp, Y_temp, A_temp, B_temp) 
            psi_d = mi

            if psi_d < psi:
                x.add((d, psi_d))
            # print(f'psi_d (bias inside the loop): {psi_d}')
        # print('---------------------------------------------------')
    # print('Out of loop')
    # Ordenar e selecionar as dimensões a serem removidas
    z = sorted(x, key=lambda item: item[1])[:n]
    # print(f'Valores de Z: {z}')

    # Remover as dimensões selecionadas
    for dim, _ in z:
        X = torch.cat([X[:, :dim], X[:, dim+1:]], dim=1)
        Y = torch.cat([Y[:, :dim], Y[:, dim+1:]], dim=1)
        A = torch.cat([A[:, :dim], A[:, dim+1:]], dim=1)
        B = torch.cat([B[:, :dim], B[:, dim+1:]], dim=1)
        print('Temporary bias - without dim {dim}: {compute_bias(X,Y,A,B}')
    # print(f'New X : {X.size()}')
    # print(X)
    return X, Y, A, B

if __name__ == "__main__":
    args = parse_args()
    print(args)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    with open(f'{args.dataset_path}/{args.language}_textual_phrases.txt') as f:
        text_dataset = json.load(f)

    labels = {}
    labels['unpleasant_phrases'] = text_dataset['unpleasant_phrases']
    labels['pleasant_phrases'] = text_dataset['pleasant_phrases']
    del text_dataset['unpleasant_phrases'], text_dataset['pleasant_phrases']

    print(">>>>>>> Loading model")
    if args.ft_open_clip == 'True':
        if args.adapter is None:
            model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:hiaac-nlp/CAPIVARA')
            tokenizer = open_clip.get_tokenizer('hf-hub:hiaac-nlp/CAPIVARA')
        else:
            model = OpenCLIPAdapter(inference=True, devices=device)
            model.load_adapters(pretrained_adapter=args.adapter)
    else:
        print('Using Baseline Model')
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        print('Model Loaded')
        tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # Define the model used to the classification
    if args.ft_open_clip == 'True':
        vision_processor = preprocess_val
        text_tokenizer = tokenizer 
    else:
        vision_processor = preprocess
        text_tokenizer = tokenizer

    # Task selected
    if args.task == 'classification':
        if args.add_signal == None:
            add_signal = 'True'
        else:
            add_signal = args.add_signal
        if args.number_concepts == None:
            number_concepts = len(labels['unpleasant_phrases']) + len(labels['pleasant_phrases'])
        else:
            number_concepts = args.number_concepts
        if args.weighted_list == None:
            weighted_list = 'True'
        else:
            weighted_list = args.weighted_list
        if args.sorted_df_similarities == None:
            sorted_df_similarities = 'True'
        else:
            sorted_df_similarities = args.sorted_df_similarities

        all_bias = extract_bias(args.concepts, args.dataset_path, vision_processor, model, labels, text_tokenizer, device, args.language, number_concepts, weighted_list, add_signal, sorted_df_similarities,args.top_similar)
