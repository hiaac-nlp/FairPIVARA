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
import random

#sys.path.append("/work/diego.moreira/CLIP-PtBr/clip_pt/src/")

#from models.open_CLIP import OpenCLIP
#from models.open_CLIP_adapter import OpenCLIPAdapter


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
    parser.add_argument("--extract-top-similar", required=False, help="Top atributes returned by CLIP for target group")
    parser.add_argument("--view-top-similar", required=False, help="Top atributes returned by CLIP for target group")
    parser.add_argument("--task", type=str, choices=["classification","comparison"], help="Task to be done" )
    parser.add_argument("--weighted-list", default=None)
    parser.add_argument("--print", default="json", choices=["json", "exel", "file", "pandas"])
    parser.add_argument("--score-or-quant", default="score", choices=["quant", "score", "both","both_operation"])
    parser.add_argument("--add-signal", default=None, choices=["True", "False"])
    parser.add_argument("--sorted-df-similarities", default=None, choices=["True", "False"])
    parser.add_argument("--sort-order", default=None, choices=["signal", "module"])
    parser.add_argument("--remove-dimensions-list", type=str, default=None, required=False,
                        help="File with list of dimensions to remove")
    parser.add_argument("--repetitions", type=int, default=1, required=False, help="Number of repetitions")
    parser.add_argument("--bias-type", default="same_as_X", choices=['same_as_selected','random_text','random'])

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

# 
def image_to_text_retrieval(image_features, text_features, all_images, all_texts, sorted_df_similarities):
    images_selected = []
    df_list = []
    all_images = [item for sublist in all_images for item in sublist]
    for image in range(image_features.shape[0]):
        images_selected.append(all_images[image])
        similarities = []
        for i in range(len(text_features)):  #tensor[:, :500]
            scores = text_features[i] @ image_features[image].t()  # shape: [batch_size, batch_size]
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
    return df_list, images_selected

def classification(model, image_dataloader, labels, tokenizer, device, language, sorted_df_similarities, remove_dimensions,bias_type):
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
    batch_texts_tok = tokenizer(batch_texts).to(device)

    all_images = []
    for image_input in image_dataloader:
        image_input = image_input.to(device)
        # compute the embedding of images and texts
        with torch.no_grad():
            image_features = F.normalize(model.encode_image(image_input), dim=-1).cpu()
            text_features = F.normalize(model.encode_text(batch_texts_tok), dim=-1).cpu()
        all_images.append(image_input)
    
    if remove_dimensions == ['']:
        remove_dimensions_size = 0
    else:
        remove_dimensions_size = len(remove_dimensions)
    if remove_dimensions_size > 0:
        if bias_type == 'random':
            id_list = random.sample(range(text_features.size()[1]), text_features.size()[1]-remove_dimensions_size)
            i_features = image_features[:,id_list]
            t_features = text_features[:,id_list]
        else:
            # remove a list of dimensions if required from the image and text embeddings
            i_features = None
            for i, dim in enumerate(range(image_features.size()[1])):
                if str(i) not in remove_dimensions:
                    if i_features == None:
                        i_features = image_features[:,i][:,None]
                    else:
                        i_features = torch.cat([i_features, image_features[:,i][:,None]], dim=1)
            if bias_type == 'same_as_selected':
                t_features = None
                for i, dim in enumerate(range(text_features.size()[1])):
                    if str(i) not in remove_dimensions:
                        if t_features == None:
                            t_features = text_features[:,i][:,None]
                        else:
                            t_features = torch.cat([t_features, text_features[:,i][:,None]], dim=1)
            elif bias_type == 'random_text':
                id_list = random.sample(range(text_features.size()[1]), text_features.size()[1]-remove_dimensions_size)
                t_features = text_features[:,id_list]
        df_list, images_selected = image_to_text_retrieval(i_features, t_features, all_images, batch_texts, sorted_df_similarities)
    else:
        df_list, images_selected = image_to_text_retrieval(image_features, text_features, all_images, batch_texts, sorted_df_similarities)
    return df_list, images_selected

def Convert(tup, di):
    di = dict(tup)
    return di

def show_results(all_bias, print_type, score_or_quant, language,top_similar,add_signal):

    # #TODO: VERIFICAR OS RESULTADOS COM MULTIPLOS VALORES NA LISTA E A QUANTIDADE DE REPETICOES
    # # print to Json
    # for bias in all_bias:
    if print_type == 'json':
        print(all_bias)
    # print to the exel
    elif print_type == 'exel':
        for i in all_bias:
            for j in all_bias[i]:
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
                if top_similar == None:
                    vk = sorted(zip(item_v_list,my_list_keys_to_print))
                else:
                    vk = sorted(zip(item_v_list,my_list_keys_to_print))[(len(item_v_list)-top_similar):]

                if add_signal == 'True':
                    # add signal
                    vk = add_list_signal_in_ziplist(vk,language)
                
                if score_or_quant=='both':
                    vk = sorted(vk,key=lambda x: x[0][1])
                
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
                    for v,_ in vk:
                        print(v, end=',')
                    print('')

        for i in all_bias:
            print(i)
            for j in all_bias[i]:
                print(f'-- {j}')
    # print to the file
    elif print_type == 'file':
        with open(f"{language}_all_bias_text_classification.json", "w") as outfile: 
            json.dump(all_bias, outfile)
    elif print_type == 'pandas': 
        data_list = []
        for i in all_bias:
            for j in all_bias[i]:
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
                if top_similar == None:
                    vk = sorted(zip(item_v_list,my_list_keys_to_print))
                else:
                    vk = sorted(zip(item_v_list,my_list_keys_to_print))[(len(item_v_list)-top_similar):]

                if add_signal == 'True':
                    # add signal
                    vk = add_list_signal_in_ziplist(vk,language)
                
                if score_or_quant=='both':
                    vk = sorted(vk,key=lambda x: x[0][1])

                for v,k in vk:
                    data_list.append([i,j,k,v[0],v[1]])
        df = pd.DataFrame(data_list, columns = ['Global Concept', 'Micro Concept', 'Concept', 'Quant', 'Score'])
        user = os.environ.get('USER', os.environ.get('USERNAME'))
        if args.remove_dimensions_list!='':
            removed_dimensions = args.remove_dimensions_list.split('/')[-1]
        else: 
            removed_dimensions = ''
        df.to_csv(f'/home/{user}/FairPIVARA/results/violin/Enviroment:language-{language},task-{args.task},score_or_quant-{args.score_or_quant},extract_top_similar-{args.extract_top_similar},view_top_similar-{args.view_top_similar},remove_dimensions_list-{removed_dimensions},repetitions-{args.repetitions},bias_type-{args.bias_type}', index=False)
        print(df)

def add_list_signal(temp_list, language):
    #check the score signal
    for item in temp_list:
        if language == 'en':
            concept_compare = item.split(' ')[0]
        elif language == 'pt-br':
            concept_compare = item.split(' ')[1]
        if concept_compare in labels['unpleasant_phrases']:
            temp_list[item] = (-temp_list[item][0], -temp_list[item][1])
    return temp_list

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

def extract_bias(concepts, dataset_path, vision_processor, model, labels, text_tokenizer, device, language, number_concepts, weighted_list, add_signal, sorted_df_similarities,top_similar,remove_dimensions_list,repetitions,bias_type):
    repetition_all_bias = {}
    for repeted in range(repetitions):
        if repeted % 100:
            print(repeted)
        # Create the file sistem
        concepts = concepts.replace('|', ' ')
        # List thought all the concepts
        bias_list = [item for item in concepts.split(',')]
        all_bias = {}
        # Calc the bias for each concept
        for bias in bias_list:
            parcial_bias = {}
            weight_parcial_bias = {}
            folder1= bias.split('/')[0]
            folder2= bias.split('/')[1]
            # Load the imagens for the bias select in the loop
            custom_dataset = MMBiasDataset(f'{dataset_path}/Images/{bias}', image_preprocessor=vision_processor)
            dataloader = DataLoader(custom_dataset, batch_size=len(custom_dataset), shuffle=False) 

            if remove_dimensions_list != None:
                remove_dimensions = remove_dimensions_list[folder2]
            else:
                remove_dimensions = None
            
            # DO the classification
            df_list, images_selected = classification(model=model, image_dataloader=dataloader, labels=labels, tokenizer=text_tokenizer, device=device, language=language, sorted_df_similarities=sorted_df_similarities, remove_dimensions=remove_dimensions,bias_type=bias_type)
            list_of_concepts = []

            # Add the firt "number of concepts" (default = 15) in the dict
            for df in df_list:
                if top_similar == None:
                    for nc in range(len(df)):
                        list_of_concepts.append((df.iloc[nc].text,df.iloc[nc].score))
                else:
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

            # if add_signal == 'True':
            #     temp_list = add_list_signal(temp_list, language=language)
            
            if folder1 not in all_bias:
                all_bias[folder1] = {}
            
            # Select the order, by itens quant or score
            # only the first "number_concepts" , if number = None all will be collected
            if weighted_list == 'True':
                all_bias[folder1][folder2] = dict(itertools.islice(weight_parcial_bias.items(), number_concepts))
            else:
                all_bias[folder1][folder2] = dict(itertools.islice(parcial_bias.items(), number_concepts))

        if repetition_all_bias == {}:
            for global_concept in all_bias:
                if global_concept not in repetition_all_bias:
                    repetition_all_bias[global_concept] = {}
                for micro_concept in all_bias[global_concept]:
                    repetition_all_bias[global_concept][micro_concept] = all_bias[global_concept][micro_concept]
        else:
            for global_concept in all_bias:
                for micro_concept in all_bias[global_concept]:
                    for item in all_bias[global_concept][micro_concept]:
                        if item not in repetition_all_bias[global_concept][micro_concept]:
                            repetition_all_bias[global_concept][micro_concept][item] = all_bias[global_concept][micro_concept][item]
                        else:
                            repetition_all_bias[global_concept][micro_concept][item] = (repetition_all_bias[global_concept][micro_concept][item][0]+all_bias[global_concept][micro_concept][item][0], repetition_all_bias[global_concept][micro_concept][item][1]+all_bias[global_concept][micro_concept][item][1])
    return repetition_all_bias

def old_caliskan_test (all_bias):
    for global_concept in all_bias:
        print(f'----------------- {global_concept} -----------------')
        micro_concept = list(all_bias[global_concept].keys())
        combination_list = list(itertools.combinations(micro_concept, 2))
        for combination in combination_list:
            print(f'----------------- {combination[0]} vs {combination[1]} -----------------')
            total_x_bias = []
            for x_bias in all_bias[global_concept][combination[0]]:
                total_x_bias.append(all_bias[global_concept][combination[0]][x_bias][1])
                num_images = all_bias[global_concept][combination[0]][x_bias][0]
            value_total_x_bias = sum(total_x_bias[int(len(total_x_bias)/2):])/(len(all_bias[global_concept][combination[0]])/2) - sum(total_x_bias[:int(len(total_x_bias)/2)])/(len(all_bias[global_concept][combination[0]])/2)
            mean_value_total_x_bias = value_total_x_bias/num_images
            
            total_y_bias = []
            for y_bias in all_bias[global_concept][combination[1]]:
                total_y_bias.append(all_bias[global_concept][combination[1]][y_bias][1])
                num_images = all_bias[global_concept][combination[1]][y_bias][0]
            value_total_y_bias = sum(total_y_bias[int(len(total_y_bias)/2):])/(len(all_bias[global_concept][combination[0]])/2) - sum(total_y_bias[:int(len(total_y_bias)/2)])/(len(all_bias[global_concept][combination[0]])/2)
            mean_value_total_y_bias = value_total_y_bias/num_images

            print(f'bias: {(mean_value_total_x_bias - mean_value_total_y_bias)/ np.std([value_total_x_bias,value_total_y_bias]) }')

def phi_calc(df,number_concepts):
    image_bias_a = 0
    image_bias_b = 0
    for nc in range(0,int(number_concepts/2)):
        image_bias_a +=  df.iloc[nc].score
    image_bias_a = image_bias_a/int(number_concepts/2)
    for nc in range(int(number_concepts/2),number_concepts):
        image_bias_b +=  df.iloc[nc].score
    image_bias_b = image_bias_b/int(number_concepts/2)

    return image_bias_a - image_bias_b

def caliskan_test(concepts, dataset_path, vision_processor, model, labels, text_tokenizer, device, language, sorted_df_similarities,number_concepts, remove_dimensions,bias_type):
    # Create the file sistem
    concepts = concepts.replace('|', ' ')
    # List thought all the concepts
    bias_list = [item for item in concepts.split(',')]
    # Calc the bias for each concept
    all_concept_bias = {}
    all_bias = {}
    mean_all_bias = {}
    std_all = {}
    for bias in bias_list:
        folder1= bias.split('/')[0]
        folder2= bias.split('/')[1]
        # Load the imagens for the bias select in the loop
        custom_dataset = MMBiasDataset(f'{dataset_path}/Images/{bias}', image_preprocessor=vision_processor)
        dataloader = DataLoader(custom_dataset, batch_size=len(custom_dataset), shuffle=False) 
        # DO the classification
        df_list, images_selected = classification(model=model, image_dataloader=dataloader, labels=labels, tokenizer=text_tokenizer, device=device, language=language, sorted_df_similarities=sorted_df_similarities, remove_dimensions=remove_dimensions[folder2],bias_type=bias_type)
        # print(f'---{folder1}--- ---{folder2}---')
        images_phi = 0
        std_list = []
        for df in df_list:
            images_phi += phi_calc(df,number_concepts)
            std_list.append(phi_calc(df,number_concepts))
            
        if folder1 not in mean_all_bias:
            mean_all_bias[folder1] = {}
            all_bias[folder1] = {}
            std_all[folder1] = {}
        mean_all_bias[folder1][folder2] = images_phi/len(df_list)
        all_bias[folder1][folder2] = images_phi      
        std_all[folder1][folder2] = std_list

    combination_list = {}
    for global_concept in mean_all_bias:
        micro_concept = list(mean_all_bias[global_concept].keys())
        combination_list[global_concept] = (list(itertools.combinations(micro_concept, 2)))

    for conj_combinations in combination_list:
        for combination in combination_list[conj_combinations]:
            # print(f'----------------- {combination[0]} vs {combination[1]} -----------------')
            bias_std = np.std(std_all[conj_combinations][combination[0]]+std_all[conj_combinations][combination[1]])
            print(f'{combination[0]}, {combination[1]}: {(mean_all_bias[conj_combinations][combination[0]]-mean_all_bias[conj_combinations][combination[1]])/bias_std}')

        

if __name__ == "__main__":
    args = parse_args()

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
        if args.extract_top_similar == '':
            extract_top_similar = None
        else:
            extract_top_similar = int(args.extract_top_similar)
        if args.view_top_similar == '':
            view_top_similar = None
        else:
            view_top_similar = int(args.view_top_similar)
        
        if args.remove_dimensions_list == '':
            remove_dimensions_list = None
        else:
            with open(args.remove_dimensions_list) as f:
                lines = f.readlines()   
                remove_dimensions_list = {}
                for line in lines:
                    partition = line.split('[')
                    value = partition[0].split(',')
                    remove_dimensions_list[value[1].strip()] = partition[1].strip()[:-1].split(', ')
            

        print(f'Enviroment:task-{args.task},gpu-{args.gpu},score_or_quant-{args.score_or_quant},extract_top_similar-{args.extract_top_similar},view_top_similar-{args.view_top_similar},remove_dimensions_list-{args.remove_dimensions_list},repetitions-{args.repetitions},bias_type-{args.bias_type}')

        all_bias = extract_bias(args.concepts, args.dataset_path, vision_processor, model, labels, text_tokenizer, device, args.language, number_concepts, weighted_list, add_signal, sorted_df_similarities,extract_top_similar, remove_dimensions_list, args.repetitions, args.bias_type)
        show_results(all_bias, args.print, args.score_or_quant, args.language,view_top_similar,add_signal)

    elif args.task == 'comparison':
        print(args.remove_dimensions_list)
        if args.add_signal == None:
            add_signal = 'False'
        if args.weighted_list == None:
            weighted_list = 'False'
        if args.number_concepts == None:
            number_concepts = len(labels['unpleasant_phrases']) + len(labels['pleasant_phrases'])
        if args.sorted_df_similarities == None:
            sorted_df_similarities = 'False'
        if args.remove_dimensions_list == '':
            remove_dimensions_list = None
        else:
            print(f'------------{args.remove_dimensions_list}------------')
            with open(args.remove_dimensions_list) as f:
                lines = f.readlines()   
                remove_dimensions_list = {}
                for line in lines:
                    partition = line.split('[')
                    value = partition[0].split(',')
                    remove_dimensions_list[value[1].strip()] = partition[1].strip()[:-1].split(', ')

        caliskan_test(args.concepts, args.dataset_path, vision_processor, model, labels, text_tokenizer, device, args.language, sorted_df_similarities,number_concepts,remove_dimensions_list,args.bias_type)
        
        # all_bias = extract_bias(args.concepts, args.dataset_path, vision_processor, model, labels, text_tokenizer, device, args.language, number_concepts, weighted_list, add_signal, sorted_df_similarities)
        # old_caliskan_test(all_bias)
