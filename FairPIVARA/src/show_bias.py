import argparse
import os
import sys
import json
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import webdataset as wds
from torch.utils.data import DataLoader
import pandas as pd
import open_clip
import itertools
import random
from utils.mmbias_dataset import MMBiasDataset
import time
from utils.all_features import all_features
from utils.WEAT_test import Test
from statistics import mean 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", help="Path to model checkpoint", )
    parser.add_argument("--dataset-path", help="Path to validation/test dataset")
    parser.add_argument("--rude-level",  type=int, default=1, required=False, help="Words set used, original o less rude.")
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
    parser.add_argument("--top-type", default="equal", choices=['top','equal'], help="top: Show the 15 biggest values or equal: the top simila positives and negatives")
    parser.add_argument("--task", type=str, choices=["by_concept","cross_concept"], help="Task to be done" )
    parser.add_argument("--weighted-list", default=None)
    parser.add_argument("--print", default="json", choices=["json", "exel", "file", "pandas"])
    parser.add_argument("--score-or-quant", default="score", choices=["quant", "score", "both","both_operation"])
    parser.add_argument("--add-signal", default=None, choices=["True", "False"])
    parser.add_argument("--sorted-df-similarities", default=None, choices=["True", "False"])
    parser.add_argument("--sort-order", default=None, choices=["signal", "module"])
    parser.add_argument("--remove-dimensions-list", type=str, default=None, required=False,
                        help="File with list of dimensions to remove")
    parser.add_argument("--bias-type", default="same_as_X", choices=['same_as_selected','random_text','random','random','random_A_B','same_as_X','none'])

    # By Concept Variables
    parser.add_argument("--repetitions", type=int, default=1, required=False, help="Number of repetitions")

    # Cross Concepts Variables
    parser.add_argument("--repeat-times", default='1', required=False, help="Number of repetitions")
    parser.add_argument("--file-read", default="multiple_sets", choices=["multiple_sets", "same_set"])
    parser.add_argument("--file-with_dimensions")
    parser.add_argument("--embedding-dimensions", type=int, help="Dimensions Size", )

    return parser.parse_args()
 
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
    
    if remove_dimensions == None:
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

def show_results(all_bias, print_type, score_or_quant, language,top_similar,add_signal,top_type,rude_level):
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
                    if top_type == 'top':
                        vk = sorted(zip(item_v_list,my_list_keys_to_print))[(len(item_v_list)-top_similar):]
                    if top_type == 'equal':
                        vk = sorted(zip(item_v_list,my_list_keys_to_print))
                        vk = vk[0:(top_similar//2)+1] + vk[-((top_similar//2)+1):]

                if add_signal == 'True':
                    # add signal
                    vk = add_list_signal_in_ziplist(vk,language)
                
                if score_or_quant=='both':
                    if top_type == 'top':
                        vk = sorted(vk,key=lambda x: x[0][1])
                    if top_type == 'equal':
                        vk = sorted(vk,key=lambda x: x[0][1])
                        vk = sorted(vk[0:(top_similar//2)+1], reverse=True) + sorted(vk[-((top_similar//2)+1):], reverse=True)
                
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
        if rude_level == 1:
            df.to_csv(f'../results/violin/Enviroment:language-{language},task-{args.task},score_or_quant-{args.score_or_quant},extract_top_similar-{args.extract_top_similar},view_top_similar-{args.view_top_similar},remove_dimensions_list-{removed_dimensions},repetitions-{args.repetitions},bias_type-{args.bias_type}', index=False)
        else:
            df.to_csv(f'../results/violin/Enviroment:language-{language},task-{args.task},score_or_quant-{args.score_or_quant},extract_top_similar-{args.extract_top_similar},view_top_similar-{args.view_top_similar},remove_dimensions_list-{removed_dimensions},repetitions-{args.repetitions},bias_type-{args.bias_type},rude_level-{rude_level}', index=False)
        print(df)

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
            
            # Classification
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
    if repetitions > 1:
        print("Dividing by the number of repetitions")
        for global_concept in repetition_all_bias:
            for micro_concept in repetition_all_bias[global_concept]:
                for item in repetition_all_bias[global_concept][micro_concept]:
                    repetition_all_bias[global_concept][micro_concept][item] = (repetition_all_bias[global_concept][micro_concept][item][0]/repetitions, repetition_all_bias[global_concept][micro_concept][item][1]/repetitions)
    return repetition_all_bias

if __name__ == "__main__":
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    if args.rude_level == 1:
        with open(f'{args.dataset_path}/{args.language}_textual_phrases.txt') as f:
            text_dataset = json.load(f)
    elif args.rude_level == 0:
        with open(f'../Less-Politically-Charged-and-Translations-Sets/{args.language}_textual_phrases_less_politically_charged.txt') as f:
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
    if args.task == 'by_concept':
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
        print('') 
        all_bias = extract_bias(args.concepts, args.dataset_path, vision_processor, model, labels, text_tokenizer, device, args.language, number_concepts, weighted_list, add_signal, sorted_df_similarities,extract_top_similar, remove_dimensions_list, args.repetitions, args.bias_type)
        show_results(all_bias, args.print, args.score_or_quant, args.language,view_top_similar,add_signal,args.top_type,args.rude_level)
    
    # Calculates the comparative bias between two classes from a set of dimensions to be removed. As input a txt file with the dimensions
    if args.task == 'cross_concept':

        # transform args to list
        file_list = args.file_with_dimensions.replace('|', ' ')
        file_list = [item for item in file_list.split(',')]
        repeat_times_list = args.repeat_times.replace('|', ' ')
        repeat_times_list = [item for item in repeat_times_list.split(',')]
            
        all_features_values, batch_texts, all_images = all_features(args.concepts, args.dataset_path, vision_processor, model, labels, text_tokenizer, device, args.language)
        text_features = torch.cat((all_features_values['unpleasant_phrases'], all_features_values['pleasant_phrases']), 0)

        combination_list = {}
        for global_concept in all_features_values:
            try:
                micro_concept = list(all_features_values[global_concept].keys())
                combination_list[global_concept] = (list(itertools.combinations(micro_concept, 2)))
            except:
                pass

        start = time.time()
        for d_file in file_list:     # PROPRIO
            print(f'file_with_dimensions: {d_file}')
            with open(d_file) as f:
                if args.file_read == 'multiple_sets':    # PROPRIO
                    lines = f.readlines()
                    concepts = {}
                    for line in lines:
                        partition = line.split('[')
                        value = partition[0].split(',')
                        concepts[value[1].strip()] = partition[1].strip()[:-1].split(', ')
                elif args.file_read == 'same_set':       # PROPRIO
                    concepts = {}
                    line = f.readline()
                    partition = line.split('[')
                    my_concepts = args.concepts.replace('|', ' ')
                    # List thought all the my_concepts
                    bias_list = [item for item in my_concepts.split(',')]
                    complete_all_features_values = []
                    for bias in bias_list:
                        folder1= bias.split('/')[0]
                        folder2= bias.split('/')[1]
                        concepts[folder2]=partition[1].strip()[:-1].split(', ')
            
            for repeat in repeat_times_list:
                mean_result = {}
                start = time.time()
                all_results = []
                for _ in range(int(repeat)):
                    for global_concept in combination_list:
                        for micro_concept in combination_list[global_concept]:
                            if concepts[micro_concept[0]] == [''] or concepts[micro_concept[1]] == ['']:
                                num_dimensions = 0
                            else:
                                num_dimensions = len(concepts[micro_concept[0]]) if len(concepts[micro_concept[0]]) < len(concepts[micro_concept[1]]) else len(concepts[micro_concept[1]])   
                            
                            if args.bias_type == 'none':    # PROPRIO
                                X_feature = all_features_values[global_concept][micro_concept[0]].clone()
                                Y_feature = all_features_values[global_concept][micro_concept[1]].clone()
                                A_feature = all_features_values["unpleasant_phrases"].clone()
                                B_feature = all_features_values["pleasant_phrases"].clone()
                                A_feature_history = A_feature.clone()
                                B_feature_history = B_feature.clone()
                            elif args.bias_type == 'random':   # PROPRIO
                                X_feature = all_features_values[global_concept][micro_concept[0]].clone()
                                Y_feature = all_features_values[global_concept][micro_concept[1]].clone()
                                A_feature = all_features_values["unpleasant_phrases"].clone()
                                B_feature = all_features_values["pleasant_phrases"].clone()
                                while A_feature.size()[1] > (args.embedding_dimension-num_dimensions):  # PROPRIO
                                    remove_value = random.randint(0,A_feature.size()[1])
                                    X_feature = torch.cat([X_feature[:, :remove_value], X_feature[:, remove_value+1:]], dim=1)
                                    Y_feature = torch.cat([Y_feature[:, :remove_value], Y_feature[:, remove_value+1:]], dim=1)
                                    A_feature = torch.cat([A_feature[:, :remove_value], A_feature[:, remove_value+1:]], dim=1)
                                    B_feature = torch.cat([B_feature[:, :remove_value], B_feature[:, remove_value+1:]], dim=1)
                                A_feature_history = A_feature.clone()
                                B_feature_history = B_feature.clone()
                            else:
                                X_feature = None
                                for i, dim in enumerate(range(all_features_values[global_concept][micro_concept[0]].size()[1])):
                                    if str(i) not in concepts[micro_concept[0]][:num_dimensions]:
                                        if X_feature == None:
                                            X_feature = all_features_values[global_concept][micro_concept[0]][:,i][:,None]
                                        else:
                                            X_feature = torch.cat([X_feature, all_features_values[global_concept][micro_concept[0]][:,i][:,None]], dim=1)

                                Y_feature = None
                                for i, dim in enumerate(range(all_features_values[global_concept][micro_concept[1]].size()[1])):
                                    if str(i) not in concepts[micro_concept[1]][:num_dimensions]:
                                        if Y_feature == None:
                                            Y_feature = all_features_values[global_concept][micro_concept[1]][:,i][:,None]
                                        else:
                                            Y_feature = torch.cat([Y_feature, all_features_values[global_concept][micro_concept[1]][:,i][:,None]], dim=1)
                                # If the dimensions are conceptually independent, the dimensions removed from A and B are random, for a fair comparison.
                                if args.bias_type == 'random_A_B':
                                    A_feature = all_features_values["unpleasant_phrases"].clone()
                                    B_feature = all_features_values["pleasant_phrases"].clone()
                                    id_list = random.sample(range(args.embedding_dimension), args.embedding_dimension-num_dimensions)
                                    A_feature = A_feature[:,id_list]
                                    B_feature = B_feature[:,id_list]

                                    A_feature_history = A_feature.clone()
                                    B_feature_history = B_feature.clone()
                                # If the dimensions are all the same, they are not removed randomly, but the same dimensions of A and B are removed.
                                elif args.bias_type == 'same_as_X':
                                    A_feature = None
                                    for i, dim in enumerate(range(all_features_values["unpleasant_phrases"].size()[1])):
                                        if str(i) not in concepts[micro_concept[0]][:num_dimensions]:
                                            if A_feature == None:
                                                A_feature = all_features_values["unpleasant_phrases"][:,i][:,None]
                                            else:
                                                A_feature = torch.cat([A_feature, all_features_values["unpleasant_phrases"][:,i][:,None]], dim=1)
                                    B_feature = None
                                    for i, dim in enumerate(range(all_features_values["pleasant_phrases"].size()[1])):
                                        if str(i) not in concepts[micro_concept[1]][:num_dimensions]:
                                            if B_feature == None:
                                                B_feature = all_features_values["pleasant_phrases"][:,i][:,None]
                                            else:
                                                B_feature = torch.cat([B_feature, all_features_values["pleasant_phrases"][:,i][:,None]], dim=1)
                                    A_feature_history = A_feature.clone()
                                    B_feature_history = B_feature.clone()
                                else:
                                    print('bias type not implemented')
                                    sys.exit("You chose to quit the program.")
                            
                            test = Test(X_feature.detach().numpy(),Y_feature.detach().numpy(),A_feature.detach().numpy(),B_feature.detach().numpy())
                            pval = test.run(n_samples=250)
                            e,p = test.run()

                            if f'{global_concept}/{micro_concept[0]} x {global_concept}/{micro_concept[1]}' not in mean_result:
                                mean_result[f'{global_concept}/{micro_concept[0]} x {global_concept}/{micro_concept[1]}'] = [e]
                                all_results.append([X_feature,Y_feature,A_feature_history,B_feature_history,e])
                            else:
                                mean_result[f'{global_concept}/{micro_concept[0]} x {global_concept}/{micro_concept[1]}'].append(e)
                                all_results.append([X_feature,Y_feature,A_feature_history,B_feature_history,e])

                for concept_value in mean_result:
                    print(f'{concept_value}: {mean(mean_result[concept_value])}')
                end = time.time()
                print(f'__________________________ Time: {end - start} __ Repeated: {repeat} __ File: {d_file} __________________________')
                print('#################################################################################################################')
        end = time.time()
        print(f'__________________________ Time: {end - start} __________________________')
