import torch
import json
import open_clip
import itertools
import time
import pandas as pd
import argparse
from tqdm import tqdm
import sys
from utils.MI import mutual_information_2d
from utils.all_features import all_features

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["calculate_bias_together","calculate_bias_separately"], help="Task to be done" )
    parser.add_argument("--ft-open-clip", default='false', type=str, required=False)
    parser.add_argument("--gpu", help="GPU", )
    parser.add_argument("--dataset-path", help="Path to validation/test dataset")
    parser.add_argument("--rude-level",  type=int, default=1, required=False, help="Words set used, original o less rude.")
    parser.add_argument("--concepts", type=str, required=False)
    parser.add_argument("--language", default="en", choices=["en", "pt-br", "xh", "hi"], required=False)
    parser.add_argument("--language-path-rude-0", help="Path to validation/test dataset to the rude level 0")
    parser.add_argument("--language-path-rude-1", help="Path to validation/test dataset to the rude level 1")
    parser.add_argument("--weighted-list", default=None)
    parser.add_argument("--adapter", default=None, required=False, help="Load the adapter weights")
    parser.add_argument("--add-signal", default=None, choices=["True", "False"])
    parser.add_argument("--sorted_df_similarities", default=None, choices=["True", "False"])
    parser.add_argument("--top-similar", required=False, help="Number of terms considered to be removed")
    parser.add_argument("--embedding-dimensions", type=int, help="Batch size", )
    parser.add_argument("--theta", default='0.05', required=False, help="Value of the in FairPIVARA")
    parser.add_argument("--n-size", default='54', required=False, help="number of values removed")
    parser.add_argument("--function-to-optimize", default='minimize', type=str, choices=["minimize","maximize"], help="Task to be done" )
    return parser.parse_args()

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

def simple_bias_from_df(df_list,weighted_list,top_similar,language,add_signal):
    parcial_bias = {}
    list_of_concepts = []
    top_one_concepts = []
    # Add the firt "number of concepts" (default = 15) in the dict
    for df in df_list:
        if df.iloc[0].text.split(' ')[0] in labels['unpleasant_phrases']:
            top_one_concepts.append(0)
        else:
            top_one_concepts.append(1)

        for nc in range(int(top_similar)):
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
        temp_list = add_list_signal(temp_list, language=language)
    return temp_list, top_one_concepts


def unique_bias_mean(X,sorted_df_similarities,weighted_list,language,add_signal,dimensions=None):
    df_list = image_to_text_retrieval(X, text_features, all_images, batch_texts, sorted_df_similarities,dimensions=dimensions)
    bias,label_list = simple_bias_from_df(df_list,weighted_list=weighted_list,top_similar=args.top_similar,language=language,add_signal=add_signal)
    mean = 0
    for text in bias:
        mean += bias[text][1]
    return mean, label_list

def single_bias_mitigation_algorithm(X, n, theta,function_to_optimize,sorted_df_similarities,weighted_list,language,add_signal):
    x = set()
    psi,_ = unique_bias_mean(X,sorted_df_similarities=sorted_df_similarities,weighted_list=weighted_list,language=language,add_signal=add_signal)  # Replace captions_vi and captions_vt with your subtitles
    for d in tqdm(range(X.size(1)), desc='Dimensions'):

        X_temp = X.clone()

        # Calculate MI
        d_bias, labels = unique_bias_mean(X_temp,dimensions=[d],sorted_df_similarities=sorted_df_similarities,weighted_list=weighted_list,language=language,add_signal=add_signal)

        mi = mutual_information_2d(X_temp[:, d],labels)

        if mi < theta:
            # If MI is less than threshold, calculate new bias
            # psi_d = compute_bias(X_temp, Y_temp, A_temp, B_temp)
            if abs(d_bias) < abs(psi):
                x.add((d, d_bias))

    # Sort and select the dimensions to be removed
    # z = sorted(x, key=lambda item: item[1])[:n]
    if function_to_optimize == 'maximize':
        z = sorted(x, key=lambda item: item[1],reverse=False)[:n]
    else:
        # if min, remove the highest values
        z = sorted(x, key=lambda item: item[1],reverse=True)[:n]

    # Remove selected dimensions
    best_dimension_bias = (99999,[])
    removed_dimensions = []
    for dim, _ in z:
        removed_dimensions.append(dim)
        psi,_ = unique_bias_mean(X, dimensions=removed_dimensions, sorted_df_similarities=sorted_df_similarities, weighted_list=args.weighted_list, language=args.language, add_signal=add_signal)
        # test if the original value is better than the new one
        if function_to_optimize != 'maximize':
            if abs(psi) < abs(best_dimension_bias[0]):
                best_dimension_bias = (psi,removed_dimensions)
        else:
            if abs(psi) > abs(best_dimension_bias[0]):
                best_dimension_bias = (psi,removed_dimensions)
    return best_dimension_bias

if __name__ == "__main__":
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("Device: ", device)
    if args.function_to_optimize == 'maximize':
        print('!!!!!!!!!!!Attention, you are maximizing the function!!!!!!!!!!!')

    if args.rude_level == 1:
        with open(f'{args.language_path_rude_1}/{args.language}_textual_phrases.txt') as f:
            text_dataset = json.load(f)
    elif args.rude_level == 0:
        with open(f'{args.language_path_rude_0}/{args.language}_textual_phrases_less_politically_charged.txt') as f:
            text_dataset = json.load(f)
    else:
        print('Invalid rude level')
        sys.exit()

    theta_list = args.theta.replace('|', ' ')
    theta_list = [item for item in theta_list.split(',')]
    n_size_list = args.n_size.replace('|', ' ')
    n_size_list = [item for item in n_size_list.split(',')]

    labels = {}
    labels['unpleasant_phrases'] = text_dataset['unpleasant_phrases']
    labels['pleasant_phrases'] = text_dataset['pleasant_phrases']
    del text_dataset['unpleasant_phrases'], text_dataset['pleasant_phrases']

    number_concepts = len(labels['unpleasant_phrases']) + len(labels['pleasant_phrases'])

    if args.ft_open_clip == 'True':
        if args.adapter == 'False':
            print('Using the CAPIVARA Model')
            model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:hiaac-nlp/CAPIVARA')
            tokenizer = open_clip.get_tokenizer('hf-hub:hiaac-nlp/CAPIVARA')
        else:
            model = OpenCLIPAdapter(inference=True, devices=device)
            model.load_adapters(pretrained_adapter=args.adapter)
    else:
        print('Using the OpenCLIP Model')
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        tokenizer = open_clip.get_tokenizer('ViT-B-32')

    if args.ft_open_clip == 'True':
        vision_processor = preprocess_val
        text_tokenizer = tokenizer
    else:
        vision_processor = preprocess
        text_tokenizer = tokenizer

    all_features_values, batch_texts, all_images = all_features(args.concepts, args.dataset_path, vision_processor, model, labels, text_tokenizer, device, args.language)
    text_features = torch.cat((all_features_values['unpleasant_phrases'], all_features_values['pleasant_phrases']), 0)

    combination_list = {}
    for global_concept in all_features_values:
        try:
            micro_concept = list(all_features_values[global_concept].keys())
            combination_list[global_concept] = (list(itertools.combinations(micro_concept, 2)))
        except:
            pass

    print(f'Running the {args.task} task')

    # Calculate which dimensions should be removed separately, by class
    if args.task == 'calculate_bias_separately':
        start = time.time()
        concepts = args.concepts.replace('|', ' ')
        for t in theta_list:
            for dimension in n_size_list:
                print(f'Theta: {int(t)}, Dimension: {dimension}')
                # List thought all the concepts
                bias_list = [item for item in concepts.split(',')]
                for bias in tqdm(bias_list, desc='Bias'):
                    folder1= bias.split('/')[0]
                    folder2= bias.split('/')[1]
                    best_dimension_bias = single_bias_mitigation_algorithm(all_features_values[folder1][folder2],float(dimension),int(t),args.function_to_optimize,sorted_df_similarities=args.sorted_df_similarities,weighted_list=args.weighted_list,language=args.language,add_signal=args.add_signal)
                    print(f'{t}, {folder2}, {unique_bias_mean(all_features_values[folder1][folder2],sorted_df_similarities=args.sorted_df_similarities,weighted_list=args.weighted_list,language=args.language,add_signal=args.add_signal)[0]}, {best_dimension_bias[0]}, {best_dimension_bias[1]}')
        end = time.time()
        print(f'__________________________ Time: {end - start} __________________________')
    # Calculate which dimensions should be removed together, for all classes
    if args.task == 'calculate_bias_together':
        concepts = args.concepts.replace('|', ' ')

        # List thought all the concepts
        bias_list = [item for item in concepts.split(',')]
        complete_all_features_values = []
        for bias in bias_list:
            folder1= bias.split('/')[0]
            folder2= bias.split('/')[1]

            complete_all_features_values.append(all_features_values[folder1][folder2])
        complete_all_features_values = torch.cat(complete_all_features_values, axis=0)

        for t in tqdm(theta_list, desc='Theta'):
            for dimension in n_size_list:
                best_dimension_bias = single_bias_mitigation_algorithm(complete_all_features_values,int(dimension),float(t),args.function_to_optimize,sorted_df_similarities=args.sorted_df_similarities,weighted_list=args.weighted_list,language=args.language,add_signal=args.add_signal)
                print(f'{t}, Total bias together, {unique_bias_mean(all_features_values[folder1][folder2],sorted_df_similarities=args.sorted_df_similarities,weighted_list=args.weighted_list,language=args.language,add_signal=args.add_signal)[0]}, {best_dimension_bias[0]}, {best_dimension_bias[1]}')
