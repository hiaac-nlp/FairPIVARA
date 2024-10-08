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
import random
from statistics import mean 
import sys

# Code extracted from https://gist.github.com/GaelVaroquaux/mutual_info.py
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

'''
Implements the WEAT tests
Adapted from https://github.com/W4ngatang/sent-bias/blob/master/sentbias/weat.py
'''
class Test:
    def __init__(self, X, Y, A, B, names=None):
        """
        A WEAT Test.

        :param X: A set of target embeddings
        :param Y: A set of target embeddings
        :param A: A set of attribute embeddings
        :param B: A set of attribute embeddings
        :param names: Optional set of names for X, Y, A, and B, in order
        :return: the effect size and p-value
        """
        self.X = X
        self.Y = Y
        self.A = A
        self.B = B
        self.names = names if names is not None else ["X", "Y", "A", "B"]
        self.reset_calc()

    def reset_calc(self):
        self.similarity_matrix = self.similarities()
        self.s_AB = None
        self.calc_s_AB()

    def run(self, randomized=False, **kwargs):
        """
        Run the test.
        """
        if randomized:
            X_orig = self.X
            Y_orig = self.Y
            A_orig = self.A
            B_orig = self.B
            D = np.concatenate((self.X, self.Y, self.A, self.B))
            np.random.shuffle(D)
            self.X = D[:X_orig.shape[0],:]
            self.Y = D[X_orig.shape[0]:2*X_orig.shape[0],:]
            self.A = D[2*X_orig.shape[0]:2*X_orig.shape[0]+A_orig.shape[0], :]
            self.B = D[2*X_orig.shape[0]+A_orig.shape[0]:, :]
            self.reset_calc()

        p = self.p(**kwargs)
        e = self.effect_size()

        if randomized:
            self.X = X_orig
            self.Y = Y_orig
            self.A = A_orig
            self.B = B_orig
            self.reset_calc()
        return e, p

    def similarities(self):
        """
        :return: an array of size (len(XY), len(AB)) containing cosine similarities
        between items in XY and items in AB.
        """
        XY = np.concatenate((self.X, self.Y))
        AB = np.concatenate((self.A, self.B))
        return cosine_similarity(XY, AB)

    def calc_s_AB(self):
        self.s_AB = self.s_wAB(np.arange(self.similarity_matrix.shape[0]))

    def s_wAB(self, w):
        """
        Return vector of s(w, A, B) across w, where
            s(w, A, B) = mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b).

        :param w: Mask on the XY axis of similarity matrix
        """
        return self.similarity_matrix[w, :self.A.shape[0]].mean(axis=1) - self.similarity_matrix[w, self.A.shape[0]:].mean(axis=1)

    def s_XAB(self, mask):
        r"""
        Given indices of target concept X and precomputed s_wAB values,
        return slightly more computationally efficient version of WEAT
        statistic for p-value computation.
        Caliskan defines the WEAT statistic s(X, Y, A, B) as
            sum_{x in X} s(x, A, B) - sum_{y in Y} s(y, A, B)
        where s(w, A, B) is defined as
            mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b).
        The p-value is computed using a permutation test on (X, Y) over all
        partitions (X', Y') of X union Y with |X'| = |Y'|.
        However, for all partitions (X', Y') of X union Y,
            s(X', Y', A, B)
          = sum_{x in X'} s(x, A, B) + sum_{y in Y'} s(y, A, B)
          = C,
        a constant.  Thus
            sum_{x in X'} s(x, A, B) + sum_{y in Y'} s(y, A, B)
          = sum_{x in X'} s(x, A, B) + (C - sum_{x in X'} s(x, A, B))
          = C + 2 sum_{x in X'} s(x, A, B).
        By monotonicity,
            s(X', Y', A, B) > s(X, Y, A, B)
        if and only if
            [s(X', Y', A, B) - C] / 2 > [s(X, Y, A, B) - C] / 2,
        that is,
            sum_{x in X'} s(x, A, B) > sum_{x in X} s(x, A, B).
        Thus we only need use the first component of s(X, Y, A, B) as our
        test statistic.

        :param mask: some random X partition of XY - in the form of a mask on XY
        """
        return self.s_AB[mask].sum()

    def s_XYAB(self, X, Y):
        r"""
        Given indices of target concept X and precomputed s_wAB values,
        the WEAT test statistic for p-value computation.

        :param X: Mask for XY indicating the values in partition X
        :param Y: Mask for XY indicating the values in partition Y
        """
        return self.s_XAB(X) - self.s_XAB(Y)

    def p(self, n_samples=10000, parametric=False):
        """
        Compute the p-val for the permutation test, which is defined as
        the probability that a random even partition X_i, Y_i of X u Y
        satisfies P[s(X_i, Y_i, A, B) > s(X, Y, A, B)]
        """
        assert self.X.shape[0] == self.Y.shape[0]
        size = self.X.shape[0]

        XY = np.concatenate((self.X, self.Y))

        if parametric:
            s = self.s_XYAB(np.arange(self.X.shape[0]), np.arange(self.X.shape[0], self.X.shape[0]+self.Y.shape[0]))

            samples = []
            for _ in range(n_samples):
                a = np.arange(XY.shape[0])
                np.random.shuffle(a)
                Xi = a[:size]
                Yi = a[size:]
                assert len(Xi) == len(Yi)
                si = self.s_XYAB(Xi, Yi)
                samples.append(si)

            # Compute sample standard deviation and compute p-value by
            # assuming normality of null distribution
            sample_mean = np.mean(samples)
            sample_std = np.std(samples, ddof=1)
            p_val = scipy.stats.norm.sf(s, loc=sample_mean, scale=sample_std)
            return p_val

        else:
            s = self.s_XAB(np.arange(self.X.shape[0]))
            total_true = 0
            total_equal = 0
            total = 0

            num_partitions = int(scipy.special.binom(2 * self.X.shape[0], self.X.shape[0]))
            if num_partitions > n_samples:
                # We only have as much precision as the number of samples drawn;
                # bias the p-value (hallucinate a positive observation) to
                # reflect that.
                total_true += 1
                total += 1
                for i in range(n_samples - 1):
                    a = np.arange(XY.shape[0])
                    np.random.shuffle(a)
                    Xi = a[:size]
                    assert 2 * len(Xi) == len(XY)
                    si = self.s_XAB(Xi)
                    if si > s:
                        total_true += 1
                    elif si == s:  # use conservative test
                        total_true += 1
                        total_equal += 1
                    total += 1
            else:
                # iterate through all possible X-length combinations of the indices of XY
                for Xi in it.combinations(np.arange(XY.shape[0]), self.X.shape[0]):
                    assert 2 * len(Xi) == len(XY)
                    si = self.s_XAB(np.array(Xi))
                    if si > s:
                        total_true += 1
                    elif si == s:  # use conservative test
                        total_true += 1
                        total_equal += 1
                    total += 1

            return total_true / total

    def effect_size(self):
        """
        Compute the effect size, which is defined as
            [mean_{x in X} s(x, A, B) - mean_{y in Y} s(y, A, B)] /
                [ stddev_{w in X u Y} s(w, A, B) ]
        args:
            - X, Y, A, B : sets of target (X, Y) and attribute (A, B) indices
        """
        numerator = np.mean(self.s_wAB(np.arange(self.X.shape[0]))) - np.mean(self.s_wAB(np.arange(self.X.shape[0], self.similarity_matrix.shape[0])))
        denominator = np.std(self.s_AB, ddof=1)
        return numerator / denominator

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
            #TODO: Armazenando somente bias atual em all_images
            all_images.append(image_input)

        text_features_un = F.normalize(model.encode_text(batch_texts_tok_un), dim=-1).cpu()
        text_features_ple = F.normalize(model.encode_text(batch_texts_tok_ple), dim=-1).cpu()

    all_features['unpleasant_phrases']=text_features_un
    all_features['pleasant_phrases']=text_features_ple

    # batch_text and all_images used only in classification pipeline
    return all_features, batch_texts, all_images

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

def single_bias_mitigation_algorithm(X, n, theta,function_to_optimize):
    x = set()
    psi,_ = unique_bias_mean(X)  # Replace captions_vi and captions_vt with your subtitles
    for d in range(X.size(1)):
        
        X_temp = X.clone()

        # Calculate MI
        d_bias, labels = unique_bias_mean(X_temp,[d])

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
        psi,_ = unique_bias_mean(X,removed_dimensions)
        # test if the original value is better than the new one
        if function_to_optimize != 'maximize':
            if abs(psi) < abs(best_dimension_bias[0]):
                best_dimension_bias = (psi,removed_dimensions)
        else:
            if abs(psi) > abs(best_dimension_bias[0]):
                best_dimension_bias = (psi,removed_dimensions)
    return best_dimension_bias

GPU = 2 
MAIN_PATH = '/hadatasets/MMBias'
DATASET_PATH = '/hadatasets/MMBias/data'
rude_level = 1 # Political (1) or non-political bias dataset
LANGUAGE_PATH = 'data'
LANGUAGE = 'en' # 'en', 'pt-br'
ft_open_clip = 'False'
adapter = 'False'
CONCEPTS='Disability/Mental|Disability,Disability/Non-Disabled,Disability/Physical|Disability,Nationality/American,Nationality/Arab,Nationality/Chinese,Nationality/Mexican,Religion/Buddhist,Religion/Christian,Religion/Hindu,Religion/Jewish,Religion/Muslim,Sexual|Orientation/Heterosexual,Sexual|Orientation/LGBT' # Concepts used in classification  (| for space and , for and)
weighted_list='False'
add_signal = 'True'
sorted_df_similarities = 'True'
top_similar = 15 # '', '15' Number of terms considered
embedding_dimension=512 # Number of model embedding dimensions
module = 'bias_calculation' #'calculate_bias_separately','bias_calculation' 
theta = [0.05] # Theta value : [0.01, 0.02, 0.03, 0.04, 0.05]
N_size = [54] # Number of dimensions to be removed : [27, 54, 81, 108, 135, 162, 189, 216, 243, 270, 297, 324, 351, 378, 405, 432, 459, 486, 512]
function_to_optimize = 'minimize' # minimize, maximize

# Parameters used for the bias calculation module
repeat_times = [1000] # number of times the algorithm is repeated : [1, 100, 1000]
file_read = 'multiple_sets' # File reading type 'multiple_sets, same_set'
bias_type = 'same_as_X' # Type of remotions for text. random, random_A_B, same_as_X, none
file_with_dimensions = ['results/theta-001to011/54_dims/same_values/results_theta_0-05_same_values.txt']  # File with the dimensions to be removed #'results/theta-001to005/results_theta_0-05.txt', 'results/theta-001to005/results_theta_same_values.txt', 'results/theta-001to005/together/005-results_theta_calculation_together.txt'

# device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
if function_to_optimize == 'maximize':
    print('!!!!!!!!!!!Attention, you are maximizing the function!!!!!!!!!!!')

if rude_level == 1:
    with open(f'{MAIN_PATH}/{LANGUAGE_PATH}/{LANGUAGE}_textual_phrases.txt') as f:
        text_dataset = json.load(f)
elif rude_level == 0:
    with open(f'{MAIN_PATH}/{LANGUAGE_PATH}/{LANGUAGE}_textual_phrases_less_politically_charged.txt') as f:
        text_dataset = json.load(f)
else:
    print('Invalid rude level')
    sys.exit()

labels = {}
labels['unpleasant_phrases'] = text_dataset['unpleasant_phrases']
labels['pleasant_phrases'] = text_dataset['pleasant_phrases']
del text_dataset['unpleasant_phrases'], text_dataset['pleasant_phrases']

number_concepts = len(labels['unpleasant_phrases']) + len(labels['pleasant_phrases'])

if ft_open_clip == 'True':
    if adapter == 'False':
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

if ft_open_clip == 'True':
    vision_processor = preprocess_val
    text_tokenizer = tokenizer
else:
    vision_processor = preprocess
    text_tokenizer = tokenizer

EPS = np.finfo(float).eps

all_features_values, batch_texts, all_images = all_features(CONCEPTS, DATASET_PATH, vision_processor, model, labels, text_tokenizer, device, LANGUAGE, number_concepts, weighted_list, add_signal, sorted_df_similarities,top_similar)
text_features = torch.cat((all_features_values['unpleasant_phrases'], all_features_values['pleasant_phrases']), 0)

combination_list = {}
for global_concept in all_features_values:
    try:
        micro_concept = list(all_features_values[global_concept].keys())
        combination_list[global_concept] = (list(itertools.combinations(micro_concept, 2)))
    except:
        pass

print(f'Running the {module} module')

# Calculate which dimensions should be removed separately, by class
if module == 'calculate_bias_separately':
    start = time.time()
    concepts = CONCEPTS.replace('|', ' ')
    for t in theta: 
        for dimension in N_size:
            print(f'Theta: {t}, Dimension: {dimension}')
            # List thought all the concepts
            bias_list = [item for item in concepts.split(',')]
            for bias in bias_list:
                folder1= bias.split('/')[0]
                folder2= bias.split('/')[1]
                best_dimension_bias = single_bias_mitigation_algorithm(all_features_values[folder1][folder2],dimension,t,function_to_optimize)
                print(f'{t}, {folder2}, {unique_bias_mean(all_features_values[folder1][folder2])[0]}, {best_dimension_bias[0]}, {best_dimension_bias[1]}')
    end = time.time()
    print(f'__________________________ Time: {end - start} __________________________')
# Calculate which dimensions should be removed together, for all classes
if module == 'calculate_bias_together':
    concepts = CONCEPTS.replace('|', ' ')

    # List thought all the concepts
    bias_list = [item for item in concepts.split(',')]
    complete_all_features_values = []
    for bias in bias_list:
        folder1= bias.split('/')[0]
        folder2= bias.split('/')[1]

        complete_all_features_values.append(all_features_values[folder1][folder2])
    complete_all_features_values = torch.cat(complete_all_features_values, axis=0)

    for t in theta: 
        best_dimension_bias = single_bias_mitigation_algorithm(complete_all_features_values,N_size,t,function_to_optimize)
        print(f'{t}, Total bias together, {unique_bias_mean(all_features_values[folder1][folder2])[0]}, {best_dimension_bias[0]}, {best_dimension_bias[1]}')

# Calculates the comparative bias between two classes from a set of dimensions to be removed. As input a txt file with the dimensions
if module == 'bias_calculation':
    start = time.time()
    for d_file in file_with_dimensions:
        print(f'file_with_dimensions: {d_file}')
        with open(d_file) as f:
            if file_read == 'multiple_sets':
                lines = f.readlines()
                concepts = {}
                for line in lines:
                    partition = line.split('[')
                    value = partition[0].split(',')
                    concepts[value[1].strip()] = partition[1].strip()[:-1].split(', ')
            elif file_read == 'same_set':
                concepts = {}
                line = f.readline()
                partition = line.split('[')
                my_concepts = CONCEPTS.replace('|', ' ')
                # List thought all the my_concepts
                bias_list = [item for item in my_concepts.split(',')]
                complete_all_features_values = []
                for bias in bias_list:
                    folder1= bias.split('/')[0]
                    folder2= bias.split('/')[1]
                    concepts[folder2]=partition[1].strip()[:-1].split(', ')
                repeat_times = [1]
        
        for repeat in repeat_times:
            mean_result = {}
            start = time.time()
            all_results = []
            for _ in range(repeat):
                for global_concept in combination_list:
                    for micro_concept in combination_list[global_concept]:
                        if concepts[micro_concept[0]] == [''] or concepts[micro_concept[1]] == ['']:
                            num_dimensions = 0
                        else:
                            num_dimensions = len(concepts[micro_concept[0]]) if len(concepts[micro_concept[0]]) < len(concepts[micro_concept[1]]) else len(concepts[micro_concept[1]])   
                        
                        if bias_type == 'none':
                            X_feature = all_features_values[global_concept][micro_concept[0]].clone()
                            Y_feature = all_features_values[global_concept][micro_concept[1]].clone()
                            A_feature = all_features_values["unpleasant_phrases"].clone()
                            B_feature = all_features_values["pleasant_phrases"].clone()
                            A_feature_history = A_feature.clone()
                            B_feature_history = B_feature.clone()
                        elif bias_type == 'random':
                            X_feature = all_features_values[global_concept][micro_concept[0]].clone()
                            Y_feature = all_features_values[global_concept][micro_concept[1]].clone()
                            A_feature = all_features_values["unpleasant_phrases"].clone()
                            B_feature = all_features_values["pleasant_phrases"].clone()
                            while A_feature.size()[1] > (embedding_dimension-num_dimensions):
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
                            if bias_type == 'random_A_B':
                                A_feature = all_features_values["unpleasant_phrases"].clone()
                                B_feature = all_features_values["pleasant_phrases"].clone()
                                id_list = random.sample(range(embedding_dimension), embedding_dimension-num_dimensions)
                                A_feature = A_feature[:,id_list]
                                B_feature = B_feature[:,id_list]

                                A_feature_history = A_feature.clone()
                                B_feature_history = B_feature.clone()
                            # If the dimensions are all the same, they are not removed randomly, but the same dimensions of A and B are removed.
                            elif bias_type == 'same_as_X':
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

            # df = pd.DataFrame(all_results, columns=['X_feature', 'Y_feature','A_feature','B_feature','e'])
            # df.to_csv(f'/home/diego.moreira/FairPIVARA/results/dimensions_removed/{LANGUAGE}_results_in_csv_{bias_type}_{repeat}.csv')
            for concept_value in mean_result:
                print(f'{concept_value}: {mean(mean_result[concept_value])}')
            end = time.time()
            print(f'__________________________ Time: {end - start} __ Repeated: {repeat} __ File: {d_file} __________________________')
            print('#################################################################################################################')
    end = time.time()
    print(f'__________________________ Time: {end - start} __________________________')
