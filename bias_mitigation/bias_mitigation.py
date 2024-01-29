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
from sklearn.metrics.pairwise import cosine_similarity
import scipy
import itertools
import time


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
            # print(f'Original Bias: {compute_bias(all_features[conj_combinations][combination[0]],all_features[conj_combinations][combination[1]],all_features["unpleasant_phrases"],all_features["pleasant_phrases"])}')
            
            test = Test(all_features[conj_combinations][combination[0]].detach().numpy(),all_features[conj_combinations][combination[1]].detach().numpy(),all_features["unpleasant_phrases"].detach().numpy(),all_features["pleasant_phrases"].detach().numpy())
            pval = test.run(n_samples=250)
            e,p = test.run()
            print(f'Original Bias: {e}, p: {p}')
        
            #print(compute_bias(all_features['Sexual Orientation']['Heterosexual'],all_features['Sexual Orientation']['LGBT'],all_features['unpleasant_phrases'],all_features['pleasant_phrases']))
            #newX, newY, newA, newB = bias_mitigation_algorithm(all_features['Sexual Orientation']['Heterosexual'],all_features['Sexual Orientation']['LGBT'],all_features['unpleasant_phrases'],all_features['pleasant_phrases'],54,0.5)
            best_bias_result = bias_mitigation_algorithm(all_features[conj_combinations][combination[0]],all_features[conj_combinations][combination[1]],all_features['unpleasant_phrases'],all_features['pleasant_phrases'],54,0.5)
            print(f'Best Result: {best_bias_result}')

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

    # psi = compute_bias(X, Y, A, B) 
    test = Test(X.detach().numpy(),Y.detach().numpy(),A.detach().numpy(),B.detach().numpy())
    pval = test.run(n_samples=250)
    psi,p = test.run()

    # print(f'Initial Bias: {psi}')
    # print(f'temps dimension: {X.size()},{Y.size()},{A.size()},{B.size()}')
    for d in range(X.size(1)):
        X_temp = X.clone()
        Y_temp = Y.clone()
        A_temp = A.clone()
        B_temp = B.clone()
        
        # Remover a dimensão d
        X_temp = torch.cat([X_temp[:, :d], X_temp[:, d+1:]], dim=1)
        Y_temp = torch.cat([Y_temp[:, :d], Y_temp[:, d+1:]], dim=1)
        A_temp = torch.cat([A_temp[:, :d], A_temp[:, d+1:]], dim=1)
        B_temp = torch.cat([B_temp[:, :d], B_temp[:, d+1:]], dim=1)
        # print(f'dimension {d} removed')
        # print(f'New temps dimensions: {X_temp.size()},{Y_temp.size()},{A_temp.size()},{B_temp.size()}')
        # Calcular o MI
        # mi = compute_bias(X_temp, Y_temp, A_temp, B_temp)
        test = Test(X_temp.detach().numpy(),Y_temp.detach().numpy(),A_temp.detach().numpy(),B_temp.detach().numpy())
        pval = test.run(n_samples=250)
        mi,p = test.run()

        # print('New bias')
        # print(f'mi: {mi}, theta: {theta}')
        if abs(psi) < abs(mi):
            x.add((d, mi))
        # print('---------------------------------------------------')
    # print('Out of loop')
    # Ordenar e selecionar as dimensões a serem removidas
    z = sorted(x, key=lambda item: item[1],reverse=True)[:n]
    # print(f'Valores de Z: {z}')

    # Remover as dimensões selecionadas
    best_dimension_bias = (99999,[])
    for n_bias in range(n):
        X_temp = X.clone()
        Y_temp = Y.clone()
        A_temp = A.clone()
        B_temp = B.clone()
        # Todo: Otimizar, passa apenas uma vez e já calcular o bias em "test"
        for dim, _ in z[:n_bias]:
            X_temp = torch.cat([X_temp[:, :dim], X_temp[:, dim+1:]], dim=1)
            Y_temp = torch.cat([Y_temp[:, :dim], Y_temp[:, dim+1:]], dim=1)
            A_temp = torch.cat([A_temp[:, :dim], A_temp[:, dim+1:]], dim=1)
            B_temp = torch.cat([B_temp[:, :dim], B_temp[:, dim+1:]], dim=1)
            # bias = compute_bias(X_temp,Y_temp,A_temp,B_temp)
        test = Test(X_temp.detach().numpy(),Y_temp.detach().numpy(),A_temp.detach().numpy(),B_temp.detach().numpy())
        pval = test.run(n_samples=250)
        bias,p = test.run()

        if abs(bias) < abs(best_dimension_bias[0]):
            best_dimension_bias = (bias,z[:n_bias])
    return best_dimension_bias

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

        inicio = time.time()
        all_bias = extract_bias(args.concepts, args.dataset_path, vision_processor, model, labels, text_tokenizer, device, args.language, number_concepts, weighted_list, add_signal, sorted_df_similarities,args.top_similar)
        fim = time.time()
        print(f'Tempo de execução: {fim - inicio}')