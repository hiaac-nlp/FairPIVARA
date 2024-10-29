import torch
from utils.mmbias_dataset import MMBiasDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

def all_features(concepts, dataset_path, vision_processor, model, labels, tokenizer, device, language):
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