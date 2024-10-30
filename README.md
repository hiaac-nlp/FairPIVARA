# <img src="assets/fairpivara.png" style="width:50px; margin-right:-5px"> FairPIVARA: Reducing and Assessing Biases in CLIP-Based Multimodal Models 

[![Arxiv](https://img.shields.io/badge/Arxiv-2409.19474_--_2024-red?color=crimson)](https://arxiv.org/abs/2409.19474)

In this work, we evaluate four different types of discriminatory practices within visual-language models and introduce <img src="assets/fairpivara.png" style="width:20px"> FairPIVARA, a method to reduce them by removing the most affected dimensions of feature embeddings. The application of <img src="assets/fairpivara.png" style="width:20px"> FairPIVARA has led to a significant reduction of up to 98\% in observed biases while promoting a more balanced word distribution within the model.

## Pipeline
<img src="assets/FairPIVARA_Diagram.png" >

In our pipeline, we employed the following models:

+ **Encoders Image/Text**
    - English: CLIP 
    - Portuguese: CAPIVARA

## Results

#### Relative bias

|                   |                   |                     |   |    CLIP   |        |               |            |               |   | CAPIVARA |            |               |
|-------------------|-------------------|---------------------|---|:---------:|:------:|:-------------:|:----------:|:-------------:|---|:--------:|:----------:|:-------------:|
|                   |      Target X     |       Target Y      |   | CLIP Base | MMBias | Reduction (%) | FairPIVARA | Reduction (%) |   | CAPIVARA | FairPIVARA | Reduction (%) |
|     Disability    | Mental Disability |     Non-Disabled    |   |    1.43   |  1.43  |     0.00%     |    0.01    |     99.30%    |   |   1.63   |    -0.01   |     99.39%    |
|                   | Mental Disability | Physical Disability |   |    0.92   |  0.92  |     0.00%     |    0.01    |     98.91%    |   |   1.12   |    0.02    |     98.21%    |
|                   |    Non-Disabled   | Physical Disability |   |   -1.06   |  -0.57 |     46.23%    |    0.02    |     98.11%    |   |   -1.32  |    0.00    |    100.00%    |
|    Nationality    |      American     |         Arab        |   |   -0.97   |  -0.81 |     16.49%    |    0.01    |     98.97%    |   |   -1.21  |    0.00    |    100.00%    |
|                   |      American     |       Chinese       |   |   -0.56   |  -0.49 |     12.50%    |    0.02    |     96.43%    |   |   -0.62  |    0.00    |    100.00%    |
|                   |      American     |       Mexican       |   |   -1.07   |  -0.99 |     7.48%     |    0.00    |    100.00%    |   |   -0.92  |    0.00    |    100.00%    |
|                   |        Arab       |       Chinese       |   |    0.53   |  0.53  |     0.00%     |    0.00    |    100.00%    |   |   0.76   |    0.00    |    100.00%    |
|                   |        Arab       |       Mexican       |   |   -0.13   |  -0.10 |     23.08%    |    -0.02   |     84.62%    |   |   0.43   |    -0.02   |     95.33%    |
|                   |      Chinese      |       Mexican       |   |   -0.65   |  -0.44 |     32.31%    |    0.00    |    100.00%    |   |   -0.37  |    -0.01   |     97.32%    |
|      Religion     |      Buddhist     |      Christian      |   |    0.80   |  0.80  |     0.00%     |    -0.01   |     98.75%    |   |   0.77   |    0.00    |    100.00%    |
|                   |      Buddhist     |        Hindu        |   |    0.00   |  0.00  |     0.00%     |    0.05    |     0.00%     |   |   0.08   |    0.01    |     87.68%    |
|                   |      Buddhist     |        Jewish       |   |   -1.66   |  -1.66 |     0.00%     |    0.01    |     99.40%    |   |   -1.62  |    0.00    |    100.00%    |
|                   |      Buddhist     |        Muslim       |   |   -1.60   |  -1.54 |     3.75%     |    0.01    |     99.38%    |   |   -1.51  |    0.01    |     99.34%    |
|                   |     Christian     |        Hindu        |   |   -0.73   |  -0.65 |     10.96%    |    -0.02   |     97.26%    |   |   -0.67  |    0.00    |    100.00%    |
|                   |     Christian     |        Jewish       |   |   -1.71   |  -1.69 |     1.17%     |    0.00    |    100.00%    |   |   -1.72  |    -0.01   |     99.42%    |
|                   |     Christian     |        Muslim       |   |   -1.67   |  -1.65 |     1.20%     |    0.01    |     99.40%    |   |   -1.65  |    0.01    |     99.39%    |
|                   |       Hindu       |        Jewish       |   |   -1.58   |  -1.58 |     0.00%     |    -0.01   |     99.37%    |   |   -1.60  |    0.02    |     98.75%    |
|                   |       Hindu       |        Muslim       |   |   -1.53   |  -1.52 |     0.65%     |    0.02    |     98.69%    |   |   -1.50  |    0.01    |     99.33%    |
|                   |       Jewish      |        Muslim       |   |   -0.18   |  -0.07 |     61.11%    |    0.02    |     88.89%    |   |   0.07   |    0.01    |     85.24%    |
| Sexua Orientation |    Heterosexual   |         LGBT        |   |   -1.33   |  -1.32 |     0.75%     |    0.02    |     98.50%    |   |   -1.18  |    0.02    |     98.30%    |

Relative bias between classes for OpenCLIP and CAPIVARA models, along with bias reduction by MMBias and FairPIVARA algorithms. In the pdf of the FairPIVAFA article, bias with a higher correlation to target $X$ is highlighted in orange, and bias with a higher correlation to target $Y$ is shown in yellow.

#### Classification Performance

|   Model  | Metric |    ImageNet   |                 |   CIFAR-100   |                 |    ELEVATER   |                 |
|:--------:|:------:|:-------------:|:---------------:|:-------------:|:---------------:|:-------------:|:---------------:|
|          |        | Original (\%) | FairPIVARA (\%) | Original (\%) | FairPIVARA (\%) | Original (\%) | FairPIVARA (\%) |
| OpenCLIP |  Top-1 |      61.8     |       61.3      |      77.0     |       76.2      |      61.6     |       60.8      |
|          |  Top-5 |      87.6     |       87.3      |      94.4     |       93.4      |               |                 |
| CAPIVARA |  Top-1 |      46.1     |       44.9      |      69.4     |       67.6      |      57.5     |       56.5      |
|          |  Top-5 |      70.6     |       69.5      |      90.2     |       89.4      |               |                 |

Performance comparison between OpenCLIP and CAPIVARA models, both without (Original) and with bias mitigation (FairPIVARA), on ImageNet, CIFAR-100, and the ELEVATER benchmark. OpenCLIP is evaluated in English, and CAPIVARA in Portuguese.


## Reproducibility
<!-- ### Installation
Run the following command to install required packages.

```bash
pip install -r requirements.txt
``` -->

### Code organization

```
├─ README.md
├─ requirements.txt
├─ assets
│  ├─ FairPIVARA_Diagram.png
│  └─ fairpivara.png
├─ Less-Politically-Charged-and-Translations-Sets
│  ├─ pt-br_textual_phrases.txt <--- translation of the textual data in English to Portuguese, from the MMBIAS dataset
│  ├─ en_textual_phrases_less_politically_charged.txt <--- textual bias data with less political charge, in English, proposed by this work
│  └─ pt-br_textual_phrases_less_politically_charged.txt <--- textual bias data with less political charge, in Portuguese, proposed by this work
├─FairPIVARA
│  ├─ prepare_environment.sh <--- bash file to prepare the environment to run the project and data organizer
│  ├─ FairPIVARA_select_dimensions.sh <--- bash file to run the FairPIVARA.py algorithm (Calculate what dimensions should be removed)
│  └─ show_bias.sh <--- bash file to run the show_bias.py. (Show the FairPIVARA results)
│  ├─ visualizers <--- .ipynb files for viewing results.				
│  │  ├─ classificationTables.ipynb <--- creating and viewing classification tables
│  │  ├─ en-visualizer.ipynb <--- graphs for results analysis in English
│  │  ├─ pt-br-visualizer.ipynb <--- graphs for results analysis in Portuguese
│  │  ├─ ZeroShotClassification.ipynb <--- zero-shot grading with visualization.
│  │  └─ ZeroShotRetrieval.ipynb <--- zero-shot retrieval with visualization.
│  ├─ src
│  │  └─ show_bias.py <--- calculates the results of individual bias, to create comparative tables of bias by concepts (task by_concept). Calculates the results for Relative Bias (task cross_concept).
│  │  └─ FairPIVARA.py <--- performs the check of which dimensions should be removed (tasks calculate_bias_separately or calculate_bias_together)
│  │  ├─ utils
│  │  │  ├─ __init__.py <--- package level
│  │  │  ├─ all_features.py <--- method for extracting features from data
│  │  │  ├─ highest_frequency_dimensions.py <--- auxiliary methods for checking the highest frequency dimensions
│  │  │  ├─ MI.py <--- Implementation of Mutual Information
│  │  │  ├─ mmbias_dataset.py <--- dataset handle
│  │  │  └─ WEAT_test.py <--- class for test the WEAT value
│  │  ├─ result_example
└─ └─ └─ └─ results_theta_0-05_same_values.txt <--- example of output file of the FairPIVARA algorithm, for use of show_bias



```

### Three main functions are implemented for the use of FairPIVARA
+ FairPIVARA Algorithm: For selecting which dimensions can be optimally removed from the embedding.
+ Individual Bias (by_concept): Calculation of Bias individually and comparatively between concepts (Tables 1, 2 and 3).
+ Relative Bias (cross_concept): Calculation of Bias in a relative way, between two classes, Table 4.

#### FairPIVARA Algorithm

The FairPIVARA.py file must be used, with the bash FairPIVARA_select_dimensions.sh script.
The following arguments are supported:

```python
TASK='calculate_bias_together' # 'calculate_bias_together' or 'calculate_bias_separately'

FT_OPEN_CLIP='False' #Used in pt fine-tuning
GPU=1
DATASET_PATH='../MMBias-main/data'
RUDE_LEVEL=1 # Political (1) or non-political bias dataset
CONCEPTS='Disability/Mental|Disability,Disability/Non-Disabled,Disability/Physical|Disability,Nationality/American,Nationality/Arab,Nationality/Chinese,Nationality/Mexican,Religion/Buddhist,Religion/Christian,Religion/Hindu,Religion/Jewish,Religion/Muslim,Sexual|Orientation/Heterosexual,Sexual|Orientation/LGBT' # Concepts used in classification  (| for space and , for and)
LANGUAGE='en' # "en", "pt-br"
LANGUAGE_PATH_RUDE_0='../Less-Politically-Charged-and-Translations-Sets'
LANGUAGE_PATH_RUDE_1='../MMBias-main/data'
WEIGHTED_LIST='False'
ADAPTER='False'
ADD_SIGNAL='True'
SORTED_DF_SIMILARITIES='True'
TOP_SIMILAR=15 # '', '15' Number of terms considered
EMBEDDING_DIMENSIONS=512 # Number of model embedding dimensions
THETA='0.05' # Theta value : '0.01, 0.02, 0.03, 0.04, 0.05'
N_SIZE='54' # Number of dimensions to be removed : '27, 54, 81, 108, 135, 162, 189, 216, 243, 270, 297, 324, 351, 378, 405, 432, 459, 486, 512'
FUNCTION_TO_OPTIMIZE='minimize' # minimize, maximize
```

#### Individual Bias

Using the by_concept task of the show_bias.sh script, calculates biases individually for each concept and its bias relationship with its labels.
The following arguments are supported:

```python
TASK='by_concept' # 'by_concept' or 'cross_concept'

FT_OPEN_CLIP='False' #Used in pt fine-tuning
GPU=1
DATASET_PATH='../MMBias-main/data'
RUDE_LEVEL=1 # Political (1) or non-political bias dataset
CONCEPTS='Disability/Mental|Disability,Disability/Non-Disabled,Disability/Physical|Disability,Nationality/American,Nationality/Arab,Nationality/Chinese,Nationality/Mexican,Religion/Buddhist,Religion/Christian,Religion/Hindu,Religion/Jewish,Religion/Muslim,Sexual|Orientation/Heterosexual,Sexual|Orientation/LGBT' # Concepts used in classification  (| for space and , for and)
LANGUAGE='en' # "en", "pt-br"
PRINT='exel' #'json' , 'exel', 'pandas'  #pandas used to violin plots
SCORE_OR_QUANT=both #'both_operation (with mean), both'
WEIGHTED_LIST='False'
EXTRACT_TOP_SIMILAR='15'  # '', '15' Number of terms considered
VIEW_TOP_SIMILAR='15'  #Number of terms considered in view. For exel, we used '15'. For the violin, it is necessary to have value ''. 
TOP_TYPE='top' # 'top', 'equal' # The firsts ones or per type # equal don't work with pandas print
REMOVE_DIMENSIONS_LIST='' # List of dimensions to be removed : '' , 'results/theta-001to005/results_theta_same_values.txt'
# By Concepts Variables
BIAS_TYPE='same_as_selected' # Type of remotions for text. 'same_as_selected','random_text','random' Used with remove-dimensions-list, if remove-dimensions-list is empty, this parameter is ignored
REPETITIONS=1000 # number of times the algorithm is repeated 
```

#### Relative Bias

Using the cross_concept task of the show_bias.sh script, it is possible to calculate the results for Relative Bias.
The following additional arguments will be required:

```python
# Cross Concepts Variables
REPEAT_TIMES='1000' # number of times the algorithm is repeated. To use more than one use a list of values like '1,10,100,1000'  (| for space and , for and)
FILE_READ='multiple_sets' # File reading type 'multiple_sets, same_set'
BIAS_TYPE='same_as_X' # Type of remotions for text. random, random_A_B, same_as_X, none
FILE_WITH_DIMENSIONS='src/result_example/results_theta_0-05_same_values.txt'  # File with the dimensions to be removed #'results/theta-001to005/results_theta_0-05.txt', 'results/theta-001to005/results_theta_same_values.txt', 'results/theta-001to005/together/005-results_theta_calculation_together.txt'
EMBEDDING_DIMENSIONS=512
```

In addition to changing task to:

```python
task = 'cross_concept' 
```

## Data

To check for potentially biased classes, we used the [MMBIAS](https://github.com/sepehrjng92/MMBias) dataset, which contains images and texts of various concepts. In addition, we added a translation for this set into Portuguese, contained in Less-Politically-Charged-and-Translations-Sets, in addition to proposing new terms, with less political charge, which can be found in the same directory.
To use this data, simply run the /FairPIVARA/prepare_environment.sh script, which can download and prepare the data for use with this project.

## Acknowledgements

This project was supported by the Ministry of Science, Technology, and Innovation of Brazil, with resources granted by the Federal Law 8.248 of October 23, 1991, under the PPI-Softex. The project was coordinated by Softex and published as Intelligent agents for mobile platforms based on Cognitive Architecture technology 01245.003479/2024-10. D.A.B.M. is partially funded by FAPESP 2023/05939-5. A.I.F. and N.S. are partially funded by Centro de Excelência em Inteligência Artificial, da Universidade Federal de Goiás. G.O.S is partially funded by FAPESP 2024/07969-1. H.P. is partially funded by CNPq 304836/2022-2. S.A. is partially funded by CNPq 316489/2023-9, FAPESP 2013/08293-7, 2020/09838-0, 2023/12086-9, and Google Award for Inclusion Research 2022.

## Citation
```bibtex
@inproceedings{moreira2024fairpivarareducingassessingbiases,
      title={FairPIVARA: Reducing and Assessing Biases in CLIP-Based Multimodal Models}, 
      author={Diego A. B. Moreira and Alef Iury Ferreira and Jhessica Silva and Gabriel Oliveira dos Santos and Luiz Pereira and João Medrado Gondim and Gustavo Bonil and Helena Maia and Nádia da Silva and Simone Tiemi Hashiguti and Jefersson A. dos Santos and Helio Pedrini and Sandra Avila},
      booktitle={BMVC},
      year={2024},
}
```
