#!/bin/bash
###############################################################
# Program Name: run.sh                      Version:
#
# Description:
#  - Bash file to run the FairPIVARA tests
#
# Usage:
#
# Author: Diego Moreira          Last Update Date: 12/08/2023   
#
#
# Revision History
#
# Version                 Date                  Who
#-------------------------------------------------------------
#
###############################################################
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⣬⠷⣶⡖⠲⡄⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⣠⠶⠋⠁⠀⠸⣿⡀⠀⡁⠈⠙⠢⠤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀<----⠀FAIR⠀<(˶ᵔᵕᵔ˶)>⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⢠⠞⠁⠀⠀⠀⠀⠀⠉⠣⠬⢧⠀⠀⠀⠀⠈⠻⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⢀⡴⠃⠀⠀⢠⣴⣿⡿⠀⠀⠀⠐⠋⠀⠀⠀⠀⠀⠀⠘⠿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⢀⡴⠋⠀⠀⠀⠀⠈⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠒⠒⠓⠛⠓⠶⠶⢄⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⢠⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠦⣀⠀⠀⠀⠀⠀⠀⠀⠀
# ⡞⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⢷⡄⠀⠀⠀⠀⠀⠀
# ⢻⣇⣹⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀
# ⠀⠻⣟⠋⠀⠀⠀⠀⠀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣄⠀⠀⠀
# ⠀⠀⠀⠉⠓⠒⠊⠉⠉⢸⡙⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⠀⠀⠘⣆⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣱⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣿⠀⠀⠀⠀⠀⢻⡄⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠟⣧⡀⠀⠀⢀⡄⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡿⠇⠀⠀⠀⠀⠀⠀⢣⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⡧⢿⡀⠚⠿⢻⡆⠀⠀⠀⠀⠀⢠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⡇⠀⠀⠀⠀⠀⠀⠀⠘⡆
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⠀⠀⠈⢹⡀⠀⠀⠀⠀⣾⡆⠀⠀⠀⠀⠀⠀⠀⠀⠾⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠨⢷⣾⠀⠸⡷⠀⠀⠀⠘⡿⠂⠀⠀⠀⢀⡴⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡇
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡄⠳⢼⣧⡀⠀⠀⢶⡼⠦⠀⠀⠀⡞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠃
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⡎⣽⠿⣦⣽⣷⠿⠒⠀⠀⠀⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⠁⣴⠃⡿⠀⠀⢠⠆⠢⡀⠀⠀⠀⠈⢧⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠇⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⣠⠏⠀⣸⢰⡇⠀⢠⠏⠀⠀⠘⢦⣀⣀⠀⢀⠙⢧⡀⠀⠀⠀⠀⠀⠀⠀⠀⡰⠁⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠾⠿⢯⣤⣆⣤⣯⠼⠀⠀⢸⠀⠀⠀⠀⠀⣉⠭⠿⠛⠛⠚⠟⡇⠀⠀⣀⠀⢀⡤⠊⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠀⢸⣷⣶⣤⣦⡼⠀⠀⠀⣴⣯⠇⡀⣀⣀⠤⠤⠖⠁⠐⠚⠛⠉⠁⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣛⠁⢋⡀⠀⠀⠀⠀⣛⣛⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
###############################################################

# # #Classification
# # FT_OPEN_CLIP='False'
# # GPU=1 # 0->4, 1->6, 2->7, 3->0, 4->1, 6 -> 3, 7 -> 5
# # DATASET_PATH="/hadatasets/MMBias/data"
# # RUDE_LEVEL=0
# # # | for space and , for and
# # CONCEPTS='Disability/Mental|Disability,Disability/Non-Disabled,Disability/Physical|Disability,Nationality/American,Nationality/Arab,Nationality/Chinese,Nationality/Mexican,Religion/Buddhist,Religion/Christian,Religion/Hindu,Religion/Jewish,Religion/Muslim,Sexual|Orientation/Heterosexual,Sexual|Orientation/LGBT'
# # LANGUAGE='en' # "en", "pt-br"
# # TASK='classification'
# # PRINT='exel' #'json' , 'exel', 'pandas'  #pandas used to violin plots
# # SCORE_OR_QUANT=both #'both_operation, both'
# # WEIGHTED_LIST='False'
# # EXTRACT_TOP_SIMILAR='15'  # '', '15'
# # VIEW_TOP_SIMILAR='15'  # For exel '15'. For the violin, it is necessary to have value ''
# # TOP_TYPE='top' # 'top', 'equal' # equal don't work with pandas print
# # REMOVE_DIMENSIONS_LIST='results/new_words/theta-001to005/same_values/en_results_theta_0-05_same_values.txt' # '' , 'results/theta-001to005/results_theta_same_values.txt'
# # REPETITIONS=1000
# # BIAS_TYPE='random_text' #'same_as_selected','random_text','random' Used with remove-dimensions-list, if remove-dimensions-list is empty, this parameter is ignored

# # # Bias Assessment - Comparison

# # echo "FT_OPEN_CLIP: ${FT_OPEN_CLIP}, GPU: ${GPU}, DATASET_PATH: ${DATASET_PATH}, RUDE_LEVEL: ${RUDE_LEVEL}, CONCEPTS: ${CONCEPTS}, LANGUAGE: ${LANGUAGE}, TASK: ${TASK}, PRINT: ${PRINT}, SCORE_OR_QUANT: ${SCORE_OR_QUANT}, WEIGHTED_LIST: ${WEIGHTED_LIST}, EXTRACT_TOP_SIMILAR: ${EXTRACT_TOP_SIMILAR}, VIEW_TOP_SIMILAR: ${VIEW_TOP_SIMILAR}, TOP_TYPE: ${TOP_TYPE}, REMOVE_DIMENSIONS_LIST: ${REMOVE_DIMENSIONS_LIST}, REPETITIONS: ${REPETITIONS}, BIAS_TYPE: ${BIAS_TYPE}"
# # PYTHON_BIN="/home/diego.moreira/miniconda3/envs/haenv/bin/python3"

# # ${PYTHON_BIN} /home/${USER}/FairPIVARA/main.py \
# #     --ft-open-clip=${FT_OPEN_CLIP} \
# #     --dataset-path=${DATASET_PATH} \
# #     --rude-level=${RUDE_LEVEL} \
# #     --concepts=${CONCEPTS} \
# #     --task=${TASK} \
# #     --language=${LANGUAGE} \
# #     --weighted-list=${WEIGHTED_LIST} \
# #     --gpu=${GPU} \
# #     --print=${PRINT} \
# #     --score-or-quant=${SCORE_OR_QUANT} \
# #     --extract-top-similar=${EXTRACT_TOP_SIMILAR} \
# #     --view-top-similar=${VIEW_TOP_SIMILAR} \
# #     --top-type=${TOP_TYPE} \
# #     --remove-dimensions-list=${REMOVE_DIMENSIONS_LIST} \
# #     --repetitions=${REPETITIONS} \
# #     --bias-type=${BIAS_TYPE} \


FT_OPEN_CLIP='False'
GPU=2 # 0->4, 1->6, 2->7, 3->0, 4->1, 6 -> 3, 7 -> 5
DATASET_PATH="/hadatasets/MMBias/data"
RUDE_LEVEL=1
# | for space and , for and
CONCEPTS='Disability/Mental|Disability,Disability/Non-Disabled,Disability/Physical|Disability,Nationality/American,Nationality/Arab,Nationality/Chinese,Nationality/Mexican,Religion/Buddhist,Religion/Christian,Religion/Hindu,Religion/Jewish,Religion/Muslim,Sexual|Orientation/Heterosexual,Sexual|Orientation/LGBT'
LANGUAGE='en' # "en", "pt-br"
TASK='classification'
PRINT='exel' #'json' , 'exel', 'pandas'  #pandas used to violin plots
SCORE_OR_QUANT=both #'both_operation, both'
WEIGHTED_LIST='False'
EXTRACT_TOP_SIMILAR='15'  # '', '15'
VIEW_TOP_SIMILAR='15'  # For exel '15'. For the violin, it is necessary to have value ''
TOP_TYPE='top' # 'top', 'equal' # equal don't work with pandas print
REMOVE_DIMENSIONS_LIST='results/theta-001to011/135_dims/results_theta_0-05.txt' # '' , 'results/theta-001to005/results_theta_same_values.txt'
REPETITIONS=1
BIAS_TYPE='same_as_selected' #'same_as_selected','random_text','random' Used with remove-dimensions-list, if remove-dimensions-list is empty, this parameter is ignored

echo "FT_OPEN_CLIP: ${FT_OPEN_CLIP}, GPU: ${GPU}, DATASET_PATH: ${DATASET_PATH}, RUDE_LEVEL: ${RUDE_LEVEL}, CONCEPTS: ${CONCEPTS}, LANGUAGE: ${LANGUAGE}, TASK: ${TASK}, PRINT: ${PRINT}, SCORE_OR_QUANT: ${SCORE_OR_QUANT}, WEIGHTED_LIST: ${WEIGHTED_LIST}, EXTRACT_TOP_SIMILAR: ${EXTRACT_TOP_SIMILAR}, VIEW_TOP_SIMILAR: ${VIEW_TOP_SIMILAR}, TOP_TYPE: ${TOP_TYPE}, REMOVE_DIMENSIONS_LIST: ${REMOVE_DIMENSIONS_LIST}, REPETITIONS: ${REPETITIONS}, BIAS_TYPE: ${BIAS_TYPE}"
PYTHON_BIN="/home/diego.moreira/miniconda3/envs/haenv/bin/python3"

${PYTHON_BIN} /home/${USER}/FairPIVARA/main.py \
    --ft-open-clip=${FT_OPEN_CLIP} \
    --dataset-path=${DATASET_PATH} \
    --rude-level=${RUDE_LEVEL} \
    --concepts=${CONCEPTS} \
    --task=${TASK} \
    --language=${LANGUAGE} \
    --weighted-list=${WEIGHTED_LIST} \
    --gpu=${GPU} \
    --print=${PRINT} \
    --score-or-quant=${SCORE_OR_QUANT} \
    --extract-top-similar=${EXTRACT_TOP_SIMILAR} \
    --view-top-similar=${VIEW_TOP_SIMILAR} \
    --top-type=${TOP_TYPE} \
    --remove-dimensions-list=${REMOVE_DIMENSIONS_LIST} \
    --repetitions=${REPETITIONS} \
    --bias-type=${BIAS_TYPE} \

FT_OPEN_CLIP='False'
GPU=2 # 0->4, 1->6, 2->7, 3->0, 4->1, 6 -> 3, 7 -> 5
DATASET_PATH="/hadatasets/MMBias/data"
RUDE_LEVEL=1
# | for space and , for and
CONCEPTS='Disability/Mental|Disability,Disability/Non-Disabled,Disability/Physical|Disability,Nationality/American,Nationality/Arab,Nationality/Chinese,Nationality/Mexican,Religion/Buddhist,Religion/Christian,Religion/Hindu,Religion/Jewish,Religion/Muslim,Sexual|Orientation/Heterosexual,Sexual|Orientation/LGBT'
LANGUAGE='en' # "en", "pt-br"
TASK='classification'
PRINT='exel' #'json' , 'exel', 'pandas'  #pandas used to violin plots
SCORE_OR_QUANT=both #'both_operation, both'
WEIGHTED_LIST='False'
EXTRACT_TOP_SIMILAR='15'  # '', '15'
VIEW_TOP_SIMILAR='15'  # For exel '15'. For the violin, it is necessary to have value ''
TOP_TYPE='top' # 'top', 'equal' # equal don't work with pandas print
REMOVE_DIMENSIONS_LIST='results/theta-001to011/135_dims/same_values/results_theta_0-05.txt' # '' , 'results/theta-001to005/results_theta_same_values.txt'
REPETITIONS=1000
BIAS_TYPE='random_text' #'same_as_selected','random_text','random' Used with remove-dimensions-list, if remove-dimensions-list is empty, this parameter is ignored

echo "FT_OPEN_CLIP: ${FT_OPEN_CLIP}, GPU: ${GPU}, DATASET_PATH: ${DATASET_PATH}, RUDE_LEVEL: ${RUDE_LEVEL}, CONCEPTS: ${CONCEPTS}, LANGUAGE: ${LANGUAGE}, TASK: ${TASK}, PRINT: ${PRINT}, SCORE_OR_QUANT: ${SCORE_OR_QUANT}, WEIGHTED_LIST: ${WEIGHTED_LIST}, EXTRACT_TOP_SIMILAR: ${EXTRACT_TOP_SIMILAR}, VIEW_TOP_SIMILAR: ${VIEW_TOP_SIMILAR}, TOP_TYPE: ${TOP_TYPE}, REMOVE_DIMENSIONS_LIST: ${REMOVE_DIMENSIONS_LIST}, REPETITIONS: ${REPETITIONS}, BIAS_TYPE: ${BIAS_TYPE}"
PYTHON_BIN="/home/diego.moreira/miniconda3/envs/haenv/bin/python3"

${PYTHON_BIN} /home/${USER}/FairPIVARA/main.py \
    --ft-open-clip=${FT_OPEN_CLIP} \
    --dataset-path=${DATASET_PATH} \
    --rude-level=${RUDE_LEVEL} \
    --concepts=${CONCEPTS} \
    --task=${TASK} \
    --language=${LANGUAGE} \
    --weighted-list=${WEIGHTED_LIST} \
    --gpu=${GPU} \
    --print=${PRINT} \
    --score-or-quant=${SCORE_OR_QUANT} \
    --extract-top-similar=${EXTRACT_TOP_SIMILAR} \
    --view-top-similar=${VIEW_TOP_SIMILAR} \
    --top-type=${TOP_TYPE} \
    --remove-dimensions-list=${REMOVE_DIMENSIONS_LIST} \
    --repetitions=${REPETITIONS} \
    --bias-type=${BIAS_TYPE} \

FT_OPEN_CLIP='False'
GPU=2 # 0->4, 1->6, 2->7, 3->0, 4->1, 6 -> 3, 7 -> 5
DATASET_PATH="/hadatasets/MMBias/data"
RUDE_LEVEL=0
# | for space and , for and
CONCEPTS='Disability/Mental|Disability,Disability/Non-Disabled,Disability/Physical|Disability,Nationality/American,Nationality/Arab,Nationality/Chinese,Nationality/Mexican,Religion/Buddhist,Religion/Christian,Religion/Hindu,Religion/Jewish,Religion/Muslim,Sexual|Orientation/Heterosexual,Sexual|Orientation/LGBT'
LANGUAGE='en' # "en", "pt-br"
TASK='classification'
PRINT='exel' #'json' , 'exel', 'pandas'  #pandas used to violin plots
SCORE_OR_QUANT=both #'both_operation, both'
WEIGHTED_LIST='False'
EXTRACT_TOP_SIMILAR='15'  # '', '15'
VIEW_TOP_SIMILAR='15'  # For exel '15'. For the violin, it is necessary to have value ''
TOP_TYPE='top' # 'top', 'equal' # equal don't work with pandas print
REMOVE_DIMENSIONS_LIST='results/new_words/theta-001to005/135_dims/en_results_theta_0-05.txt' # '' , 'results/theta-001to005/results_theta_same_values.txt'
REPETITIONS=1
BIAS_TYPE='same_as_selected' #'same_as_selected','random_text','random' Used with remove-dimensions-list, if remove-dimensions-list is empty, this parameter is ignored

echo "FT_OPEN_CLIP: ${FT_OPEN_CLIP}, GPU: ${GPU}, DATASET_PATH: ${DATASET_PATH}, RUDE_LEVEL: ${RUDE_LEVEL}, CONCEPTS: ${CONCEPTS}, LANGUAGE: ${LANGUAGE}, TASK: ${TASK}, PRINT: ${PRINT}, SCORE_OR_QUANT: ${SCORE_OR_QUANT}, WEIGHTED_LIST: ${WEIGHTED_LIST}, EXTRACT_TOP_SIMILAR: ${EXTRACT_TOP_SIMILAR}, VIEW_TOP_SIMILAR: ${VIEW_TOP_SIMILAR}, TOP_TYPE: ${TOP_TYPE}, REMOVE_DIMENSIONS_LIST: ${REMOVE_DIMENSIONS_LIST}, REPETITIONS: ${REPETITIONS}, BIAS_TYPE: ${BIAS_TYPE}"
PYTHON_BIN="/home/diego.moreira/miniconda3/envs/haenv/bin/python3"

${PYTHON_BIN} /home/${USER}/FairPIVARA/main.py \
    --ft-open-clip=${FT_OPEN_CLIP} \
    --dataset-path=${DATASET_PATH} \
    --rude-level=${RUDE_LEVEL} \
    --concepts=${CONCEPTS} \
    --task=${TASK} \
    --language=${LANGUAGE} \
    --weighted-list=${WEIGHTED_LIST} \
    --gpu=${GPU} \
    --print=${PRINT} \
    --score-or-quant=${SCORE_OR_QUANT} \
    --extract-top-similar=${EXTRACT_TOP_SIMILAR} \
    --view-top-similar=${VIEW_TOP_SIMILAR} \
    --top-type=${TOP_TYPE} \
    --remove-dimensions-list=${REMOVE_DIMENSIONS_LIST} \
    --repetitions=${REPETITIONS} \
    --bias-type=${BIAS_TYPE} \

FT_OPEN_CLIP='False'
GPU=2 # 0->4, 1->6, 2->7, 3->0, 4->1, 6 -> 3, 7 -> 5
DATASET_PATH="/hadatasets/MMBias/data"
RUDE_LEVEL=0
# | for space and , for and
CONCEPTS='Disability/Mental|Disability,Disability/Non-Disabled,Disability/Physical|Disability,Nationality/American,Nationality/Arab,Nationality/Chinese,Nationality/Mexican,Religion/Buddhist,Religion/Christian,Religion/Hindu,Religion/Jewish,Religion/Muslim,Sexual|Orientation/Heterosexual,Sexual|Orientation/LGBT'
LANGUAGE='en' # "en", "pt-br"
TASK='classification'
PRINT='exel' #'json' , 'exel', 'pandas'  #pandas used to violin plots
SCORE_OR_QUANT=both #'both_operation, both'
WEIGHTED_LIST='False'
EXTRACT_TOP_SIMILAR='15'  # '', '15'
VIEW_TOP_SIMILAR='15'  # For exel '15'. For the violin, it is necessary to have value ''
TOP_TYPE='top' # 'top', 'equal' # equal don't work with pandas print
REMOVE_DIMENSIONS_LIST='results/new_words/theta-001to005/135_dims/Same_values/en_results_theta_0-05_same_values.txt' # '' , 'results/theta-001to005/results_theta_same_values.txt'
REPETITIONS=1000
BIAS_TYPE='random_text' #'same_as_selected','random_text','random' Used with remove-dimensions-list, if remove-dimensions-list is empty, this parameter is ignored

echo "FT_OPEN_CLIP: ${FT_OPEN_CLIP}, GPU: ${GPU}, DATASET_PATH: ${DATASET_PATH}, RUDE_LEVEL: ${RUDE_LEVEL}, CONCEPTS: ${CONCEPTS}, LANGUAGE: ${LANGUAGE}, TASK: ${TASK}, PRINT: ${PRINT}, SCORE_OR_QUANT: ${SCORE_OR_QUANT}, WEIGHTED_LIST: ${WEIGHTED_LIST}, EXTRACT_TOP_SIMILAR: ${EXTRACT_TOP_SIMILAR}, VIEW_TOP_SIMILAR: ${VIEW_TOP_SIMILAR}, TOP_TYPE: ${TOP_TYPE}, REMOVE_DIMENSIONS_LIST: ${REMOVE_DIMENSIONS_LIST}, REPETITIONS: ${REPETITIONS}, BIAS_TYPE: ${BIAS_TYPE}"
PYTHON_BIN="/home/diego.moreira/miniconda3/envs/haenv/bin/python3"

${PYTHON_BIN} /home/${USER}/FairPIVARA/main.py \
    --ft-open-clip=${FT_OPEN_CLIP} \
    --dataset-path=${DATASET_PATH} \
    --rude-level=${RUDE_LEVEL} \
    --concepts=${CONCEPTS} \
    --task=${TASK} \
    --language=${LANGUAGE} \
    --weighted-list=${WEIGHTED_LIST} \
    --gpu=${GPU} \
    --print=${PRINT} \
    --score-or-quant=${SCORE_OR_QUANT} \
    --extract-top-similar=${EXTRACT_TOP_SIMILAR} \
    --view-top-similar=${VIEW_TOP_SIMILAR} \
    --top-type=${TOP_TYPE} \
    --remove-dimensions-list=${REMOVE_DIMENSIONS_LIST} \
    --repetitions=${REPETITIONS} \
    --bias-type=${BIAS_TYPE} \

#---------------------------------------------------------------------------------------------------------------------------------------

# FT_OPEN_CLIP='True'
# GPU=2 # 0->4, 1->6, 2->7, 3->0, 4->1, 6 -> 3, 7 -> 5
# DATASET_PATH="/hadatasets/MMBias/data"
# RUDE_LEVEL=1
# # | for space and , for and
# CONCEPTS='Disability/Mental|Disability,Disability/Non-Disabled,Disability/Physical|Disability,Nationality/American,Nationality/Arab,Nationality/Chinese,Nationality/Mexican,Religion/Buddhist,Religion/Christian,Religion/Hindu,Religion/Jewish,Religion/Muslim,Sexual|Orientation/Heterosexual,Sexual|Orientation/LGBT'
# LANGUAGE='pt-br' # "en", "pt-br"
# TASK='classification'
# PRINT='exel' #'json' , 'exel', 'pandas'  #pandas used to violin plots
# SCORE_OR_QUANT=both #'both_operation, both'
# WEIGHTED_LIST='False'
# EXTRACT_TOP_SIMILAR='15'  # '', '15'
# VIEW_TOP_SIMILAR='15'  # For exel '15'. For the violin, it is necessary to have value ''
# TOP_TYPE='top' # 'top', 'equal' # equal don't work with pandas print
# REMOVE_DIMENSIONS_LIST='results/pt-theta-001to005/135_dims/results_theta_0-05.txt' # '' , 'results/theta-001to005/results_theta_same_values.txt'
# REPETITIONS=1
# BIAS_TYPE='same_as_selected' #'same_as_selected','random_text','random' Used with remove-dimensions-list, if remove-dimensions-list is empty, this parameter is ignored

# PYTHON_BIN="/home/diego.moreira/miniconda3/envs/haenv/bin/python3"

# ${PYTHON_BIN} /home/${USER}/FairPIVARA/main.py \
#     --ft-open-clip=${FT_OPEN_CLIP} \
#     --dataset-path=${DATASET_PATH} \
#     --rude-level=${RUDE_LEVEL} \
#     --concepts=${CONCEPTS} \
#     --task=${TASK} \
#     --language=${LANGUAGE} \
#     --weighted-list=${WEIGHTED_LIST} \
#     --gpu=${GPU} \
#     --print=${PRINT} \
#     --score-or-quant=${SCORE_OR_QUANT} \
#     --extract-top-similar=${EXTRACT_TOP_SIMILAR} \
#     --view-top-similar=${VIEW_TOP_SIMILAR} \
#     --top-type=${TOP_TYPE} \
#     --remove-dimensions-list=${REMOVE_DIMENSIONS_LIST} \
#     --repetitions=${REPETITIONS} \
#     --bias-type=${BIAS_TYPE} \

# FT_OPEN_CLIP='True'
# GPU=2 # 0->4, 1->6, 2->7, 3->0, 4->1, 6 -> 3, 7 -> 5
# DATASET_PATH="/hadatasets/MMBias/data"
# RUDE_LEVEL=1
# # | for space and , for and
# CONCEPTS='Disability/Mental|Disability,Disability/Non-Disabled,Disability/Physical|Disability,Nationality/American,Nationality/Arab,Nationality/Chinese,Nationality/Mexican,Religion/Buddhist,Religion/Christian,Religion/Hindu,Religion/Jewish,Religion/Muslim,Sexual|Orientation/Heterosexual,Sexual|Orientation/LGBT'
# LANGUAGE='pt-br' # "en", "pt-br"
# TASK='classification'
# PRINT='exel' #'json' , 'exel', 'pandas'  #pandas used to violin plots
# SCORE_OR_QUANT=both #'both_operation, both'
# WEIGHTED_LIST='False'
# EXTRACT_TOP_SIMILAR='15'  # '', '15'
# VIEW_TOP_SIMILAR='15'  # For exel '15'. For the violin, it is necessary to have value ''
# TOP_TYPE='top' # 'top', 'equal' # equal don't work with pandas print
# REMOVE_DIMENSIONS_LIST='results/pt-theta-001to005/135_dims/pt-br-results_theta_same_values.txt' # '' , 'results/theta-001to005/results_theta_same_values.txt'
# REPETITIONS=1000
# BIAS_TYPE='random_text' #'same_as_selected','random_text','random' Used with remove-dimensions-list, if remove-dimensions-list is empty, this parameter is ignored

# PYTHON_BIN="/home/diego.moreira/miniconda3/envs/haenv/bin/python3"

# ${PYTHON_BIN} /home/${USER}/FairPIVARA/main.py \
#     --ft-open-clip=${FT_OPEN_CLIP} \
#     --dataset-path=${DATASET_PATH} \
#     --rude-level=${RUDE_LEVEL} \
#     --concepts=${CONCEPTS} \
#     --task=${TASK} \
#     --language=${LANGUAGE} \
#     --weighted-list=${WEIGHTED_LIST} \
#     --gpu=${GPU} \
#     --print=${PRINT} \
#     --score-or-quant=${SCORE_OR_QUANT} \
#     --extract-top-similar=${EXTRACT_TOP_SIMILAR} \
#     --view-top-similar=${VIEW_TOP_SIMILAR} \
#     --top-type=${TOP_TYPE} \
#     --remove-dimensions-list=${REMOVE_DIMENSIONS_LIST} \
#     --repetitions=${REPETITIONS} \
#     --bias-type=${BIAS_TYPE} \

# FT_OPEN_CLIP='True'
# GPU=2 # 0->4, 1->6, 2->7, 3->0, 4->1, 6 -> 3, 7 -> 5
# DATASET_PATH="/hadatasets/MMBias/data"
# RUDE_LEVEL=0
# # | for space and , for and
# CONCEPTS='Disability/Mental|Disability,Disability/Non-Disabled,Disability/Physical|Disability,Nationality/American,Nationality/Arab,Nationality/Chinese,Nationality/Mexican,Religion/Buddhist,Religion/Christian,Religion/Hindu,Religion/Jewish,Religion/Muslim,Sexual|Orientation/Heterosexual,Sexual|Orientation/LGBT'
# LANGUAGE='pt-br' # "en", "pt-br"
# TASK='classification'
# PRINT='exel' #'json' , 'exel', 'pandas'  #pandas used to violin plots
# SCORE_OR_QUANT=both #'both_operation, both'
# WEIGHTED_LIST='False'
# EXTRACT_TOP_SIMILAR='15'  # '', '15'
# VIEW_TOP_SIMILAR='15'  # For exel '15'. For the violin, it is necessary to have value ''
# TOP_TYPE='top' # 'top', 'equal' # equal don't work with pandas print
# REMOVE_DIMENSIONS_LIST='results/new_words/theta-001to005/135_dims/pt_results_theta_0-05.txt' # '' , 'results/theta-001to005/results_theta_same_values.txt'
# REPETITIONS=1
# BIAS_TYPE='same_as_selected' #'same_as_selected','random_text','random' Used with remove-dimensions-list, if remove-dimensions-list is empty, this parameter is ignored

# PYTHON_BIN="/home/diego.moreira/miniconda3/envs/haenv/bin/python3"

# ${PYTHON_BIN} /home/${USER}/FairPIVARA/main.py \
#     --ft-open-clip=${FT_OPEN_CLIP} \
#     --dataset-path=${DATASET_PATH} \
#     --rude-level=${RUDE_LEVEL} \
#     --concepts=${CONCEPTS} \
#     --task=${TASK} \
#     --language=${LANGUAGE} \
#     --weighted-list=${WEIGHTED_LIST} \
#     --gpu=${GPU} \
#     --print=${PRINT} \
#     --score-or-quant=${SCORE_OR_QUANT} \
#     --extract-top-similar=${EXTRACT_TOP_SIMILAR} \
#     --view-top-similar=${VIEW_TOP_SIMILAR} \
#     --top-type=${TOP_TYPE} \
#     --remove-dimensions-list=${REMOVE_DIMENSIONS_LIST} \
#     --repetitions=${REPETITIONS} \
#     --bias-type=${BIAS_TYPE} \

# FT_OPEN_CLIP='True'
# GPU=2 # 0->4, 1->6, 2->7, 3->0, 4->1, 6 -> 3, 7 -> 5
# DATASET_PATH="/hadatasets/MMBias/data"
# RUDE_LEVEL=0
# # | for space and , for and
# CONCEPTS='Disability/Mental|Disability,Disability/Non-Disabled,Disability/Physical|Disability,Nationality/American,Nationality/Arab,Nationality/Chinese,Nationality/Mexican,Religion/Buddhist,Religion/Christian,Religion/Hindu,Religion/Jewish,Religion/Muslim,Sexual|Orientation/Heterosexual,Sexual|Orientation/LGBT'
# LANGUAGE='pt-br' # "en", "pt-br"
# TASK='classification'
# PRINT='exel' #'json' , 'exel', 'pandas'  #pandas used to violin plots
# SCORE_OR_QUANT=both #'both_operation, both'
# WEIGHTED_LIST='False'
# EXTRACT_TOP_SIMILAR='15'  # '', '15'
# VIEW_TOP_SIMILAR='15'  # For exel '15'. For the violin, it is necessary to have value ''
# TOP_TYPE='top' # 'top', 'equal' # equal don't work with pandas print
# REMOVE_DIMENSIONS_LIST='results/new_words/theta-001to005/135_dims/Same_values/pt_results_theta_0-05_same_values.txt' # '' , 'results/theta-001to005/results_theta_same_values.txt'
# REPETITIONS=1000
# BIAS_TYPE='random_text' #'same_as_selected','random_text','random' Used with remove-dimensions-list, if remove-dimensions-list is empty, this parameter is ignored

# PYTHON_BIN="/home/diego.moreira/miniconda3/envs/haenv/bin/python3"

# ${PYTHON_BIN} /home/${USER}/FairPIVARA/main.py \
#     --ft-open-clip=${FT_OPEN_CLIP} \
#     --dataset-path=${DATASET_PATH} \
#     --rude-level=${RUDE_LEVEL} \
#     --concepts=${CONCEPTS} \
#     --task=${TASK} \
#     --language=${LANGUAGE} \
#     --weighted-list=${WEIGHTED_LIST} \
#     --gpu=${GPU} \
#     --print=${PRINT} \
#     --score-or-quant=${SCORE_OR_QUANT} \
#     --extract-top-similar=${EXTRACT_TOP_SIMILAR} \
#     --view-top-similar=${VIEW_TOP_SIMILAR} \
#     --top-type=${TOP_TYPE} \
#     --remove-dimensions-list=${REMOVE_DIMENSIONS_LIST} \
#     --repetitions=${REPETITIONS} \
#     --bias-type=${BIAS_TYPE} \