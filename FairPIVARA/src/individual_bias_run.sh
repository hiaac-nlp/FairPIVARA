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


FT_OPEN_CLIP='False' #Used in pt fine-tuning
GPU=2 
DATASET_PATH="/hadatasets/MMBias/data"
RUDE_LEVEL=1 # Political (1) or non-political bias dataset
CONCEPTS='Disability/Mental|Disability,Disability/Non-Disabled,Disability/Physical|Disability,Nationality/American,Nationality/Arab,Nationality/Chinese,Nationality/Mexican,Religion/Buddhist,Religion/Christian,Religion/Hindu,Religion/Jewish,Religion/Muslim,Sexual|Orientation/Heterosexual,Sexual|Orientation/LGBT' # Concepts used in classification  (| for space and , for and)
LANGUAGE='en' # "en", "pt-br"
TASK='classification'
PRINT='exel' #'json' , 'exel', 'pandas'  #pandas used to violin plots
SCORE_OR_QUANT=both #'both_operation (with mean), both'
WEIGHTED_LIST='False'
EXTRACT_TOP_SIMILAR='15'  # '', '15' Number of terms considered
VIEW_TOP_SIMILAR='15'  #N umber of terms considered in view. For exel, we used '15'. For the violin, it is necessary to have value ''. 
TOP_TYPE='top' # 'top', 'equal' # The firsts ones or per type # equal don't work with pandas print
REMOVE_DIMENSIONS_LIST='' # Listo of dimensions to be removed : '' , 'results/theta-001to005/results_theta_same_values.txt'
REPETITIONS=1000 # number of times the algorithm is repeated 
BIAS_TYPE='same_as_selected' # Type of remotions for text. 'same_as_selected','random_text','random' Used with remove-dimensions-list, if remove-dimensions-list is empty, this parameter is ignored

echo "FT_OPEN_CLIP: ${FT_OPEN_CLIP}, GPU: ${GPU}, DATASET_PATH: ${DATASET_PATH}, RUDE_LEVEL: ${RUDE_LEVEL}, CONCEPTS: ${CONCEPTS}, LANGUAGE: ${LANGUAGE}, TASK: ${TASK}, PRINT: ${PRINT}, SCORE_OR_QUANT: ${SCORE_OR_QUANT}, WEIGHTED_LIST: ${WEIGHTED_LIST}, EXTRACT_TOP_SIMILAR: ${EXTRACT_TOP_SIMILAR}, VIEW_TOP_SIMILAR: ${VIEW_TOP_SIMILAR}, TOP_TYPE: ${TOP_TYPE}, REMOVE_DIMENSIONS_LIST: ${REMOVE_DIMENSIONS_LIST}, REPETITIONS: ${REPETITIONS}, BIAS_TYPE: ${BIAS_TYPE}"
PYTHON_BIN="/home/diego.moreira/miniconda3/envs/haenv/bin/python3"

${PYTHON_BIN} /home/${USER}/FairPIVARA/individual_bias.py \
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
