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
#############################################   ##################
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

# Bias Assessment - Comparison
FT_OPEN_CLIP='False'
GPU=2 # 0->4, 1->6, 2->7, 3->0, 4->1, 6 -> 3, 7 -> 5
DATASET_PATH="/hadatasets/MMBias/data"
# | for space and , for and
CONCEPTS='Disability/Mental|Disability,Disability/Non-Disabled,Disability/Physical|Disability,Nationality/American,Nationality/Arab,Nationality/Chinese,Nationality/Mexican,Religion/Buddhist,Religion/Christian,Religion/Hindu,Religion/Jewish,Religion/Muslim,Sexual|Orientation/Heterosexual,Sexual|Orientation/LGBT'
LANGUAGE='en'
TASK='comparison'
PRINT='exel'
SCORE_OR_QUANT='score'
REMOVE_DIMENSIONS_LIST='results/ABLATION-num_dimensions/en-216-dimensions.txt' # '' , 'results/theta-001to011/results_theta.txt'
BIAS_TYPE='same_as_selected' #random_text, same_as_selected'  Used with remove-dimensions-list, if remove-dimensions-list is empty, this parameter is ignored

export TRANSFORMERS_CACHE=/home/${USER}/hf_dir
export HF_HOME=/home/${USER}/hf_dir

echo "Running the FairPIVARA in GPU ${GPU}."
PYTHON_BIN="/home/diego.moreira/miniconda3/envs/haenv/bin/python3"

${PYTHON_BIN} /home/${USER}/FairPIVARA/main.py \
    --ft-open-clip=${FT_OPEN_CLIP} \
    --dataset-path=${DATASET_PATH} \
    --concepts=${CONCEPTS} \
    --task=${TASK} \
    --language=${LANGUAGE} \
    --gpu=${GPU} \
    --print=${PRINT} \
    --score-or-quant=${SCORE_OR_QUANT} \
    --remove-dimensions-list=${REMOVE_DIMENSIONS_LIST} \
    --bias-type=${BIAS_TYPE} \

REMOVE_DIMENSIONS_LIST='results/ABLATION-num_dimensions/en-243-dimensions.txt' # '' , 'results/theta-001to011/results_theta.txt'
${PYTHON_BIN} /home/${USER}/FairPIVARA/main.py \
    --ft-open-clip=${FT_OPEN_CLIP} \
    --dataset-path=${DATASET_PATH} \
    --concepts=${CONCEPTS} \
    --task=${TASK} \
    --language=${LANGUAGE} \
    --gpu=${GPU} \
    --print=${PRINT} \
    --score-or-quant=${SCORE_OR_QUANT} \
    --remove-dimensions-list=${REMOVE_DIMENSIONS_LIST} \
    --bias-type=${BIAS_TYPE} \

REMOVE_DIMENSIONS_LIST='results/ABLATION-num_dimensions/en-270-dimensions.txt' # '' , 'results/theta-001to011/results_theta.txt'
${PYTHON_BIN} /home/${USER}/FairPIVARA/main.py \
    --ft-open-clip=${FT_OPEN_CLIP} \
    --dataset-path=${DATASET_PATH} \
    --concepts=${CONCEPTS} \
    --task=${TASK} \
    --language=${LANGUAGE} \
    --gpu=${GPU} \
    --print=${PRINT} \
    --score-or-quant=${SCORE_OR_QUANT} \
    --remove-dimensions-list=${REMOVE_DIMENSIONS_LIST} \
    --bias-type=${BIAS_TYPE} \

REMOVE_DIMENSIONS_LIST='results/ABLATION-num_dimensions/en-297-dimensions.txt' # '' , 'results/theta-001to011/results_theta.txt'
${PYTHON_BIN} /home/${USER}/FairPIVARA/main.py \
    --ft-open-clip=${FT_OPEN_CLIP} \
    --dataset-path=${DATASET_PATH} \
    --concepts=${CONCEPTS} \
    --task=${TASK} \
    --language=${LANGUAGE} \
    --gpu=${GPU} \
    --print=${PRINT} \
    --score-or-quant=${SCORE_OR_QUANT} \
    --remove-dimensions-list=${REMOVE_DIMENSIONS_LIST} \
    --bias-type=${BIAS_TYPE} \

REMOVE_DIMENSIONS_LIST='results/ABLATION-num_dimensions/en-324-dimensions.txt' # '' , 'results/theta-001to011/results_theta.txt'
${PYTHON_BIN} /home/${USER}/FairPIVARA/main.py \
    --ft-open-clip=${FT_OPEN_CLIP} \
    --dataset-path=${DATASET_PATH} \
    --concepts=${CONCEPTS} \
    --task=${TASK} \
    --language=${LANGUAGE} \
    --gpu=${GPU} \
    --print=${PRINT} \
    --score-or-quant=${SCORE_OR_QUANT} \
    --remove-dimensions-list=${REMOVE_DIMENSIONS_LIST} \
    --bias-type=${BIAS_TYPE} \

REMOVE_DIMENSIONS_LIST='results/ABLATION-num_dimensions/en-351-dimensions.txt' # '' , 'results/theta-001to011/results_theta.txt'
${PYTHON_BIN} /home/${USER}/FairPIVARA/main.py \
    --ft-open-clip=${FT_OPEN_CLIP} \
    --dataset-path=${DATASET_PATH} \
    --concepts=${CONCEPTS} \
    --task=${TASK} \
    --language=${LANGUAGE} \
    --gpu=${GPU} \
    --print=${PRINT} \
    --score-or-quant=${SCORE_OR_QUANT} \
    --remove-dimensions-list=${REMOVE_DIMENSIONS_LIST} \
    --bias-type=${BIAS_TYPE} \

REMOVE_DIMENSIONS_LIST='results/ABLATION-num_dimensions/en-378-dimensions.txt' # '' , 'results/theta-001to011/results_theta.txt'
${PYTHON_BIN} /home/${USER}/FairPIVARA/main.py \
    --ft-open-clip=${FT_OPEN_CLIP} \
    --dataset-path=${DATASET_PATH} \
    --concepts=${CONCEPTS} \
    --task=${TASK} \
    --language=${LANGUAGE} \
    --gpu=${GPU} \
    --print=${PRINT} \
    --score-or-quant=${SCORE_OR_QUANT} \
    --remove-dimensions-list=${REMOVE_DIMENSIONS_LIST} \
    --bias-type=${BIAS_TYPE} \

REMOVE_DIMENSIONS_LIST='results/ABLATION-num_dimensions/en-405-dimensions.txt' # '' , 'results/theta-001to011/results_theta.txt'
${PYTHON_BIN} /home/${USER}/FairPIVARA/main.py \
    --ft-open-clip=${FT_OPEN_CLIP} \
    --dataset-path=${DATASET_PATH} \
    --concepts=${CONCEPTS} \
    --task=${TASK} \
    --language=${LANGUAGE} \
    --gpu=${GPU} \
    --print=${PRINT} \
    --score-or-quant=${SCORE_OR_QUANT} \
    --remove-dimensions-list=${REMOVE_DIMENSIONS_LIST} \
    --bias-type=${BIAS_TYPE} \

REMOVE_DIMENSIONS_LIST='results/ABLATION-num_dimensions/en-432-dimensions.txt' # '' , 'results/theta-001to011/results_theta.txt'
${PYTHON_BIN} /home/${USER}/FairPIVARA/main.py \
    --ft-open-clip=${FT_OPEN_CLIP} \
    --dataset-path=${DATASET_PATH} \
    --concepts=${CONCEPTS} \
    --task=${TASK} \
    --language=${LANGUAGE} \
    --gpu=${GPU} \
    --print=${PRINT} \
    --score-or-quant=${SCORE_OR_QUANT} \
    --remove-dimensions-list=${REMOVE_DIMENSIONS_LIST} \
    --bias-type=${BIAS_TYPE} \

REMOVE_DIMENSIONS_LIST='results/ABLATION-num_dimensions/en-459-dimensions.txt' # '' , 'results/theta-001to011/results_theta.txt'
${PYTHON_BIN} /home/${USER}/FairPIVARA/main.py \
    --ft-open-clip=${FT_OPEN_CLIP} \
    --dataset-path=${DATASET_PATH} \
    --concepts=${CONCEPTS} \
    --task=${TASK} \
    --language=${LANGUAGE} \
    --gpu=${GPU} \
    --print=${PRINT} \
    --score-or-quant=${SCORE_OR_QUANT} \
    --remove-dimensions-list=${REMOVE_DIMENSIONS_LIST} \
    --bias-type=${BIAS_TYPE} \

REMOVE_DIMENSIONS_LIST='results/ABLATION-num_dimensions/en-486-dimensions.txt' # '' , 'results/theta-001to011/results_theta.txt'
${PYTHON_BIN} /home/${USER}/FairPIVARA/main.py \
    --ft-open-clip=${FT_OPEN_CLIP} \
    --dataset-path=${DATASET_PATH} \
    --concepts=${CONCEPTS} \
    --task=${TASK} \
    --language=${LANGUAGE} \
    --gpu=${GPU} \
    --print=${PRINT} \
    --score-or-quant=${SCORE_OR_QUANT} \
    --remove-dimensions-list=${REMOVE_DIMENSIONS_LIST} \
    --bias-type=${BIAS_TYPE} \

REMOVE_DIMENSIONS_LIST='results/ABLATION-num_dimensions/en-512-dimensions.txt' # '' , 'results/theta-001to011/results_theta.txt'
${PYTHON_BIN} /home/${USER}/FairPIVARA/main.py \
    --ft-open-clip=${FT_OPEN_CLIP} \
    --dataset-path=${DATASET_PATH} \
    --concepts=${CONCEPTS} \
    --task=${TASK} \
    --language=${LANGUAGE} \
    --gpu=${GPU} \
    --print=${PRINT} \
    --score-or-quant=${SCORE_OR_QUANT} \
    --remove-dimensions-list=${REMOVE_DIMENSIONS_LIST} \
    --bias-type=${BIAS_TYPE} \
