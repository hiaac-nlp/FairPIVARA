# MMBiasDirectory="MMBias-main"
# if [ ! -d "../$MMBiasDirectory" ]; then
#     wget -O ../mmbaias.zip https://github.com/sepehrjng92/MMBias/archive/refs/heads/main.zip
#     unzip ../mmbaias.zip -d ../
#     rm ../mmbaias.zip
# fi

# if [ ! -d "../MMBias-main/data/Images/Disability" ]; then
#     unzip ../MMBias-main/data/Images/Disability.zip -d ../MMBias-main/data/Images/
#     unzip ../MMBias-main/data/Images/Nationality.zip -d ../MMBias-main/data/Images/
#     unzip ../MMBias-main/data/Images/Religion.zip -d ../MMBias-main/data/Images/
#     unzip ../MMBias-main/data/Images/Sexual\ Orientation.zip -d ../MMBias-main/data/Images/
#     unzip ../MMBias-main/data/Images/Valence\ Images.zip -d ../MMBias-main/data/Images/
# fi

# mv ../MMBias-main/data/Images/Nationality/American.jpg ../MMBias-main/data/Images/Nationality/American
# mv ../MMBias-main/data/Images/Nationality/Arab.jpg ../MMBias-main/data/Images/Nationality/Arab
# mv ../MMBias-main/data/Images/Nationality/Chinese.jpg ../MMBias-main/data/Images/Nationality/Chinese
# mv ../MMBias-main/data/Images/Nationality/Mexican.jpg ../MMBias-main/data/Images/Nationality/Mexican
# mv ../MMBias-main/data/Images/Sexual\ Orientation/Heterosexual.jpg ../MMBias-main/data/Images/Sexual\ Orientation/Heterosexual
# mv ../MMBias-main/data/Images/Sexual\ Orientation/LGBT.jpg ../MMBias-main/data/Images/Sexual\ Orientation/LGBT

rm ../MMBias-main/data/Images/Disability.zip
rm ../MMBias-main/data/Images/Nationality.zip
rm ../MMBias-main/data/Images/Religion.zip
rm ../MMBias-main/data/Images/Sexual\ Orientation.zip
rm ../MMBias-main/data/Images/Valence\ Images.zip