# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:07:27 2020

@author: 20202407
"""

# Zet het excel bestand in dezelfde file als dit programma!
# Check de directories voor de foto's, inclusief je studentnummer na users
# Verander de naam van het excel bestand nog even

import pandas as pd # Nodig om het excelbestand te lezen
import shutil # Nodig om files te kopiëren, os kan daar vgm ook voor worden gebruikt

IDwaarden = pd.read_excel(r"class2020_group00_id.xlsx") # Je excel bestand met ID's.
IDwaarden = IDwaarden.values 

for i in range(len(IDwaarden)):
    pad_in = "C:\\Users\\20202407\\Documents\\OGO beeldanalyse\\ISIC-2017_Training_Data\\" # Het pad waar je de foto's uit haalt
    pad_uit = "C:\\Users\\20202407\\Documents\\OGO beeldanalyse\\Onze data\\" # Het pad waar de foto's in gaan
    pad_in = pad_in + IDwaarden[i, 0] + ".jpg" # Zet het ID achter het pad en .jpg erachteraan
    pad_uit = pad_uit + IDwaarden[i, 0] + ".jpg"
    shutil.copyfile(pad_in, pad_uit) # Kopieer de foto