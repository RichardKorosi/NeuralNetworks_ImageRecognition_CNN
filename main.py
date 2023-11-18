import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

# ZDROJE KU KODOM ------------------------------------------------------------------------------------------------------
# ======================================================================================================================
# Zdrojove kody z cviceni (dostupne na dokumentovom serveri AIS):
#   Autor: Ing. Vanesa Andicsová
#   Subory:
#       seminar2.py
# Zdrojove kody boli vyuzite napriklad pre vypisy do konzoly, vytvorenie zakladnych grafov,
# ktore sme mali za ulohu  vypracovat
# Taktiez kody boli vyuzite pri zakladnom nastavovani vstupnych/vystupnych dat (X,y) a pri zakladnom nastavovani modelu
# ======================================================================================================================
# Grafy, Pomocne funkcie, Casti funkcii...:
#  Autor/Spoluautor: Github Copilot
#  Grafy, pomocne funkcie a casti funkcii boli vypracoavane za pomoci Github Copilota
# ======================================================================================================================


# Uvod -----------------------------------------------------------------------------------------------------------------
# Uvod bol inspirovany zdrojovim kodom seminar2.py (vid. ZDROJE KU KODOM)
pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
pd.set_option('mode.chained_assignment', None)
