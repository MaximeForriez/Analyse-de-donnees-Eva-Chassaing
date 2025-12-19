#coding:utf8

import pandas as pd
import matplotlib.pyplot as plt

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/

with open("./data/resultats-elections-presidentielles-2022-1er-tour.csv","r", encoding="utf8") as fichier:
    contenu = pd.read_csv(fichier)

# Question 5
print (contenu.head(5))


# Question 6
nb_lignes = len(contenu)
nb_colonnes = len(contenu.columns)
print("\nLe nombre de lignes est :",nb_lignes)
print("Le nombre de colonnes est :",nb_colonnes)
print()

# Question 7
print(contenu.describe())
lstType = contenu.dtypes
print(lstType)


# Question 8
print("\nPremière ligne du tableau",contenu.head(0))
print()

# Question 9
inscrits = contenu['Inscrits'].sum()
print("Nombre d'inscrits :",inscrits)

# Question 10 
lstSum = []
for i, sNomCol in enumerate(contenu.columns) : 
    if lstType[i] == object : continue
    lstSum.append((sNomCol, contenu[sNomCol].sum()))
print()
print(lstSum)
print()

# Question 11 
import os # pour gérer des dossiers 

if not os.path.exists("./images"): # Crée le fichier images si celui-ci n'est pas déjà crée, ça évite d'avoir des erreurs quand on éxecute le code depuis un nouveau pc / quand on bouge le fichier .py
    os.makedirs("./images")
for index, row in contenu.iterrows():
    code = row['Code du département']
    plt.figure()
    plt.bar(['Inscrits', 'Votants'], [row['Inscrits'], row['Votants']])
    plt.title(f"Département {code} : Inscrits vs Votants")
    plt.tight_layout()
    plt.savefig(f"./images/bar_{code}.png")
    plt.close()


# Question 12 
for index, row in contenu.iterrows():
    code = row['Code du département']
    valeurs = [
        row['Blancs'],
        row['Nuls'],
        row['Exprimés'],
        row['Abstentions']
    ]

    labels = ['Blancs', 'Nuls', 'Exprimés', 'Abstentions']
    plt.figure()
    plt.pie(valeurs, labels=labels, autopct='%1.1f%%') # pour ne pas avoir trop de chiffres significatifs 
    plt.title(f"Répartition des votes - Département {code}")
    plt.tight_layout()
    plt.savefig(f"./images/pie_{code}.png")
    plt.close()

# Question 13 
plt.figure()
plt.hist(contenu['Inscrits'], bins=20, density=True)
# sans le paramètre density, on représente un diagramme en bâton et pas un histogramme, cf remarque
plt.title("Distribution des inscrits")
plt.xlabel("Nombre d'inscrits")
plt.ylabel("Densité")
plt.savefig("./images/histogramme_inscrits.png")
plt.close()
