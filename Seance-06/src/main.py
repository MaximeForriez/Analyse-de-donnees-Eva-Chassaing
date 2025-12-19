#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import math

#Fonction pour ouvrir les fichiers
def ouvrirUnFichier(nom):
    with open(nom, "r", encoding="utf8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

#Fonction pour convertir les données en données logarithmiques
def conversionLog(liste):
    log = []
    for element in liste:
        log.append(math.log(element))
    return log

#Fonction pour trier par ordre décroissant les listes (îles et populations)
def ordreDecroissant(liste):
    liste.sort(reverse = True)
    return liste

#Fonction pour obtenir le classement des listes spécifiques aux populations
def ordrePopulation(pop, etat):
    ordrepop = []
    for element in range(0, len(pop)):
        if (pop[element] == pop[element]):  # on véruifie que pop[element] n'est pas NaN
            ordrepop.append([pop[element], etat[element]])
    ordrepop.sort(key=lambda tup: tup[0], reverse=True)
    for element in range(0, len(ordrepop)):
        ordrepop[element] = [element + 1, ordrepop[element][1]]
    return ordrepop

#Fonction pour obtenir l'ordre défini entre deux classements (listes spécifiques aux populations)
def classementPays(ordre1, ordre2):
    classement = []
    if len(ordre1) <= len(ordre2):
        for element1 in range(0, len(ordre2)):
            for element2 in range(0, len(ordre1)):
                if ordre2[element1][1] == ordre1[element2][1]:
                    classement.append([ordre1[element2][0], ordre2[element1][0], ordre1[element2][1]])
    else:
        for element1 in range(0, len(ordre1)):
            for element2 in range(0, len(ordre2)):
                if ordre2[element2][1] == ordre1[element1][1]:
                    classement.append([ordre1[element1][0], ordre2[element2][0], ordre1[element1][1]])
    return classement

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Partie sur les îles
iles = pd.DataFrame(ouvrirUnFichier("./data/island-index.csv"))
#Attention ! Il va falloir utiliser des fonctions natives de Python dans les fonctions locales que je vous propose pour faire l'exercice. Vous devez caster l'objet Pandas en list().

# Q3
superficies = iles["Surface (km²)"].tolist() # comme indiqué, on cast le résult en list Python
new_val = [85545323, 37856841, 768030, 7605049]
superficies.extend([float(island) for island in new_val]) # on cast les valeurs en float comme indiqué

# Q4
superficies = ordreDecroissant(superficies) # pas de memory leak parce que ordreDecroissant utilise .sort()

# Q5
plt.figure(figsize=(10, 6)) # (largeur, hauteur)
plt.plot(superficies, linestyle='-')

plt.title("Superficie des îles selon leur rang")
plt.xlabel("Rang")
plt.ylabel("Superficie (en km²)")
plt.grid(True)
plt.legend()

# plt.show()
plt.savefig("q5")
plt.close() # puisqu'on va devoir plot de nouvelles données plus tard ..

# Q6
plt.figure(figsize=(10, 6)) # (largeur, hauteur)
plt.plot(conversionLog(superficies), linestyle='-')
plt.xscale('log') 
# On aurait pu utiliser plt.yscale('log') ce qui permet d'avoir une graduation logarithmique, mais là on fait les choses à la main!

plt.title("Superficie des îles selon leur rang\n échelle logarithmique")
plt.xlabel("Rang")
plt.ylabel("log(Superficie)")
plt.grid(True)
plt.legend()

# plt.show()
plt.savefig("q6")
plt.close() # puisqu'on va devoir plot de nouvelles données plus tard ..

# Pourquoi a-t-on une superficie négative ? 
print(superficies[40000]) 
print("")
# Comme on le voit, à partir d'un moment (environ le rangf 20000 d'après le plot) 
# on a des iles de superficie < 1km² donc leur logarithme est négatif

# Q7
# Puisqu'on a converti nos superficies en list Python pour opérer dessus plus facilement (via la méthode .tolist())
# on a perdu l'information sur les noms des iles que l'on manipule. Ce n'est pas dérengeant si l'on veut vérifier 
# que nos données suivent bien une loi rang-taille mais ça rend tout test sur les rangs inutiles (ou alors il faut
# faire des tests sur les rangs sans les étiquettes puis aller récuprer dans le DataFrame initial les étiquettes
# des iles... et encore ça ne marche que s'il n'y a pas de doublons!).
# Si l'on voulait faire des tests sur nos rangs et en tirer des informations, on aurait pu créer une liste Python
# superficie_avec_etiquette = [(area1, name1), ...] et faire le tri sur le premier champ du tuple. 


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Partie sur les populations des États du monde
#Source. Depuis 2007, tous les ans jusque 2025, M. Forriez a relevé l'intégralité du nombre d'habitants dans chaque États du monde proposé par un numéro hors-série du monde intitulé États du monde. Vous avez l'évolution de la population et de la densité par année.
monde = pd.DataFrame(ouvrirUnFichier("./data/Le-Monde-HS-Etats-du-monde-2007-2025.csv"))
#Attention ! Il va falloir utiliser des fonctions natives de Python dans les fonctions locales que je vous propose pour faire l'exercice. Vous devez caster l'objet Pandas en list().

# Q10
etat = monde["État"].tolist() 
pop2007 = monde["Pop 2007"].tolist() 
pop2025 = monde["Pop 2025"].tolist() 
densite2007 = monde["Densité 2007"].tolist() 
densite2025 = monde["Densité 2025"].tolist() 

# Q11
pop2007 = ordrePopulation(pop2007, etat)
pop2025 = ordrePopulation(pop2025, etat)
densite2007 = ordrePopulation(densite2007, etat)
densite2025 = ordrePopulation(densite2025, etat)

# Q12
classementPop = classementPays(pop2007, pop2025)
classementDens = classementPays(densite2007, densite2025)
classementPop.sort(key=lambda tup: tup[0])
classementDens.sort(key=lambda tup: tup[0])
# si on veut trier par rapport à la population / densité de 2025, il faut remplacer tup[0] par tup[1]

# Q13
popL1 = [pays[0] for pays in classementPop]
popL2 = [pays[1] for pays in classementPop]
densL1 = [pays[0] for pays in classementDens]
densL2 = [pays[1] for pays in classementDens]

# Q14
coeff_spearmanr_pop = scipy.stats.spearmanr(popL1, popL2)
coeff_spearmanr_dens = scipy.stats.spearmanr(densL1, densL2)
print("Coefficient de Spearman (population) : ", coeff_spearmanr_pop[0])
print("Coefficient de Spearman (densité) : ", coeff_spearmanr_dens[0])

coeff_kendalltau_pop = scipy.stats.kendalltau(popL1, popL2)
coeff_kendalltau_dens = scipy.stats.kendalltau(densL1, densL2)
print("Coefficient de Kendall (population) : ", coeff_kendalltau_pop[0])
print("Coefficient de Kendall (densité) : ", coeff_kendalltau_dens[0])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Question Bonus 1
long_cote = iles["Trait de côte (km)"].tolist()
surface = iles["Surface (km²)"].tolist()

n = len(long_cote)
longCoteId = sorted([[long_cote[i], i+1] for i in range(n)])
surfaceId = sorted([[surface[i], i+1] for i in range(n)])
for i in range(n):
    longCoteId[i] = [i+1, longCoteId[i][1]] # classement, id 
    surfaceId[i] = [i+1, surfaceId[i][1]] # classement, id 

# Equivalent d'une jointure SQL sur l'id (qu'on met dans le tableau classement)
# On trie les tableau par id pour aller *beaucoup* plus vite (temps linéaire au lieu de quadratique)
# Avec mon ordinateur la méthode naïve utilisée dans les question précédentes mettrai plusieurs heures à s'executer
classement = []
longCoteId.sort(key=lambda tup: tup[1])
surfaceId.sort(key=lambda tup: tup[1])
for i in range(n):
    classement.append([surfaceId[i][0], longCoteId[i][0], longCoteId[i][1]]) # classement surface, classement longueur cote, id

print("\nCorrélation surface / longueur des côtes ?")
print("Coefficient de Spearman :", 
    scipy.stats.spearmanr(
    [classement[i][0] for i in range(n)], 
    [classement[i][1] for i in range(n)])[0])
print("Coefficient de Kendall :", 
    scipy.stats.kendalltau(
    [classement[i][0] for i in range(n)], 
    [classement[i][1] for i in range(n)])[0])
print("")

# Solution alternative en une ligne :
# print("\n\n spearman : ", 
    #   scipy.stats.spearmanr(
    #       iles["Trait de côte (km)"].tolist(), 
    #       iles["Surface (km²)"].tolist()))


# Question Bonus 2
def analyser_concordance(annee_A, annee_B):
    """
    Calcule Spearman et Kendall entre les classements de deux années données.
    df : le DataFrame contenant les données
    annee_A, annee_B : les années à comparer (ex: '2007', '2008')
    """
    data_A = monde[f"Pop {annee_A}"].tolist()
    data_B = monde[f"Pop {annee_B}"].tolist()
    rho, _ = scipy.stats.spearmanr(data_A, data_B)
    tau, _ = scipy.stats.kendalltau(data_A, data_B)

    return rho, tau

annee_debut = 2015
annee_fin = 2025
annees = range(annee_debut, annee_fin + 1)

for annee in range(2007, 2025):
    if (analyser_concordance(annee, 2025) != analyser_concordance(annee, 2025)):
        print(f"Pop {annee} contient des données NaN")
    else : 
        print(f"Pop {annee} ne contient aucune données ! :)")

tableau_rangs = []
for annee in annees:
    # On transforme les valeurs en rangs (1 = plus peuplé)
    # rankdata donne 1 au plus petit, donc on prend l'opposé pour que 1 soit le plus grand
    rangs_annee = scipy.stats.rankdata(-monde[f"Pop {annee}"])
    tableau_rangs.append(rangs_annee)

tableau_rangs = np.array(tableau_rangs).T # on transpose pour que Lignes=Pays, Cols=Années
p = tableau_rangs.shape[1] # Nombre d'années
n = tableau_rangs.shape[0] # Nombre de pays

R = np.sum(tableau_rangs, axis=1) # Somme des rangs pour chaque pays (= r_i dans le cours)
R_bar = np.mean(R) # Moyenne des sommes des rangs
S = np.sum((R - R_bar)**2) # Somme des carrés des écarts (S)

# Formule du W
W = (12 * S) / (p**2 * (n**3 - n))

print(f"\nCoefficient de Concordance W de Kendall sur la période {annee_debut}-{annee_fin} : {W}")
