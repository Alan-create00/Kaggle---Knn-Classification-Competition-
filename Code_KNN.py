# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 00:51:44 2025

@author: AlanD__ & Lazare AYACHI & Sofiane Beaumont
"""



import pandas as pd
import numpy as np
from collections import Counter
from itertools import product

# Chargement des données
donnees_entrainement = pd.read_csv('train.csv')
donnees_test = pd.read_csv('test.csv')

# Séparation des caractéristiques et des étiquettes
X_entrainement = donnees_entrainement.iloc[:, 1:8].values
y_entrainement = donnees_entrainement['Label'].values

X_test = donnees_test.iloc[:, 1:8].values

# Normalisation des caractéristiques
moyenne_X = np.mean(X_entrainement, axis=0)
ecart_type_X = np.std(X_entrainement, axis=0)

X_entrainement_normalise = (X_entrainement - moyenne_X) / ecart_type_X
X_test_normalise = (X_test - moyenne_X) / ecart_type_X

# Implémentation améliorée de SMOTE pour les classes minoritaires
def sur_echantillonnage_smote(X, y, k_voisins=5, etat_aleatoire=None):
    np.random.seed(etat_aleatoire)
    compteur = Counter(y)
    compte_max = max(compteur.values())
    classes = list(compteur.keys())
    
    X_reechantillonne = X.copy()
    y_reechantillonne = y.copy()
    
    for classe in classes:
        X_classe = X[y == classe]
        y_classe = y[y == classe]
        n_echantillons = X_classe.shape[0]
        n_echantillons_synthetiques = compte_max - n_echantillons
        
        if n_echantillons_synthetiques > 0:
            echantillons_synthetiques = []
            for i in range(n_echantillons):
                xi = X_classe[i]
                # Calcul des distances euclidiennes aux autres échantillons de la même classe
                distances = np.sqrt(np.sum((X_classe - xi) ** 2, axis=1))
                distances[i] = np.inf  # Exclure le point lui-même
                indices_voisins = np.argsort(distances)[:k_voisins]
                
                for _ in range(int(n_echantillons_synthetiques / n_echantillons) + 1):
                    indice_voisin = np.random.choice(indices_voisins)
                    x_voisin = X_classe[indice_voisin]
                    gap = np.random.rand()
                    echantillon_synthetique = xi + gap * (x_voisin - xi)
                    echantillons_synthetiques.append(echantillon_synthetique)
                    if len(echantillons_synthetiques) >= n_echantillons_synthetiques:
                        break
                if len(echantillons_synthetiques) >= n_echantillons_synthetiques:
                    break
            X_synthetique = np.array(echantillons_synthetiques)
            y_synthetique = np.array([classe] * X_synthetique.shape[0])
            X_reechantillonne = np.vstack((X_reechantillonne, X_synthetique))
            y_reechantillonne = np.hstack((y_reechantillonne, y_synthetique))
    return X_reechantillonne, y_reechantillonne

# Appliquer le suréchantillonnage SMOTE amélioré
X_reechantillonne, y_reechantillonne = sur_echantillonnage_smote(
    X_entrainement_normalise, y_entrainement, k_voisins=5, etat_aleatoire=42
)

# Définition d'une grille d'hyperparamètres plus précise
grille_parametres = {
    'n_voisins': range(1, 21),  # Tester de 1 à 20 voisins
    'poids': ['uniform', 'distance'],
    'metrique': ['euclidean', 'manhattan', 'minkowski']
}

combinaisons_parametres = list(product(
    grille_parametres['n_voisins'],
    grille_parametres['poids'],
    grille_parametres['metrique']
))

# Implémentation du classificateur KNN
class ClassifieurKNN:
    def _init_(self, n_voisins=5, poids='uniform', metrique='euclidean'):
        self.n_voisins = n_voisins
        self.poids = poids
        self.metrique = metrique
    
    def fit(self, X, y):
        self.X_entrainement = X
        self.y_entrainement = y
    
    def predict(self, X):
        y_predits = []
        for x in X:
            if self.metrique == 'euclidean':
                distances = np.sqrt(np.sum((self.X_entrainement - x) ** 2, axis=1))
            elif self.metrique == 'manhattan':
                distances = np.sum(np.abs(self.X_entrainement - x), axis=1)
            elif self.metrique == 'minkowski':
                distances = np.power(np.sum(np.abs(self.X_entrainement - x) ** 3, axis=1), 1/3)
            else:
                raise ValueError(f"Métrique inconnue : {self.metrique}")
            indices_voisins = np.argsort(distances)[:self.n_voisins]
            etiquettes_voisins = self.y_entrainement[indices_voisins]
            distances_voisins = distances[indices_voisins]
            if self.poids == 'uniform':
                etiquettes, comptes = np.unique(etiquettes_voisins, return_counts=True)
                y_predits.append(etiquettes[np.argmax(comptes)])
            elif self.poids == 'distance':
                distances_voisins[distances_voisins == 0] = 1e-5
                poids_voisins = 1 / distances_voisins
                poids_etiquettes = {}
                for etiquette, poids in zip(etiquettes_voisins, poids_voisins):
                    poids_etiquettes[etiquette] = poids_etiquettes.get(etiquette, 0) + poids
                y_predits.append(max(poids_etiquettes, key=poids_etiquettes.get))
            else:
                raise ValueError(f"Type de poids inconnu : {self.poids}")
        return np.array(y_predits)

# Implémentation de la validation croisée stratifiée
def validation_croisee_stratifiee(X, y, n_folds=5, melanger=True, etat_aleatoire=None):
    np.random.seed(etat_aleatoire)
    y = np.array(y)
    classes, y_indices = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    class_counts = np.bincount(y_indices)
    class_indices = [np.where(y_indices == i)[0] for i in range(n_classes)]
    if melanger:
        for indices in class_indices:
            np.random.shuffle(indices)
    indices_folds = [[] for _ in range(n_folds)]
    for indices_classe in class_indices:
        tailles_folds = np.full(n_folds, len(indices_classe) // n_folds, dtype=int)
        tailles_folds[:len(indices_classe) % n_folds] += 1
        current = 0
        for fold_idx, taille_fold in enumerate(tailles_folds):
            debut, fin = current, current + taille_fold
            indices_folds[fold_idx].extend(indices_classe[debut:fin])
            current = fin
    for fold_idx in range(n_folds):
        indices_test = indices_folds[fold_idx]
        indices_entrainement = np.hstack([indices_folds[i] for i in range(n_folds) if i != fold_idx])
        yield indices_entrainement, indices_test

# Recherche des meilleurs hyperparamètres
meilleur_score = 0
meilleurs_parametres = None

n_folds = 5
etat_aleatoire = 42

for n_voisins, poids, metrique in combinaisons_parametres:
    scores = []
    for indices_entrainement, indices_test in validation_croisee_stratifiee(
        X_reechantillonne, y_reechantillonne, n_folds=n_folds, melanger=True, etat_aleatoire=etat_aleatoire
    ):
        X_entrainement_cv = X_reechantillonne[indices_entrainement]
        X_test_cv = X_reechantillonne[indices_test]
        y_entrainement_cv = y_reechantillonne[indices_entrainement]
        y_test_cv = y_reechantillonne[indices_test]
        knn = ClassifieurKNN(n_voisins=n_voisins, poids=poids, metrique=metrique)
        knn.fit(X_entrainement_cv, y_entrainement_cv)
        y_pred_cv = knn.predict(X_test_cv)
        exactitude = np.mean(y_pred_cv == y_test_cv)
        scores.append(exactitude)
    score_moyen = np.mean(scores)
    if score_moyen > meilleur_score:
        meilleur_score = score_moyen
        meilleurs_parametres = {'n_voisins': n_voisins, 'poids': poids, 'metrique': metrique}

# Entraînement du modèle optimisé avec les meilleurs paramètres
knn_optimise = ClassifieurKNN(
    n_voisins=meilleurs_parametres['n_voisins'],
    poids=meilleurs_parametres['poids'],
    metrique=meilleurs_parametres['metrique']
)
knn_optimise.fit(X_reechantillonne, y_reechantillonne)

# Prédictions sur les données de test
predictions = knn_optimise.predict(X_test_normalise)

# Sauvegarde des résultats dans un fichier CSV
resultats = pd.DataFrame({
    'Id': donnees_test['Id'],
    'Label': predictions
})

resultats.to_csv('resultats_ameliore.csv', index=False)

print(f"Meilleurs paramètres trouvés : {meilleurs_parametres}")
print("Les résultats optimisés ont été sauvegardés dans 'resultats_ameliore.csv'.")