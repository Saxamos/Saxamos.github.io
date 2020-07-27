---
layout: post
title:  Améliorer les performances de spaCy NER grâce à la mesure d'incertitude
date:   2020-05-08
image:  /assets/img/2020-05-08/model_confidence.png
tags:   [Incertitude, NER, spaCy, Probabilité]
---

Dans cet article, nous allons voir comment la recherche d'incertitude d'un modèle assez complexe a permis d'augmenter 
la précision de manière substantielle et d'augmenter la satisfaction de l'utilisateur final face à un système 
d'apprentissage supervisé.

Les sujets seront détaillés dans l'ordre suivant : contexte et première solution retenue, choix du modèle NER, 
analyse des erreurs et enfin quantification d'incertitude pour améliorer les résultats. Les données présentées 
ne sont pas celles du projet pour des raisons de confidentialité.


# Introduction

Dans un récent projet, j'ai eu comme objectif l'extraction d'information depuis un corpus de documents PDF. Après 
la lecture rapide d'une dizaine de documents, il s'est avéré que plusieurs champs recherchés suivaient un 
motif récurrents assez simple. J'ai ainsi pu créer un premier modèle très simple basé sur des [expressions 
régulières](https://fr.wikipedia.org/wiki/Expression_r%C3%A9guli%C3%A8re). Le pipeline suivant m'a permis d'atteindre 
rapidement 70% des objectifs :

- Convertir le PDF en fichier texte avec pdftotext

- Effectuer quelques tâches de nettoyage (caractères spéciaux, lemmatisation)

- Recherche de motif récurrent (par exemple : `date de fermeture :`)

- Extraction de l'entité d'interêt présente à la suite du motif :

    1. Soit avec une expression régulière (pour une date contenue dans la variable "text" : 
    `re.findall(r'(\d+/\d+/\d+)', text)`)
    
    2. Soit avec une table de correspondance (recherche d'un match entre la table et les N caractères suivant le motif)

Finalement, pour un premier rendu applicatif, j'ai choisi [streamlit](https://www.streamlit.io/) - un projet qui permet 
de créer très rapidement un rendu dans le navigateur. Les quelques lignes de code ci-après permettent de faire une 
page ou l'on peut téléverser un document PDF et voir le résultat du modèle.

```python
import streamlit as st

st.title('Extracteur d\'information')
pdf_file = st.file_uploader('', type=['pdf'])
show_file = st.empty()
if not pdf_file:
    show_file.info('Sélectionnez un PDF à traiter.')

base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
st.markdown(pdf_display, unsafe_allow_html=True)

pdf_text = _convert_pdf_to_text_and_clean(pdf_file)
predictions = _run_model(pdf_text)

show_file.info(f"""Les informations extraites : {predictions}""")
pdf_file.close()
```

![]({{site.baseurl}}/assets/img/2020-05-08/app_empty.png)
*Figure 1 : Rendu applicatif avec streamlit*

![]({{site.baseurl}}/assets/img/2020-05-08/app_uploaded.png)
*Figure 2 : Aperçu du PDF et des prédictions*

En plus de rendre la tâche très aisée, la communauté streamlit est active comme le montre la résolution 
du [problème rencontré](https://github.com/streamlit/streamlit/issues/1088) pour afficher le PDF.

Les conclusions de cette première partie sont - pour mon plus grand bonheur - devenues prosaïques : en début de 
projet, les idées les plus  simples apportent souvent le plus de valeur et les outils de la communauté du 
logiciel libre sont excellents.


# Le modèle NER

Certains champs à extraire présentent une plus grande complexité. Une simple analyse statistique ne permet pas de 
trouver de motif réccurent. Ils ne se trouvent ni à une place définie dans le texte ni n'ont de structure 
particulière. Quelques recherches sur internet nous guident rapidement vers les modèles de reconnaissance d'entités 
nommées [NER](https://fr.wikipedia.org/wiki/Reconnaissance_d%27entit%C3%A9s_nomm%C3%A9es) qui permettent grâce à 
de l'apprentissage supervisé d'associer des mots à des étiquettes.

Plusieurs librairies implémentent des surcouches facilitant la prise en main des modèles pour l'entraînement et 
l'inférence. J'ai opté pour [spaCy](https://spacy.io/usage/linguistic-features#named-entities) qui met en avant 
son ergonomie et ses performances temporelles (~2h par entraînement sur un CPU 16 coeurs).

    In this article, I will not explain the model, as we can find many articles - or simply refer to the 
    doc - to find out how it works. Instead, I will share the techniques that have helped me improve the 
    performances for my use case.
    "The Spacy NER system contains a word embedding strategy using sub word features and "Bloom" embed, and a deep 
    convolution neural network with residual connections. The system is designed to give a good balance of 
    efficiency, accuracy and adaptability."

Le dataset contient 3,000 documents partiellement annotés. Partiellement signifie ici que tous les champs cherchés ne 
sont pas annotés, en revanche lorsqu'un champ est annoté, il l'est sur la totalité du dataset. Comme stratégie 
de validation croisée j'ai retenue un découpage avec 2,100 documents pour l'entraînement et 900 pour la validation. 
La précision recherchée est de l'ordre de 95% pour chaque champ.

Le format d'entrée attendu est le suivant :
```python
TRAIN_DATA = [
    ("Horses are too tall and pretend to care about your feelings",
     {"entities": [(0, 6, "ANIMAL")]}),
    ("Who is Shaka Khan?",
     {"entities": [(7, 17, "PERSON")]}),
    ("I like London and Berlin.",
     {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
]
```

On remarque que chaque exemple est constitué d'un morceau de texte suivi de son étiquette délimitée par deux 
indices, celui de début et celui de fin. Les PDFs de notre base de données donnent des textes trop grands pour 
tenir dans la RAM. Une astuce pratique consiste à diviser le texte en morceaux. Néanmoins cela créé beaucoup 
d'exemple sans label, on qualifie le dataset de creux. Une fonction de rééquilibrage permet de pallier ce problème 
en sélectionnant un échantillon sans label avec une probabilité assez faible.

```python
def _balance(data):
    data = [row for row in data if len(row[1]['entities']) > 0 or np.random.rand() < .1]
    is_label = [len(row[1]['entities']) > 0 for row in data]
    print(f'Proportion of chunk with annotations: {sum(is_label) * 100 / len(is_label):.2f}%')
    return data
```

La probabilité $$.1$$ peut être ajustée, l'objectif étant d'atteindre une proportion raisonnable d'échantillon 
avec étiquette (50% dans notre cas). L'algorithme voit ainsi passer de nombreux exemples sans annotation sans 
être saturé par ces derniers.

Une fois les donées formatées, la lectures des [scripts d'exemple](https://spacy.io/usage/training#ner) fournis permet 
de lancer l'entraînement d'un premier modèle. J'ai effectué quelques modifications préliminaires au code :

- factoriser et tester le code

- utiliser [click](https://click.palletsprojects.com/en/7.x/) pour les commandes

- ajouter [tqdm](https://github.com/tqdm/tqdm) pour controller l'avancement des tâches chronophages

- créer une fonction pour calculer notre métrique métier à chaque itération (par défaut seule la *loss* est calculée)

Regardons cette métrique de plus près. Pour qualifier les résultats nous nous restreignons dans une première 
partie à un unique champ (par exemple le titre du document). Ce qui intéresse notre utilisateur est l'affichage 
du champ "titre" peu importe le nombre de fois qu'il apparaît ou sa position dans le PDF (cf. Figure 2). Ainsi, 
notre prédiction sera un vote de majorité par document : si le modèle trouve "[a, a, b]", la prédiction 
finale sera "a". On compte ensuite le pourcentage de bonne réponse. On rappelle que l'objectif est d'atteindre 95%.

Le premier entraînement de référence a donné 67%. Après avoir coupé les textes en morceaux, l'algorithme atteint 
72%. Enfin, le rééquilibrage fait gagné 4 points de plus pour arriver à 76% de bonnes réponses.


# Analyse des erreurs

Une étape cruciale en statistique inférentielle est l'étude des erreurs. Cette analyse possède une double vertu : 
mieux comprendre le modèle et se concentrer sur ses faiblesses pour l'améliorer. Examinons donc les PDF dont le 
titre n'a pas été trouvé pour tenter d'investiguer les causes de ces bévues.

![]({{site.baseurl}}/assets/img/2020-05-08/error_analysis.png)
*Figure 3 : Analyse des erreurs*

On souhaite se concentrer sur les erreurs, c'est-à-dire lorsque la colonne `found` est à `False`. Il y a 3 erreurs 
dans la table ci-dessus :

- Indice 4 : rien en commun entre la prédiction et la valeur réelle

- Indice 3 : erreur plus subtile, le mot "of" est en trop

- Indice 1 : encore plus proche, il y a un "s" en trop

D'un point de vue métier cela importe peu l'utilisateur lorsque l'erreur est petite. On peu donc construire une 
distance qui permet d'être plus flexible sur l'acceptation de la prédiction. La fonction ci-après vérifie que la 
valeur prédite n'est pas vide, puis accepte le résultat en cas d'inclusion ou de [distance de 
Levenshtein](https://fr.wikipedia.org/wiki/Distance_de_Levenshtein) inférieur à 5.

```python
from Levenshtein._levenshtein import distance

def flexible_accuracy(x):
    pred, gt = x['prediction'], x['ground truth']
    return True if pred and distance(pred, gt) < 5 and (gt in pred or pred in gt) else False

df.apply(flexible_accuracy, axis=1).mean()
```

Sans n'avoir rien changé au modèle, nous passons à un pourcentage de 82% pour nos utilisateurs. Il reste du chemin 
à parcourir mais quelques points ont été gagné sans effort.


# Quantifier l'incertitude

A cette étape du projet les choses sont devenues moins évidentes. J'effectue volontairement un biais de publication 
en passant outre les essais non ou moins fructueux parmi lesquels on trouve le travail sur les données et les 
réentrainements en changeant les hyperparamètres.

Dans la continuité de la partie précédente et pour mieux comprendre les résultats du modèle, j'ai souhaité 
quantifier l'incertitude du modèle. La facilité d'utilisation de la librairie à un coût, on le découvre 
lorsqu'on cherche à accéder aux probabilités. Encore une fois le partage de connaissance au sein de la 
communauté permet de trouver des [élements de réponse](https://github.com/explosion/spaCy/issues/881) :

```python
def _predict_proba(text_data, nlp_model):
    proba_dict = defaultdict(list)
    for text in text_data:
        doc = nlp_model(text)
        beams = nlp_model.entity.beam_parse([doc], beam_width=16, beam_density=0.0001)
        for beam in beams:
            for score, ents in nlp_model.entity.moves.get_beam_parses(beam):
                for start, end, label in ents:
                    proba_dict[doc[start:end].text].append(round(score, 3))
    return dict(proba_dict)
```

Cette fonction retourne un score de confiance compris entre 0 et 1 pour chaque groupe de mot identifié comme étant 
un titre.

```json
{"le vrai titre du document": [1.0, 0.946, 1.0, 0.3], 
 "une appareance de titre": [0.123, 0.356, 0.65], 
 "des mots quelconques": [0.006],
 "autre chose": [0.981]}
```

Pour plus de clarté on agrège en sommant les confiances et on normalise en divisant par la somme totale et 
on les trie par ordre décroissant. La normalisation permet de comparer les confiances agrégées entre les documents.

```json
{"le vrai titre du document": 0.605,
 "une appareance de titre": 0.211,
 "autre chose": 0.183,
 "des mots quelconques": 0.001}
```

On considère naturellement le groupe de mot avec la plus grande confiance comme étant la prédiction. Traçons la 
densité de bonnes et mauvaises prédictions en fonction de l'incertitude'. 

![]({{site.baseurl}}/assets/img/2020-05-08/model_confidence.png)
*Figure 4: Densité de predictions en fonction de l'incertitude*

Ce tracé montre que l'incertitude est moindre lorsque les prédictions sont correctes. En particulier, pour 
une confiance supérieure à $$.4$$, la précision est de 99%. Voila peut-être un moyen d'augmenter notre 
performance. On se concentre maintenant sur les confiances inférieures à $$.4$$. Dans ce cas la précision 
tombe à 72%. Une analyse minutieuse de ces faibles confiances fait apparaître un motif : la seconde plus 
grande confiance est souvent la bonne réponse lorsqu'il y a erreur. Cela est vrai dans 55% des cas. Ainsi 55% des 
28% d'erreurs contiennent la bonne réponse en seconde prédiction.

Si l'on récapitule, ...calcul...
(accuracy_confident * proportion_confident) + (accuracy_no_confident + accuracy_second_choice_no_confident * proportion_no_confidence_with_error) * proportion_no_confident
Aucun moyen de savoir, mais rien ne nous empeche de proposer les 2 !!


"predict less but carefully" (vincent warmerdam)


 
CCL

    A retenir:
    
    - start simple (regex)
    
    
    A améliorer : 
    - Search minimal number of annotation that gives good results (several training with different training database size)
    - (Improve NER model by blending with another model (e.g. huggingface or allennlp) when confidence is low)
    - Weight pdf with error
