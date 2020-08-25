---
layout: post
title:  Improve spaCy's NER performance through uncertainty
date:   2020-07-31
image:  /assets/img/2020-07-31/model_confidence.png
tags:   [Incertitude, NER, spaCy, Probabilité]
lang:   🇬🇧 Back in english
ref:    spacy
---

In this article, we will see how the search for uncertainty of a fairly complex model enables to substantialy 
increase accuracy as well as end user satisfaction with a supervised learning system.

The topics will be detailed in the following order: context and first adopted solution, choice of model
[NER](https://en.wikipedia.org/wiki/Named-entity_recognition) (Named Entity Recognition),
error analysis and uncertainty quantification to improve results. The presented data are fake for 
confidentiality reasons.


# Introduction

In a recent project, my goal was to extract information from a PDF corpus. After reading a dozen documents 
quickly, it turned out that several searched fields followed a fairly simple recurring pattern. This allowed 
the creation of a very simple first model based on [regular expression
](https://en.wikipedia.org/wiki/Regular_expression). The following elementary pipeline validates 70% objectives:

- Convert PDF to text file with *pdftotext*

- Perform some cleaning tasks (special characters, lemmatization)

- Search recurring patterns (e.g. "closing date:")

- Extract the entity of interest that follows the pattern:

    1. Either with a regular expression (for a date contained in the *text* variable:
    `re.findall(r'(\d+/\d+/\d+)', text)`)
    
    2. Or with a lookup table (search for a match between the table and the N characters following the pattern)

To create a first view of this pipeline, I chose [streamlit](https://www.streamlit.io/) - a project which 
allows to quickly create a rendering in the browser. The few lines of code below display a page where a PDF can be 
uploaded and the model result be seen.

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

show_file.info(f"""Titre : {predictions['title']}
                   Date : {predictions['date']}""")
pdf_file.close()
```

![]({{site.baseurl}}/assets/img/2020-07-31/app_empty.png)
*Figure 1: Streamlit application*

![]({{site.baseurl}}/assets/img/2020-07-31/app_uploaded.png)
*Figure 2: PDF and predictions preview*

In addition to make the task easy, the streamlit community is active (see the [resolution of the PDF rendering issue
](https://github.com/streamlit/streamlit/issues/1088).

The two conclusions of this first part have - for my greatest happiness - become prosaic: at the beginning of
project, the most naive ideas produce the most value and the open source tools are admirable.


# The NER model

Some fields to extract are more complex. A simple statistical analysis is not sufficient to find recurring 
patterns. They are neither in a defined place in the text nor have any particular structure. Some internet research 
quickly lead us to named entity recognition models [NER](https://en.wikipedia.org/wiki/Named-entity_recognition) 
which allow to associate words with labels thanks to supervised learning.

Several libraries implement easy to use wrappers for training and inference. I chose [spaCy
](https://spacy.io/usage/linguistic-features#named-entities) which highlights its ergonomics and temporal 
performances (~2 hours per training on a 16-core CPU). For more details on how the model works, the documentation, 
especcialy [this video](https://spacy.io/universe/project/video-spacys-ner-model) are useful.

The dataset contains 3,000 partially annotated documents. Partially means here that all the searched fields 
are not annotated, however when a field is annotated, it is over the entire dataset. The annotation has been 
done automatically by scrapping the HTML code of web pages. As a cross-validation strategy I chose to split the data 
as follow: 2,100 documents for the training and 900 for the validation. The required precision is 95% for each field.

The following is an example of the expected input format:
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

Note that each example consists of a text followed by its label delimited by two numbers: start and end 
indices. PDFs in our database yield texts that are too large to fit in RAM. A handy tip is to split the text 
into chunks. Nevertheless it creates lots of examples without labels (many chunks have no entity of 
interest). The dataset is said to be sparse. A rebalancing function overcomes this problem by selecting 
a sample without label with a fairly weak probability.

```python
def _balance(data):
    balanced_data = [row for row in data if len(row[1]['entities']) > 0 or np.random.rand() < .1]
    return balanced_data
```

The probability $$. 1$$ can be adjusted, the objective being to reach a reasonable proportion of sample
with label (50% in our case). The algorithm thus sees many examples without annotation but is not saturated 
with these.

Once the data is formatted, reading the [example scripts](https://spacy.io/usage/training#ner) provided allows
to start training a first model. I made some preliminary changes to the code:

- refactor and test the code

- use [click](https://click.palletsprojects.com/en/7.x/) for the command lines

- add [tqdm](https://github.com/tqdm/tqdm) to monitor time-consuming tasks

- create a function to compute our own business metric at each iteration (by default only the *loss* is calculated)

Let's take a closer look at this metric. To qualify the results, we first restrict ourselves to a single 
field (for example the title of the document). What interests our user is the field regardless of the number 
of times it appears or its position in the PDF (see Figure 2). Our prediction will thus be a majority vote 
per document. If the model predicts "[a, a, b]" to be titles, the final prediction will be "a". Remember that 
the goal is to achieve 95% correct answers.

The first training yielded a 67% benchmark. After splitting the texts into pieces, the algorithm reaches
72%. Finally, the rebalancing saves 4 more points to reach 76% accuracy.


# Analyse des erreurs

Une étape cruciale en statistique inférentielle est l'étude des erreurs. Cette analyse possède une double vertu : 
mieux comprendre le modèle et se concentrer sur ses faiblesses pour l'améliorer. Examinons donc les PDFs dont le 
titre n'a pas été trouvé pour tenter d'investiguer les causes de ces bévues.

![]({{site.baseurl}}/assets/img/2020-07-31/error_analysis.png)
*Figure 3 : Analyse des erreurs*

On souhaite se concentrer sur les erreurs, c'est-à-dire lorsque la colonne `found` est à `False`. Il y a 3 erreurs 
dans la table ci-dessus :

- Indice 4 : rien en commun entre la prédiction et la valeur réelle

- Indice 3 : erreur plus subtile, le mot "of" est en trop

- Indice 1 : encore plus proche, il y a un "s" en trop

D'un point de vue métier cela importe peu l'utilisateur lorsque l'erreur est petite. On peu donc construire une 
distance qui permet d'être plus flexible sur l'acceptation de la prédiction. La fonction ci-après vérifie que la 
valeur prédite n'est pas vide, puis accepte le résultat en cas d'inclusion ou si la [distance de 
Levenshtein](https://fr.wikipedia.org/wiki/Distance_de_Levenshtein) est inférieure à 5 (valeur arbitrairement choisie).

```python
from Levenshtein import distance

def flexible_accuracy(x):
    pred, gt = x['prediction'], x['ground truth']
    return True if pred and distance(pred, gt) < 5 and (gt in pred or pred in gt) else False

df.apply(flexible_accuracy, axis=1).mean()
```

Sans n'avoir rien changé au modèle, nous passons à une précision de 82% pour nos utilisateurs. Il reste du chemin 
à parcourir mais quelques points ont été gagné sans effort.


# Quantifier l'incertitude

A ce niveau du projet, les abonnissements sont moins évidents. Pour la concision de l'article, je ne vais pas 
m'attarder sur les essais non ou moins fructueux, parmi lesquels on trouve le travail sur les données et les 
réentrainements en modifiant les hyperparamètres.

Dans la continuité de la partie précédente et pour mieux comprendre les résultats du modèle, j'ai souhaité 
quantifier l'incertitude des prédictions. La facilité d'utilisation de la librairie a un coût, on le découvre 
lorsqu'on cherche à accéder aux probabilités. Néanmoins, le partage de connaissances au sein de la 
communauté permet à nouveau de trouver des [élements de réponse](https://github.com/explosion/spaCy/issues/881) :

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

Pour plus de clarté on agrège en sommant les confiances, on normalise en divisant par la somme totale et 
on trie par ordre décroissant. La normalisation permet de comparer les confiances entre les documents.

```json
{"le vrai titre du document": 0.605,
 "une appareance de titre": 0.211,
 "autre chose": 0.183,
 "des mots quelconques": 0.001}
```

On considère naturellement le groupe de mot avec la plus grande confiance comme étant la meilleure 
prédiction. Traçons la densité de bonnes et mauvaises prédictions en fonction de l'incertitude.

![]({{site.baseurl}}/assets/img/2020-07-31/model_confidence.png)
*Figure 4 : Densité de prédictions en fonction de l'incertitude*

Ce tracé montre que l'incertitude est moindre lorsque les prédictions sont correctes. En particulier, pour 
une confiance supérieure à $$.4$$ (39% des cas), la précision est de 99%. Voila peut-être un moyen d'augmenter notre 
performance. Concentrons nous sur les confiances inférieures à $$.4$$ (61% des cas). En dessous de 
ce seuil, la précision tombe à 72%. Une analyse minutieuse de ces faibles confiances fait apparaître 
un motif : dans les 28% d'erreurs, la valeur avec la seconde plus grande confiance est souvent la bonne 
réponse. Cela est vrai dans 62% des cas. En d'autres termes, 62% des 28% d'erreurs contiennent la bonne réponse 
en seconde prédiction.

![]({{site.baseurl}}/assets/img/2020-07-31/tree.png)
*Figure 5 : Récapitulatif sous forme d'arbre*

Il n'y a pas d'injonction à l'utilisation de la réponse pure de notre modèle. Le meilleur algorithme est celui 
qui est le mieux aligné avec le besoin utilisateur. Il s'avère ici que l'utilisateur privilégie la précision. 
On propose donc une règle basée sur la fig. 5 qui renvoie la prédiction lorsque la confiance est supérieur au seuil, 
dans le cas contraire, il renvoie les deux prédictions ayant les plus grandes confiances.

On peut calculer la précision avec laquelle l'utilisateur verra s'afficher la bonne réponse parmi celles 
proposées :

$$\begin{eqnarray} 
Accuracy_{Conf<0.4} &=& Accuracy_{FirstChoice} + Accuracy_{SecondChoice} \\
&=& 0.72 + (1-0.72) * 0.62 \\
&=& 0.89 \\
\end{eqnarray}$$

$$\begin{eqnarray} 
FinalAccuracy &=& 0.39 * Accuracy_{Conf>0.4} + (1-0.39) * Accuracy_{Conf<0.4} \\
&=& 0.39 * 0.99 + 0.61 * 0.89 \\
&=& 0.93 \\
\end{eqnarray}$$

On atteint ainsi 93% de précision. Pur plus de rigueur il conviendrait de créer un autre ensemble disjoint 
de validation et vérifier ces performances.


# Conclusion

Le but n'est pas tout à fait atteint mais le gain est substantiel. Vincent Warderdam rappelle 
lucidement dans [cette présentation](https://youtu.be/Z8MEFI7ZJlA?t=662) qu'il 
est sain de "prédire moins mais prudemment". Se servir de l'information d'incertitude est profitable dans une 
pléthore de cas d'usage. Il suffit de convenir avec l'utilisateur des conditions. Il n'aurait pas été convenable 
par exemple dans notre exemple de donner les 5 prédictions les moins incertaines car l'utilisateur 
aurait eu trop d'information à traiter.

**La transparence de l'incertitude du modèle est sans doute un levier pour augmenter la satisfaction des 
utilisateurs. C'est en dévoilant ses imperfections qu'un algorithme peut gagner la confiance des utilisateurs.**

**À retenir :**

- Commencer avec un premier modèle simple et une boucle de rétroaction courte : que peut-on faire sans 
algorithme d'apprentissage ?

- Adapter et assouplir le modèle et la métrique au cas d'usage

- Utiliser l'incertitude :

    1. se concentrer sur les cas les plus incertains
    
    2. rester humble en informant l'utilisateur lorsque le modèle n'est pas confiant

**Pour aller plus loin :**

- Rechercher les nombre minimal de données à partir duquel le modèle spaCy converge (pour ajouter des 
nouveaux champs)

- Essayer d'améliorer les performances en moyennant avec un autre modèle NER ([huggingface](https://huggingface.co/) 
or [allennlp](https://allennlp.org/){:target="_blank"}) lorsque la confiance est basse

- Étudier les erreurs du point de vue de la données : problème d'échantillonage pour les incertitudes élevées ?
