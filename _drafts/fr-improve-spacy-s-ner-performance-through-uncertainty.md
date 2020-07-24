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

Les sujets seront détaillés dans l'ordre suivant : contexte et première solution retenue, choix du modèle NER 
et astuces, quantification d'incertitude pour améliorer les résultats. Les données présentées ne sont pas celles 
du projet pour des raisons de confidentialité.


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
*Figure 1 : le rendu applicatif avec streamlit*

![]({{site.baseurl}}/assets/img/2020-05-08/app_uploaded.png)
*Figure 2 : l'aperçu du PDF et des prédictions*

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


# Quantifier l'incertitude

An interesting part of modeling is the study of errors. It often helps to see where our model is wrong. 







    A retenir:
    
    - start simple (regex)
    
    
    A améliorer : 
    - Search minimal number of annotation that gives good results (several training with different training database size)
    - (Improve NER model by blending with another model (e.g. huggingface or allennlp) when confidence is low)
    - Weight pdf with error




|    | found   | ground truth            | prediction              |
|---:|:--------|:------------------------|:------------------------|
|  0 | True    | atomic ionization       | atomic ionization       |
|  1 | False   | kinetics of germination | kinetics of germinations|
|  2 | True    | infrared image synthesis| infrared image synthesis|
|  3 | False   | quantum brakes          | quantum brakes of       |
|  4 | False   | matter waves            | probability amplitudes  |

What interests us most is errors (*found = False*). There are 3 errors in the table above:

- Index 4 is a big mistake as there is nothing in common between the prediction and the ground truth.

- Index 3 is close. There is a small artifact "of" that we want to get rid of.

- Index 1 is also close, there is just one extra "s".

Now the question is: does small artifacts really bother the user of our application? It turns out 
that no, those aren't critical for the use case. Hence the use of a more flexible metric is possible. We now 
calculate our accuracy with inclusion and levenshtein distance. Only index 4 remains wrong in the previous 
example. Below is the code that compute the *found* columns according to the new metric. 

{% highlight python %}
def flexible_accuracy(x):
    pred, gt = x['prediction'], x['ground truth']
    return True if distance(pred, gt) < 5 and (gt in pred or pred in gt) and pred else False

df.apply(flexible_accuracy, axis=1).mean()
{% endhighlight %}

The flexible accuracy is now equal to 82%.


# Uncertainty insights

One thing that is not straightforward with spaCy is how to get probabilities. The library is indeed designed 
in a pragmatic way (fast and easy to use), that is why things seems a bit obscure when you want to deep dive. 
However, one can find related topics through 
[stackoverflow](https://stackoverflow.com/questions/46934523/spacy-ner-probability) or 
[github](https://github.com/explosion/spaCy/issues/881). Answers on these sites helped me get the probability 
of the predictions instead of a start index, end index and a label. Based on that I have added the following 
function to my validation module.

{% highlight python %}
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
{% endhighlight %}

It retrieves the confidence score (between 0 and 1) for each part of my text seen as the name 
of the document (i.e. our unique label) and returns the following format:

```json
{"The real title": [1.0, 0.946, 1.0, 0.3], 
 "Another semblance of title": [0.123, 0.356, 0.65], 
 "something else": [0.006],
 "yet another error": [0.981]}
```

An easy trick to make this response more readble is to sum the probabilities:

```json
{"The real title": 3.246,
 "Another semblance of title": 1.129,
 "something else": 0.006,
 "yet another error": 0.981}
```

Let's sort it and divide it by the sum of the values in order to be able to compare confidence value for each documents 
(confidence value lies between 0 and 1). 

```json
{"The real title": 0.605,
 "Another semblance of title": 0.211,
 "yet another error": 0.183,
 "something else": 0.001}
```

We observe that the field "yet another error" that have been found only once with high confidence is now considered 
as low confidence prediction. On the contrary "The real title" that have been found several times with high confidence 
keep a high confidence after this transformation.

We consider that the first field with greatest confidence is our prediction. We now plot the histogram of right 
and wrong predictions according to the confidence value that we just computed.

![]({{site.baseurl}}/assets/img/2020-05-08/model_confidence.png)
*Figure 1: Density of right and wrong predictions according to confidence*

This figure shows that the more confident the model, the better the chances that the prediction will be correct. 
When the confidence is greater than $$0.4$$, the accuracy is 99%. 

To enhance the model, we want to focus on the errors. Let's take a closer look at the prediction with confidence 
smaller than $$0.4$$.

When we focus on the low confidence prediction, it is obvious




"predict less but carefully" (vincent warmerdam)
me achieve gap.
How to ??? The strategy will be...

Conclusion:
- spacy top
- dure de trouver proba
- worthhhh it


Fake data because... for privacy reasons
 