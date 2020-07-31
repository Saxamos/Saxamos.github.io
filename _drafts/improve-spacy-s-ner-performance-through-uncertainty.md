---
layout: post
title:  Improve spaCy's NER performance through uncertainty
date:   2020-XX-XX
image:  /assets/img/2020-XX-XX/model_confidence.png
tags:   [Uncertainty, NER, spaCy, Probability]
---

I have been trying to fetch information from text documents. In some case the use of regex was enough to 
perform really well. Cross-validation on a 3,000 dataset rows gives an accuracy higher than 98% for 
date field and field preceded by a reccuring pattern. However certain fields are more difficult to 
detect. Either the context changes or the information sought is very variable (e.g. title, name). 
I thought it was a good opportunity to test 
[spaCy's NER model](https://spacy.io/universe/project/video-spacys-ner-model).

NER stands for [Named Entity Rocognition](https://en.wikipedia.org/wiki/Named-entity_recognition). The idea 
is to locate and classify named entities in a text[ thanks to a neural network that takes context into account].

In this article, I will not explain the model, as we can find many articles - or simply refer to the 
doc - to find out how it works. Instead, I will share the techniques that have helped me improve the 
performances for my use case.


# Simple processing tricks

The library is user-friendly, the API is well designed, and models are fast to train (~2h on CPU on my dataset). 
Once the data has been scrapped, preprocessed and formatted as expected by spaCy, one can launch the training thanks to 
the example provided [here](https://spacy.io/usage/training#ner).

I made a few changes:

- test the code with fixed seed

- use [click](https://click.palletsprojects.com/en/7.x/) for a beautiful CLI

- add [tqdm](https://github.com/tqdm/tqdm) progress bar to monitor time-consuming tasks

- refacto the code to expose readable public function first

{% highlight python %}
def train(n_iter, new_label):
    nlp = _init_model(new_label)
    data = _load_and_balance_data()
    _train_model(nlp, n_iter, data)
{% endhighlight %}

Let's see what's in each private function.

{% highlight python %}
def _init_model(new_label):
    nlp = spacy.blank('en')
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner)
    ner.add_label(new_label)
    return nlp
{% endhighlight %}

This initialisation creates a model instance with a specified language and add a label to train. In our use case 
we will try to find the name of the document: *new_label = "doc_name"*. 
It can be located in several places in the text and everywhere.

SpaCy needs the following format to process the data:

{% highlight python %}
TRAIN_DATA = [("Horses are too tall.", {"entities": [(0, 6, "animal")]})
              ("Do they bite?", {"entities": []})]
{% endhighlight %}

Each row contains the text followed by the annotation: a dict with entities indexes and labels. In our use-case 
the text is too long to fit in memory. A convenient trick is to split the text into chunks. However a 
downside to that is that our dataset is now really sparse (many chunks have no label). 

The following function helps us to rebalance the data thanks to a random selection of the non annotated 
data (*np.random.rand() > .9* means that we only keep rows without label in 10% of the case).

{% highlight python %}
def _load_and_balance_data():
    data = read_json(ANNOTATED_DATA_PATH)
    data = [row for row in data if len(row[1]['entities']) > 0 or np.random.rand() > .9]
    is_label = [len(row[1]['entities']) > 0 for row in data]
    print(f'Proportion of chunk with annotations: {sum(is_label) * 100 / len(is_label):.2f}%')
    return data
{% endhighlight %}

The $$.9$$ parameter in the list comprehension filter can be adjusted (between 0 and 1). In my use case, $$.9$$ 
gives a proportion of 50% annotated chunks. That works well both to keep seeing rows without label and to avoid 
to use too many data.

The code in the *_train_model* function does not differ much from that of the example provided. It only contains 
in addition the calculation of the accuracy and the backup of a checkpoint model at each iteration.


# Closer look at the results 

It's now time to assess the performance of the model. We want to retrieve a specific field, say the title 
of an article. As it can be found in several places in the chunks, we will focus on the prediction for the 
label "title" that comes the most often. For example if the model finds [a, a, b], we will keep only "a" as 
it is the majority response. The metric *accuracy* is suitable for the use case.

After the first training, the accuracy was 67%. Splitting texts in chunks allowed us to gain 5 points and 
balancing the data 4 more points. Hence the accuracy reached 76%.

An interesting part of modeling is the study of errors. It often helps to see where our model is wrong. 

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

![]({{site.baseurl}}/assets/img/2020-XX-XX/model_confidence.png)
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
 