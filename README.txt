Names of the Students:

Kleomenis KOGIAA
Camille Morand_Duval
Félix Poirier
Michaël Ullah


Description of the solution:

classification model:
We use a transfer learning approach with the BERT model,
on top of which a single linear layer is added for the classification.

Specifically, we use the BertForSequenceClassification implementation of HuggingFace,
in a pair sentence classification scheme.

Also, we enrich the vocabulary of the corresponding tokenizer :
We add new tokens that correspond to the 12 aspects:
('ambience#general','food#quality','service#general', ..., 'restaurant#prices')
And we extend the size of the vocabulary of the BERT model accordingly,
meaning 12 new representation-vectors are added before training

Feature representation:
As mentioned, we use a pair classification scheme.
- The first sentence is the original one.
- The second sentence, or auxiliary sentence, is build from the ASPECT anf the TARGET.
ex of an auxiliary sentence: 'ambience#general - trattoria'

Feature preprocessing:
Sentences are tokenized with the tokenizer of the pretrained model.

Accuracy: 




