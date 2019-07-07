# Keras Implementation for Sentiment Style Transfer

This repository contains the keras implementation of sentiment style transfer using recurrent neural networks.

Model architecture and approach to perform sentiment style transfer are based on the paper
[Delete, Retrieve, Generate: A Simple Approach to Sentiment and Style Transfer](https://arxiv.org/abs/1804.06437)

Presently, the repository implements **Delete Only** pipeline explained in the above paper.

## Usage
1. Download Yelp reviews dataset from [here](https://github.com/rpryzant/delete_retrieve_generate).
   Extract the contents in folder `../data/sentiment_transfer_data/yelp`

2. Compute attribute markers for the dataset. Attribute markers are words/ phrases for a given attribute / label
   such as positive or negative sentiment, presence or absence of which, is indicative of the given sentiment.      
   Example positive sentiment attributes markers: [`comfortable`, `amazing`, `beautiful`]<br>
   Example negative sentiment attributes markers: [`horrible`, `careless`, `confused`]<br>
   To compute attribute markers basis training data run the script `find_attribute_markers.py`

3. Generate content only sentences from the given reviews and computed attribute markers using the script: `delete_attribute_markers.py`

4. Train the delete only architecture pipline using script `delete_only_pipeline.py`

5. Finally, to check how well the model got trained run the script `prediction_pipeline.py`, to obtain predictions from the learnt model.
   Presently, the pipeline uses greedy decoding strategy instead of beam search to generate the final text sequence.

## Sample results

Some good Results 😄

**Original review**: this place is just ok <br>
**Reverted review**: this place is awesome <br>

**Original review**: now the food , drinks , and desserts are amazing <br>
**Reverted review**: now the food drinks and desserts are terrible <br>

**Original review**: i had a horrible experience , and i sadly would not come back <br>
**Reverted review**: i had a great experience and i would definitely back <br>

Some not so good results 😔

**Original review**: no stars is what in want to give <br>
**Reverted review**: thank stars is what to want to <br>

**Original review**: this is the worst walmart neighborhood market out of any of them <br>
**Reverted review**: this is the most authentic store out of them <br>


## More to come
1. Will soon try to add to the repo `Delete and retrieve pipeline`, the second sentiment style transfer approach as shared in the paper.
2. `Delete Only` implementation which uses `beam search` decoding strategy.

## References
1. Research work: [Delete, Retrieve, Generate: A Simple Approach to Sentiment and Style Transfer](https://arxiv.org/abs/1804.06437)
2. Earlier implementations: 
    1. [From the authors](https://github.com/lijuncen/Sentiment-and-Style-Transfer)
    2. [Pytorch implementation](https://github.com/rpryzant/delete_retrieve_generate)
