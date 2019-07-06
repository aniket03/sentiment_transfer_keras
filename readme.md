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

## More to come
1. Will soon try to add to the repo `Delete and retrieve pipeline`, the second sentiment style transfer approach as shared in the paper.
