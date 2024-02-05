## Fine-tuning regression on arbitrarily long texts

Data: https://www.cs.cornell.edu/people/pabo/movie-review-data/

Target task: prediction of movie rating via regression to the scale of 0.0-1.0 (the same that is used in dataset, but continuous)

### Problem:

Normally, transformers accept no more than 512 tokens on the input, including the beginning and end of text markers. Some of the texts in our dataset have number of tokens approaching 3K.

### Possible solutions

1. Truncate the input size
2. Use a model with a larger maximum sequence length
3. Split the input text into separate chunks and use the same label for each chunk
4. Run the input text through a text summarization model and train on the summarized version (kind of like dimensionality reduction)
5. **Pooling / sliding window: split text into chunks, then take the average of each chunk's prediction**, as described here: https://github.com/google-research/bert/issues/27

I implement the last approach.

Some resources I used:

https://towardsdatascience.com/how-to-apply-transformers-to-any-length-of-text-a5601410af7f

Good chunking solution, but only for inference, not fine-tuning.

#### Main problem I ran into:

I couldn't figure out how to batch the different length input first with mapping the dataset for tokenizing, and then with torch's DataLoader: data instances couldn't be stacked into a batch tensor due to a variable amount of chunks. Found this resource offering a solution:

https://github.com/mim-solutions/bert_for_longer_texts

It is proposed to simply force torch to have each batch as a list, not a stacked tensor. It's a good enough solution, but it would probably slow down the training if we had more data. I don't like all code in this repository, but I borrowed their pooling fuction.

The longest string in the dataset had 6 chunks, so its input_ids had the shape of [6, 512]. I would be interested to see how the model would behave if we padded every data instance with zero tensors to have the shape [6, 512].

Finally, in order to fine-tune for regression, I simply specified `num_labels=1` while loading the pre-trained model, and used mean squared error as the loss function.

In order to evaluate how well the model performs, it would be a good idea to establish an MSE baseline (for instance, what is the MSE if every prediction is 0.5), and compare this to the current loss.

To improve this solution further, I'd look into pre-processing the text differently, fine-tuning the model hyperparameters, maybe using a learning rate scheduler, increasing the amount of epochs and stopping when the validation loss begins to increase, using a heavier transformer version.
