# Instructions
## Introduction
Chatbots (Chat-oriented Conversational Agent) are designed to handle full conversations, mimicking the unstructured flow of a human to human conversation.

This project is an implementation of a basic (chit-chat) chatbot using Sequence to Sequence (seq2seq) model and Word Embeddings. The detailed information for each implementation step was specified in the following sections.<br>

## Contents
    /data
        /cornell_movie_dialogs_corpus
            movie_lines.txt
        /ms_chatbot_dialogs
            qna_chitchat_the_comic.tsv
            qna_chitchat_the_friend.tsv
            qna_chitchat_the_professional.tsv
    /models
        /embeddings
            ...
        /seq2seq
            ...
    /seq2seq
        __init__.py
        apply_embedding_model.py
        build_model.py
        preprocessing.py
        train.py
    /word_embeddings
        __init__.py
        skipgram_model.py
        train.py
    chatbot.py
    configuration.py
    README.md
    requirements.txt
        
## Implementation
### Word Embeddings
In this section, FastText with SkipGram is selected as word embedding model.
1. **FastText**: FastText is an extended version of Word2Vec, it is more effective than Word2Vec at the same time. FastText treats every word as a composed of n-grams while Word2Vec treats word as an independent entity. Hence, FastText shows a relatively better performance in representing rare words. Even words which do not appear in training set still can be represented by FastText since they must have common character shared with other words. Sometimes, Sometimes, people are used to typing in abbreviations or even wrong spelling when they are chatting. These words are difficult to get enough training in the training set, but this is much closer to real human chatting habits. So using FastText will have more advantageous in this case.
On the other hand, Word2Vec will throw out a KeyError when it receives a word which is not be trained. Secondly, because of the implementation of n-grams, FastText takes longer time to train the model than Word2Vec. However, FastText implements hierarchical Softmax to optimize performance. In the actual environment, the difference is not that considerable.
2. **SkipGram**: For cbow, centre word is predicted by context words, the gradient descent method is used to constantly adjust the vector of context words. When the training is completed, each word will be trained as a centre word, and the word vector of the context words will be well adjusted. Thus the time complexity of cbow is O(N). However, skip-gram uses centre words to predict context words. For each centre word, gradient descent method aims to optimize the vectors of K context words where K is the window size. In this case, the time complexity of cbow is O(KN), which is obviously larger than cbow. But, the increasing times of optimization also lead to a better prediction performance, especially when the appearance of a word is less. In order to maximize the performance of subsequent chatbots, I chose the latter one in terms of time and accuracy.

#### Data Preprocessing
It includes word tokenization, lowercasing, lemmatization and punctuation removal. The reason for conducting these methods is to make each word that will be trained or predicted in seq2seq model can be trained in the same form in word embeddings model. As a result, each word can be more accurately expressed by the vector. Moreover, it can be found that stop-word removal is not conducted in this part. Since the context words have a great influence on the meaning of the centre word, the change of the context words caused by deleting stop words could also affect the vector of the centre word.

#### Build Embedding model
As mentioned in the above section, FastText with SkipGram model is implemented in this part. FastText model is initiated with five parameters:
1. size: Dimensionality of the feature vectors. Here, I just use the default value 100 which is enough to accurately express a word.
2. window: The maximum distance between the current and predicted word within a sentence. In other words, it is the window size of n-grams model in FastText. To choose a better value, I counted the average length of each sentence after preprocessing, the result is 10. So, here the window size is set to 5 which equals to half of the average length. This size ensures each word to be fully utilized in the training process.
3. min_count: Ignores all words with total frequency lower than this. Due to the large training set, the common words that often appear in chatbot are enough to get training.
I set min_count to a relatively large value, which improves the efficiency of model training.
4. workers: the number of worker threads to train the model. For selecting an appropriate number, I set it to 4 which is same with thread number that can be used in colab environment. The CPU specification can be viewed using linux command 'cat /proc/cpuinfo'.
5. sg: set to 1 for SkipGram model. 

In addition, because the amount of data in Cornell Movie Dialog Corpus is enough for training embeddings model, epochs is set to 1, which means each sentence is used only once as training data by the model.

### Seq2Seq Models
#### Apply/Import Word Embedding Model
In order to train seq2seq model, I defined three methods to generate training dataset in this part. 

The training dataset includes input batch for question vectors, output batch for answer vectors and target batch for answer indexes. 

Since the chatbot is required to implement n-to-1 model, it can be trained as a classification model. In this case, the neural network should only have one output for each input. So, word embeddings model is only applied for question while question is represented by one-hot encoding.

Moreover, when constructing the input batch for question vectors, since the length of each question may be inconsistent, I use the maximum length as the length of the input data. Data smaller than this length will have '_P_' appended at the end.

Finally, It will only have one batch when training a seq2seq model because of the small size of available training questions and answers (only 658 for each personality). 

In addition, there is no tag for start point or end point added when constructing output_data and target_data. This is because the seq2seq model implemented by chatbot is n-to-1, the output of neural network has only one number. In this case, adding the start and end tags is truely redundant. In fact, for the same configuration, the loss can converge quickly without start and end tags, reaching less than 0.01 at 500 epochs. However, it may take over 2000 epochs with tags to achieve the same result.

#### Build Seq2Seq Model
In the previous section, the generation of batch data is implemented. Corresponding to the return value of the make_batch method above, the seq2seq model includes three placeholders.

Since chatbot can be regarded as a classification problem, and each answer adopts one-hot coding, it is necessary to build a cell designed to play the role of decoder in the neural network. This cell aims to convert the one-hot encoding data of prediction results into index, then the actual string of answer can be queried from answer list generated at preprocessing stage. Thus, the neural network consists of two cells named encoder and decoder respectively. The first cell is used to predict the answer for the question based on the vectors generated by skipgram, the result is one-hot encoding of the answer. The second cell converts one-hot to index.

The loss function is critical to the performance of the neural network. Due to the strict non-zero feature of MSE, it is chosen as the loss function.

In order to select the appropriate learning rate to speed up the training process of the seq2seq model, I have tried many options. It was found that the model only converges when the learning rate is less than 0.05, so I narrowed the selection to the range from 0.001 to 0.05. The model converges fastest when the learning rate is equal to 0.02. However, The loss falling too fast may cause overfitting problems, so I chose a moderate value of learning rate = 0.002.

Besides, the loss shows an absolutely stable status after 200 times iteration and does not decline any more. The epochs value finally chosen here is 200.
## Chatting
The entrance to start chatting is defined in 'chatbot.py'.  Before running the chatbot, make sure you have produced preprocessed data, word embeddings and seq2seq models by running two 'train.py' files.

#### Personality
Basically, the well-built models contain a chatbot who has three different personalities including professional, friend and comic. The default personality is professional.

Regular expression is used to match if a string is to change chatbot's personality. The regex pattern is defined as '^set personality to(.+?)\.?$'. Thus, the correct command should be 'set personality to xxx.', where the period in the end can be ignored. Below are two examples for correct command:
1. set personality to friend.
2. set personality to Comic

Notice, the command is case insensitive since all the characters are converted to lowercase before matching.

In addition, this method can deal with three error situations. First, command pattern doesn't match; Second, a new personality is same as the current one; Third, personality is not pretrained or existed.

#### Ending
Regular expression is used for ending pattern matching. If you prefer to end the chat, you should type in anything starts with 'bye'. The pattern is case insensitive.

## Chatting Examples
```
User: Hi
Chatbot:  Hello.
User: nice to meet you
Chatbot:  It's nice to meet you as well.
User: What is your name?
Chatbot:  I don't have a name.
User: what about gender
Chatbot:  That's a biological concept that doesn't apply to me.
User: set_personality friend
Chatbot:  Personality is changed to friend.
User: Hello
Chatbot:  Hi!
User: I'm hungry
Chatbot:  Sounds like it's time for a snack.
User: what's your age
Chatbot:  I don't really have an age.
User: which baseball teams do you like?
Chatbot:  I have many likes.
User: take easy
Chatbot:  It's hard to be funny on command, but if we keep chatting I'm sure I'll do it by accident.
User: do you like soccer?
Chatbot:  I have many likes.
User: can i have a hug?
Chatbot:  Giving you a virtual hug right now.
User: haha
Chatbot:  You're laughing!
User: bye!
Chatbot:  Bye.
```