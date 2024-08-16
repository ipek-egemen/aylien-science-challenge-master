# Project Report

## File structure

* 'data': This folder contains the data used in the project.
* 'research': This folder contains the jupyter notebook 'make_model' which was used for data manipulation, model training, testing and saving.
* 'sentiment_classifier': This folder contains every module for our sentiment classifier app as well as the requirements.txt which is used for installing dependencies. It also contains a dockerfile, however, it was not used for the project.
* log.md: This is the developer log.
* project instructions.md: Project instructions provided by stakeholders.
* Readme.md: This file. It is a comprehensive report on the whole project.

## Task

* Design, implement and evaluate a Twitter sentiment classifier.
* Sentiment classifier input is a txt file with where each line contains a single tweet.
* When given an input in the format described above, the sentiment classifier will generate a txt file where each line, contains the label for the specific tweet corresponding to the lines in input.
* In the scope of the project, the labels are 'positive', 'neutral', and 'negative'. 

## Key constraints

* The model is based on the following architectures of CNN, RNN, or Transformer.
* The model operates on a character basis.

# Design

## Base Model

I have opted to use the Canine-s pre-trained transformer by Google.
It is a character-based transformer.
Model card: https://huggingface.co/google/canine-s

Transformers are currently the state-of-the-art in many NLP tasks. Therefore it makes sense to implement a transformer-based classifier for the project.

Unfortunately, we do not have enough data to pre-train a transformer from scratch. This is the rationale behind the inclusion of Google's Canine-s in the project. The strategy for creating our sentiment classifier consists of attaching Canine-s with a classifier head to create a cohesive model and fine-tune it.

## Data

The data counts :
* Positive Examples: 2364
* Neutral Examples: 3100
* Negative Examples: 9179

As you can observe from the counts, our data is fairly imbalanced.

Exposing the model to each class equally is important for our model to not bias towards classes with more examples. Since this is a character-based model, this is all the more important.

Strategy: Oversample positive and neutral classes and undersample the negative class to the standard deviation value of the example counts of all classes which was 3740.

Therefore, our training-validation-test set together contains:
* Positive Examples: 3740
* Neutral Examples: 3740
* Negative Examples: 3740
* Total data: 11220

This strategy is not optimal because there is data duplication and removal, however, the model turned-out good. There was no need to apply other strategies e.g., class weighting, SMOTE.

## Text preprocessing

Current SOTA transformer models with word-piece tokenizers don't need much processing to achieve good performance.

However, text pre-processing is going to be one of the most important steps for a character-based model. This is because in a character-based transformer, the attention layer will not be able to ignore less important tokens in the example as well as a, for instance, a word-piece-based transformer. Meaning, these less important tokens will make our data quite noisy in a character-based setting. 

In the consideration of the above, the text preprocessing strategy is as follows:
* Remove numbers: Numbers will not bring much useful information MAJORITY of the time. Although sometimes important, not removing all numbers is not worth the noise.
* Remove stopwords: Most of the stopwords will not bring anything meaningful regarding inference.
* Remove punctuation: Although some are important like the exclamation mark '!', not removing all punctuation is not worth the noise.
* Remove URLs: URLs could provide additional context to the example, however, our model can't interact with URLs, therefore they do not bring any useful information in this case.
* Remove user handle content: Removing the user handle content is common practice, however, it is especially important in this case. Manual inspection of the data shows that there are many instances where the user handle belongs to a corporate entity. Removing the user handle content will prevent our model to have a bias towards a connection between certain class and corporate entity.
* Lemmatize tokens: Lemmatization is a good strategy for a character-based model. With minimal loss of information, it allows us to reduce the noise in the data significantly.
* Remove named entities: The rationale behind removing named entities is the same as removing user handle content. 

## Training

The Canine-s base model was supplemented with the huggingface distribution of Canine-s classifier head.

Tokenization of the data, character-based, was carried out with Canine-s' tokenizer.

Training data was split into training, validation and test sets with the respective ratios of 0.7 (8078 examples), 0.2 (2020 examples), 0.1 (1122 examples), each having a balanced class representation.

General settings for training:
* Number of epochs: 7 (I have arrived at this number iteratively. I wanted to observe where the model stops learning a meaningful amount)
* Optimizer: AdamW with lr=2e-5. This setting is my starting point for fine-tuning any transformer model. Adam optimizer is one of the best performing optimizers available and AdamW is an improved implementation of Adam.
* Loss function: Pytorch's cross entropy loss is my go-to choice for multiclass classification problems. It applies softmax internally and it is very easy to implement. Very good performance too.

## Model Performance Results

Metrics:
* Accuracy: A widely used indicator of model performance. Doesn't provide in-depth insights about performance. But I include it in every project to get a rough indication of how the model is performing. It is also very good for communicating model performance to stakeholders who are not familiar with more complicated performance metrics used in the field. Everybody can understand what accuracy is.
* Precision and Recall: Standard stuff. The main indicator for classifier for performance is the F1 score and it is based on these two metrics. When the F1 score is low, I'll look at these two to try and identify issues. For instance, lower precision and high recall would allow me reduce the value for the decision threshold for signmoid_fn output, increasing performance.
* F1-Macro and F1-Micro: F1 Macro is the macro average of the model's F1 scores for each class. This is the main indicator of model performance in multiclass classification problems and it is widely used in nlp literature. F1-Micro is also included here. If there is a significant gap between the F1-Macro and F1-Micro scores, it indicates that model is having difficulties classifying one or more classes while being proficient at classifying classes.

Validation Scores:
* Epoch 1: Accuracy: 0.6667491953453826, Precision: 0.6660800652002289, Recall: 0.6730369269112728, F1 Macro: 0.6615512598590766, F1 Micro: 0.6667491953453826

* Epoch 2: Accuracy: 0.8101015102748205, Precision: 0.8103895716322175, Recall: 0.8105429475831042, F1 Macro: 0.809965003293999, F1 Micro: 0.8101015102748205

* Epoch 3: Accuracy: 0.8820252537756871, Precision: 0.8822841954273634, Recall: 0.8829501488377595, F1 Macro: 0.8820479993724634, F1 Micro: 0.8820252537756871

* Epoch 4: Accuracy: 0.9164397128001981, Precision: 0.9164709390948147, Recall: 0.9164487166986239, F1 Macro: 0.9164547987486981, F1 Micro: 0.9164397128001981

* Epoch 5: Accuracy: 0.9337707353305273, Precision: 0.9337607759736586, Recall: 0.9337476922932612, F1 Macro: 0.9337530324770849, F1 Micro: 0.9337707353305273

* Epoch 6: Accuracy: 0.947511760336717, Precision: 0.9475854804056388, Recall: 0.9479302317889481, F1 Macro: 0.9475126346633882, F1 Micro: 0.947511760336717

* Epoch 7: Accuracy: 0.9535776182223322, Precision: 0.9536551972199313, Recall: 0.9537904891681773, F1 Macro: 0.9535690954592821, F1 Micro: 0.9535776182223322

The performance kept increasing, but by a small margin. I do not think it's necessary to train it further.

Test Scores:

* Epoch 1: Accuracy: 0.9535776182223322, Precision: 0.9536551972199313, Recall: 0.9537904891681773, F1 Macro: 0.9535690954592821, F1 Micro: 0.9535776182223322

# Install and Use

## Create Venv

Navigate to the project folder using the terminal

Run the following command in the terminal:

python -m venv env

This command will create a virtual environment folder named 'env'

Run the following command in the terminal:

source env/bin/activate

This command will activate the newly created 'env' virtual environment.

## Install dependencies

Navigate to the project folder using the terminal

Run the following command in the terminal:

pip install -r sentiment_classifier/requirements.txt

This command will install all the dependencies to the virtual environment 'env'

## Run the sentiment classifier for inference

Navigate to the project folder using the terminal

Run the following command in the terminal:

python sentiment_classifier/main.py

This command will run the script for our sentiment classifier.

The script will prompt the user for inputs two times.

* Directory to the input path: Provide the path to the input file. This file should be in txt format and each data entry should have its own line.
* Example: /Users/egemenipek/projects/aylien-science-challenge-master/data/input.txt
* Directory of the export path: Provide the path to the output file. Do not forget to name your file and format it with .txt extention.
* Example: /Users/egemenipek/projects/aylien-science-challenge-master/data/output.txt

After the processing is complete, the script will print the string 'Success!' on terminal. The output will be found in the directory given by the user for the second prompt of the script: "Directory of the export path" e.g., /Users/egemenipek/projects/aylien-science-challenge-master/data/output.txt.

# IMPORTANT

The model can't be uploaded to GitHub due to its size.
If you wish to use the model, please refer to the 'make_model.ipynb' file in the 'research' folder.
There you can go ahead and create the same model from scratch.

