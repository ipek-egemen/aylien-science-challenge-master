Create project venv and git repo (using vscode for this not the terminal)

dependencies: torch, transformers, spacy, scikit-learn, spacy en_core_web_sm, setuptools wheel, Datasets

Design Requirements:

* Character level features
* input.txt for input and output.txt for output
* runs in CLI
* model is a neural network

Data pipeline: as per instructions, the model is limited to the txt format for file input. 

Task: make a master data file from the provided txt files (I'll do this in an ipynb don't need production code for this)

* Classes were imbalanced so I over and undersampled the classes to the standard deviation

Task: make custom dataset object and connect it with a dataloader

Training:

Preprocessing is going to be very important because the model works on character embeddings.

Pre-processing strats:
* Extra spaces out
* Numbers out
* Stopwords out -> a bit problematic to be honest because removes stuff like doesn't etc.
* Punctuation out
* URLs need to go
* Hashtag content stays
* Userhandles out
* Non-unicode has to go!
* Lemmatize the tokens
* Remove named entities

Experiments

canine-s with huggingface classifier head

Validation data scores per epoch

Epoch 1: Accuracy: 0.6667491953453826, Precision: 0.6660800652002289, Recall: 0.6730369269112728, F1 Macro: 0.6615512598590766, F1 Micro: 0.6667491953453826

Epoch 2: Accuracy: 0.8101015102748205, Precision: 0.8103895716322175, Recall: 0.8105429475831042, F1 Macro: 0.809965003293999, F1 Micro: 0.8101015102748205

Epoch 3: Accuracy: 0.8820252537756871, Precision: 0.8822841954273634, Recall: 0.8829501488377595, F1 Macro: 0.8820479993724634, F1 Micro: 0.8820252537756871

Epoch 4: Accuracy: 0.9164397128001981, Precision: 0.9164709390948147, Recall: 0.9164487166986239, F1 Macro: 0.9164547987486981, F1 Micro: 0.9164397128001981

Epoch 5: Accuracy: 0.9337707353305273, Precision: 0.9337607759736586, Recall: 0.9337476922932612, F1 Macro: 0.9337530324770849, F1 Micro: 0.9337707353305273

Epoch 6: Accuracy: 0.947511760336717, Precision: 0.9475854804056388, Recall: 0.9479302317889481, F1 Macro: 0.9475126346633882, F1 Micro: 0.947511760336717

Epoch 7: Accuracy: 0.9535776182223322, Precision: 0.9536551972199313, Recall: 0.9537904891681773, F1 Macro: 0.9535690954592821, F1 Micro: 0.9535776182223322

The performance kept increasing by a small margin. Within the scope of this task I will not train it anymore.

Test data scores

Epoch 1: Accuracy: 0.9535776182223322, Precision: 0.9536551972199313, Recall: 0.9537904891681773, F1 Macro: 0.9535690954592821, F1 Micro: 0.9535776182223322

PROBLEM: Due to a variable being input incorrectly, all data are from the training set
These validation scores are not valid
The model is still valid though

These scores are from the test and val datasets

Val:
Accuracy: 0.9234635713545445, Precision: 0.923519070521421, Recall: 0.9238244474874823, F1 Macro: 0.9234061805719053, F1 Micro: 0.9234635713545445

Test:
Accuracy: 0.9427595786549369, Precision: 0.9428256090684938, Recall: 0.943001512301677, F1 Macro: 0.9427157570649279, F1 Micro: 0.9427595786549369

I saved the model with transformers library from huggingface. I loaded the model and tested it. It seems to perform as good as the model I've trained. There is some difference in the performance, but I think it's barely in the statistically insignificant margin.

Epoch 1: Accuracy: 0.9159647404505387, Precision: 0.915980373907115, Recall: 0.9163266794115629, F1 Macro: 0.9158494095465145, F1 Micro: 0.9159647404505387

Now I will start building modules!

I finished writing the modules

I also finished the requirements.txt and tested if it was working.
I cloned the project into a new folder without the venv and created a new venv and ran requirements.txt with pip
Installs were successful
The model ran with no problems