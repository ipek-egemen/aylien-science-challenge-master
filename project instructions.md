# Aylien NLP Data Scientist Coding Challenge

## The task

* Design, implement and evaluate a Twitter sentiment classifier: you must create a model that given a tweet, predicts whether the tweet is of positive, negative, or neutral sentiment.
* The dataset that you should use is provided (in the `data` directory), you may not use any additional data to train your model.
* You should provide a CLI to the (trained) model that takes two file paths as arguments, one will point to an existing input file, the other to a location to write an output file. The input file will consist of one tweet per line. Your CLI should produce an output file with one value per line, with each line being the predicted sentiment value for the corresponding line in the input file.
For example, given an input file (called `input.txt`):

```
This is a really great movie, very impressive.
Would not recommend.
```

Running:
```
> python predict.py --input input.txt --output output.txt
```

is expected to produce a file called `output.txt` containing:
```
positive
negative
```
* You should include a copy of the trained model in your submission.
* You should provide a suitable evaluation of the performance of your model.
* You should provide a brief report covering the evaluation results, possible shortcomings of the model and ideas for how to address them, and anything else that you think is important for us to understand your approach and code.

### Additional Constraints

* You must solve the task using a neural network, which must be based on a CNN, RNN or Transformer (or some closely-related architecture).
* You must *only* use character-level input in your model (so no subword tokens or word-level n-grams).

## Implementation notes

* Your code should run on recent versions of Debian/Ubuntu.
* You can use any programming language and any popular deep learning framework. We work primarily in Python using TensorFlow or PyTorch, so if you have no strong preference for an alternative then this is what we would prefer to see.
* Having said that, please try to keep the external dependencies of your implementation to a minimum. Instructions should be provided for installing anything that is not in the standard library of your chosen language (with exceptions for TensorFlow and PyTorch).
* We love readable, clean and maintainable code (so aim for close to production-ready).
* It is more important for the code to be clear than for your model to achieve the best possible performance.
* feel free to use our [open-source template for data science projects](https://github.com/AYLIEN/datascience-project-quickstarter) -(https://github.com/AYLIEN/datascience-project-quickstarter) - to bootstrap your project if you like. No pressure though, we won't evaluate based upon whether or not you choose to use the template.
* On LLMs like ChatGPT and Co-pilot etc... -- we expect our team to be able to write excellent code with no assistance. That being said, productive programmers use all the tools at their disposal, so make your own judgement. We will ask candidates to explain their solutions and we expect candidates to be able to easily explain and motivate every line of code. 

If you have any questions then please get in touch sooner rather than later. Good luck! 
