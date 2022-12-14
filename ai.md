# Crop Disease Detection From Leaves


The process of detecting crop diseases from leaves poses two problems

1. Detecting the leaves in the Image
2. Detecting the disease from the leaves

This two problems are beyond pure logic and they need to be solved using ARTITIFICAL INTELLIGENCE

> You can't write a program with only logic to detect disease from plant leaves we need to allow the computer to figure out the logic itself by training it on many examples to predict plant diease given their images

This is similar to how human beings learn how to do everyday tasks.

&nbsp;
> For example one can learn how to cook egusi soup by watching examples of how to do it on youtube. by doing this you are giving your brain examples. After watching a few videos or many (depending on how fast you learn ) your brain captures the most important steps and details of the process.

&nbsp;

Similarly by giving the computer a lot of Labelled images of healthy and infected plant leaves the Computer is able to create a function Tries to map The Plant leaf images to their labels (wether they are infected or not)

&nbsp;

> Just like a human being can get better at solving math problems by attempting many questions, failing and learning from it. a computer can also be taught to solve a problem by allowing attempt solving the problem in different cases and scoring it each time it fails or passes. as we would see later, the more examples we give the computer the better it gets at solving the problem.

# AI Terminologies
>Algorithm: A set of rules that a machine can follow to learn how to do a task.

&nbsp;
>Artificial intelligence: This refers to the general concept of machines acting in a way that simulates or mimics human intelligence. AI can have a variety of features, such as human-like communication or decision making.

&nbsp;
>Autonomous: A machine is described as autonomous if it can perform its task or tasks without needing human intervention.

&nbsp;
>Backward chaining: A method where the model starts with the desired output and works in reverse to find data that might support it.

&nbsp;
>Bias: Assumptions made by a model that simplify the process of learning to do its assigned task. Most supervised machine learning models perform better with low bias, as these assumptions can negatively affect results.

&nbsp;
>Big data: Datasets that are too large or complex to be used by traditional data processing applications.

&nbsp;
>Bounding box: Commonly used in image or video tagging, this is an imaginary box drawn on visual information. The contents of the box are labeled to help a model recognize it as a distinct type of object.

&nbsp;
>Chatbot: A chatbot is program that is designed to communicate with people through text or voice commands in a way that mimics human-to-human conversation.

&nbsp;
>Cognitive computing: This is effectively another way to say artificial intelligence. It’s used by marketing teams at some companies to avoid the science fiction aura that sometimes surrounds AI.

&nbsp;
>Computational learning theory: A field within artificial intelligence that is primarily concerned with creating and analyzing machine learning algorithms.

&nbsp;
>Corpus: A large dataset of written or spoken material that can be used to train a machine to perform linguistic tasks.

&nbsp;
>Data mining: The process of analyzing datasets in order to discover new patterns that might improve the model.

&nbsp;
>Data science: Drawing from statistics, computer science and information science, this interdisciplinary field aims to use a variety of scientific methods, processes and systems to solve problems involving data.

&nbsp;
>Dataset: A collection of related data points, usually with a uniform order and tags.

&nbsp;
>Deep learning: A function of artificial intelligence that imitates the human brain by learning from the way data is structured, rather than from an algorithm that’s programmed to do one specific thing.

&nbsp;
>Entity annotation: The process of labeling unstructured sentences with information so that a machine can read them. This could involve labeling all people, organizations and locations in a document, for example.

&nbsp;
>Entity extraction: An umbrella term referring to the process of adding structure to data so that a machine can read it. Entity extraction may be done by humans or by a machine learning model.

&nbsp;
>Forward chaining: A method in which a machine must work from a problem to find a potential solution. By analyzing a range of hypotheses, the AI must determine those that are relevant to the problem.

&nbsp;
General AI: AI that could successfully do any intellectual task that can be done by any human being. This is sometimes referred to as strong AI, although they aren’t entirely equivalent terms.

&nbsp;
>Hyperparameter: Occasionally used interchangeably with parameter, although the terms have some subtle differences. Hyperparameters are values that affect the way your model learns. They are usually set manually outside the model.

&nbsp;
>Intent: Commonly used in training data for chatbots and other natural language processing tasks, this is a type of label that defines the purpose or goal of what is said. For example, the intent for the phrase “turn the volume down” could be “decrease volume”.

&nbsp;
>Label: A part of training data that identifies the desired output for that particular piece of data.

&nbsp;
>Linguistic annotation: Tagging a dataset of sentences with the subject of each sentence, ready for some form of analysis or assessment.

&nbsp;
>Common uses for linguistically annotated data include sentiment analysis and natural language processing.

&nbsp;
>Machine intelligence: An umbrella term for various types of learning algorithms, including machine learning and deep learning.

&nbsp;
>Machine learning: This subset of AI is particularly focused on developing algorithms that will help machines to learn and change in response to new data, without the help of a human being.

&nbsp;
>Machine translation: The translation of text by an algorithm, independent of any human involvement.

&nbsp;
>Model: A broad term referring to the product of AI training, created by running a machine learning algorithm on training data.

&nbsp;
>Neural network: Also called a neural net, a neural network is a computer system designed to function like the human brain. Although researchers are still working on creating a machine model of the human brain, existing neural networks can perform many tasks involving speech, vision and board game strategy.

&nbsp;
>Natural language generation (NLG): This refers to the process by which a machine turns structured data into text or speech that humans can understand. Essentially, NLG is concerned with what a machine writes or says as the end part of the communication process.

&nbsp;
>Natural language processing (NLP): The umbrella term for any machine’s ability to perform conversational tasks, such as recognizing what is said to it, understanding the intended meaning and responding intelligibly.

&nbsp;
>Natural language understanding (NLU): As a subset of natural language processing, natural language understanding deals with helping machines to recognize the intended meaning of language — taking into account its subtle nuances and any grammatical errors.

&nbsp;
>Overfitting: An important AI term, overfitting is a symptom of machine learning training in which an algorithm is only able to work on or identify specific examples present in the training data. A working model should be able to use the general trends behind the data to work on new examples.

&nbsp;
>Parameter: A variable inside the model that helps it to make predictions. A parameter’s value can be estimated using data and they are usually not set by the person running the model.

&nbsp;
>Pattern recognition: The distinction between pattern recognition and machine learning is often blurry, but this field is basically concerned with finding trends and patterns in data.

&nbsp;
>Predictive analytics: By combining data mining and machine learning, this type of analytics is built to forecast what will happen within a given timeframe based on historical data and trends.

&nbsp;
>Python: A popular programming language used for general programming.

&nbsp;
>Reinforcement learning: A method of teaching AI that sets a goal without specific metrics, encouraging the model to test different scenarios rather than find a single answer. Based on human feedback, the model can then manipulate the next scenario to get better results.

&nbsp;
>Semantic annotation: Tagging different search queries or products with the goal of improving the relevance of a search engine.

&nbsp;
>Sentiment analysis: The process of identifying and categorizing opinions in a piece of text, often with the goal of determining the writer’s attitude towards something.

&nbsp;
>Strong AI: This field of research is focused on developing AI that is equal to the human mind when it comes to ability. General AI is a similar term often used interchangeably.

&nbsp;
>Supervised learning: This is a type of machine learning where structured datasets, with inputs and labels, are used to train and develop an algorithm.

&nbsp;
>Test data: The unlabeled data used to check that a machine learning model is able to perform its assigned task.

&nbsp;
>Training data: This refers to all of the data used during the process of training a machine learning algorithm, as well as the specific dataset used for training rather than testing.

&nbsp;
>Transfer learning: This method of learning involves spending time teaching a machine to do a related task, then allowing it to return to its original work with improved accuracy. One potential example of this is taking a model that analyzes sentiment in product reviews and asking it to analyze tweets for a week.

&nbsp;
>Turing test: Named after Alan Turing, famed mathematician, computer scientist and logician, this tests a machine’s ability to pass for a human, particularly in the fields of language and behavior. After being graded by a human, the machine passes if its output is indistinguishable from that of human participant’s.

&nbsp;
>Unsupervised learning: This is a form of training where the algorithm is asked to make inferences from datasets that don’t contain labels. These inferences are what help it to learn.

&nbsp;
>Validation data: Structured like training data with an input and labels, this data is used to test a recently trained model against new data and to analyze performance, with a particular focus on checking for overfitting.

&nbsp;
>Variance: The amount that the intended function of a machine learning model changes while it’s being trained. Despite being flexible, models with high variance are prone to overfitting and low predictive accuracy because they are reliant on their training data.

&nbsp;
>Variation: Also called queries or utterances, these work in tandem with intents for natural language processing. The variation is what a person might say to achieve a certain purpose or goal. For example, if the intent is “pay by credit card,” the variation might be “I’d like to pay by card, please.”

&nbsp;
>Weak AI: Also called narrow AI, this is a model that has a set range of skills and focuses on one particular set of tasks. Most AI currently in use is weak AI, unable to learn or perform tasks outside of its specialist skill set.
&nbsp;

[Source](https://www.telusinternational.com/insights/ai-data/article/50-beginner-ai-terms-you-should-know)

## Gathering the Data For training

&nbsp;

Getting data to traing an **AI MODEL** can be done in Three major ways

&nbsp;
1. Creating the dataset yourself manually  _in our case it would require snapping multiple images of different plant leaves and annotating_
2. Downloading it from publicly available websites that provide datasets for machine learning including. (The most ideal method for our use case since most of these datasets are posted by very brilliant scientist )
   * [kaggle](https://www.kaggle.com/) _These is the most popular dataset repository of all the sites_
   * [Lion5B](https://laion.ai/blog/laion-5b/) _used popularly by open ai for their large language models_
   * [Roboflow](https://roboflow.com/) _Popular for computer vision tasks_
   etc..
3. Scrapping The Data from the web using a bot (This method is very popular in Natural Language processing tasks that use a lot of text)
   and would not be  ideal for our use case.

After extensively searching the web we found two datasets in __Roboflow__ and __Kaggle__ (At the mercy of God).
Each dataset solved one problem from the two we mentioned at the beggining of these write up.
* [Dataset for leaves detection](https://public.roboflow.com/object-detection/plantdoc/1)
* [Dataset for leaves disease detection](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

&nbsp;

> Although the first dataset for leaves detection from roboflow contains dieases annotations on them, The images are not enough to create an accurate model for both leaves identification and disease classification _but it could be used in leaves identification we is a much simpler task_


## Sample Image for leaves detection 
![Sample Image for leaves Detection](RemoteCropDisease/resources/images/leaves_detection_thumbnail.png)

> Note The datatsets contains the images of the leaves and the coordinates for the bounding boxes including their labels

## Sample Image from leaves __Disease__ detection dataset
![Sample Image from leaves disease detection dataset](RemoteCropDisease/resources/images/disease_dataset_thumbnail.jpeg)

---
# Training the AI

we would be Training two _ai models_ for this projects
* leaves Detection
* leaves Disease Detection
  
## Training The AI model for leaves disease detection

&nbsp;

> We would be working on an interactive notebook online made freely available by __kaggle__ with powerful _gpus_ that speed up the training process of our ai models

The dataset contains mored than 8000 images split into
* Training
* Validation
* Testing


All images are grouped by the class the fall into which could be one of the following
```python
# all the classes to be predicted by the model
CLASSES = [
    "Tomato  Late_blight",
    "Tomato  healthy",
    "Grape   healthy",
    "Orange  Haunglongbing_(Citrus_greening)",
    "Soybean healthy",
    "Squash  Powdery_mildew",
    "Potato  healthy",
    "maize   Northern_Leaf_Blight",
    "Tomato  Early_blight",
    "Tomato  Septoria_leaf_spot",
    "maize   Gray_leaf_spot",
    "Strawberry Leaf_scorch",
    "Peach   healthy",
    "Apple   Apple_scab",
    "Tomato  Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato  Bacterial_spot",
    "Apple   Black_rot",
    "Blueberry  healthy",
    "Cherry  Powdery_mildew",
    "Peach   Bacterial_spot",
    "Apple   Cedar_apple_rust",
    "Tomato     Target_Spot",
    "Pepper_bell  healthy",
    "Grape   Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Potato  Late_blight",
    "Tomato  Tomato_mosaic_virus",
    "Strawberry  healthy",
    "Apple   healthy",
    "Grape   Black_rot",
    "Potato  arly_blight",
    "Cherry  healthy",
    "maize   Common_rust_",
    "Grape   Esca_(Black_Measles)",
    "Raspberry  healthy",
    "Tomato  Leaf_Mold",
    "Tomato  Two-spotted_spider_mite",
    "Pepper_bell Bacterial_spot",
    "maize  healthy",
]
```

> Training Data is used to train the neural network (Tune the model weights based on the parameters) _This is done automatically by the computer_

&nbsp;

> Validation Data is used to Tune network parameters such as _number of layers, number neurons in each layer, network architecture e.t.c_ we would be done manually by the person training the neural network (usually the neural doesn't train on validation data) but is evaluted using validation

&nbsp;

> Test is the final dataset for evaluating the models performance

&nbsp;

> This is similar to teaching a student , Then Giving them a test (validation in the case of our model) , then giving them an exam (testing in the case of our model)


## lets Do a quick Data analysis to see The distribution of images per class

```python
# Code to diplay a pie chart of the image distribution for each class
import matplotlib.pyplot as plt
import seaborn

train_cat_num = categories_percentage(train_dir)
def autopct_generator(limit):
    """Remove percent on small slices."""
    def inner_autopct(pct):
        return ('%.2f%%' % pct) if pct > limit else ''
    return inner_autopct

#define Seaborn color palette to use
#palette_color = seaborn.color_palette('bright')
fig1, ax1 = plt.subplots(figsize=(6, 5))
box = box = ax1.get_position()
ax1.set_position([box.x0, box.y0-box.height, box.width * 8, box.height*2])
# plotting data on chart
_, _, autotexts = ax1.pie(
    train_cat_num.values(), autopct=autopct_generator(7), startangle=90, radius=4000)
for autotext in autotexts:
    autotext.set_weight('bold')
ax1.axis('equal')
total = sum(train_cat_num.values())
plt.legend(
    loc='upper left',
    labels=['%s, %1.1f%%' % (
        l, (float(s) / total) * 100) for l, s in zip(train_cat_num.keys(), train_cat_num.values())],
    prop={'size': 20},
    bbox_to_anchor=(0.0, 1),
    bbox_transform=fig1.transFigure
)

```

These gives us the following result
![data plot](RemoteCropDisease/resources/images/data_plot.png)

> As we can see the images seem to be evenly distributed which means we have no problem


lets also preview some of the images with their associated class labels
```python
#vizualise the loaded images
def show_batch(image_batch, label_batch):
  fig = plt.figure(figsize=(20,20))
  fig.patch.set_facecolor('white')
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[n].title(), fontsize=14)
      plt.axis('off')
image_batch, label_batch = next(train_generator)
show_batch(image_batch, label_batch)
```
![images preview](RemoteCropDisease/resources/images/data_sample.png)

## About the model for leaves disease classification

We would be using a pretrained neural network that has been trained on imagenet dataset _one of the largest image datasets available_
called __mobilnetv2__ which is one of the small ones _since we have limited computing resources

Here is a summary of the model architecture
&nbsp;

![mobilenetv2 architecture](RemoteCropDisease/resources/images/mobilnetv2.jpeg)

> we modify the model by replacing its classification layer _which classifies over a thousand classes_ with ours which would classify the 38 classes above

&nbsp;

> Then we train on our dataset


```python 
#fit the model
history = model.fit(train_generator, 
                    epochs=epochs,
                    validation_data = validation_generator,
                    callbacks=[early_stopper, reduce_lr,model_checkpoint])
```

> look at the interactive notebook in the resources folder for more details

&nbsp;

---
# Training The model for Detecting Leaves
We use the Dataset Gotten from Roboflow Train This model
> The model is trained to draw bounding boxes around plant leaves and also predict their associated classes

&nbsp;
> This model is a very small model that has been optimized for speed rather than accuracy __it is very good at predicting the bounding boxes but not class names which is why we are augmenting it with the mobilnetv2 model that was mentioned previously _check the efficient.ipynb file for more details about the training of these model.



# Inferencing
This is the process of using the trained models on real world data
> Running on an edge device such as
>  * mobile phones
>  * Iot Devices etc

During the inferencing we combine both the efficient_lite model _model for detecting leaves_ with the mobilenetv2 _model for leaves diseases classification_

&nbsp;

> although the efficient lite model also gives class prediction for the detected leaves it has a poor accuracy on class prediction, so we discard the class predictions and use its bounding box predictions to crop out leaves in the image and pass them to the moblilnetv2 model to get more accurate class predictions

The following diagram summarizes the entire inferencing process
![inferencing](RemoteCropDisease/resources/images/inferencing.jpg)



