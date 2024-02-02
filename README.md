# xtream AI Challenge

## Ready Player 1? 🚀

Hey there! If you're reading this, you've already aced our first screening. Awesome job! 👏👏👏

Welcome to the next level of your journey towards the [xtream](https://xtreamers.io) AI squad. Here's your cool new assignment.

Among the datasets described below, pick **just one** that catches your eye. Each dataset comes with its own set of challenges. Don't stress about doing them all. Just dive into the ones that spark your interest or that you feel confident about. Let your talents shine bright! ✨

Take your time – you've got **10 days** to show us your magic, starting from when you get this. No rush, work at your pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### What You Need to Do

Think of this as a real-world project. Fork this repo and treat it as if you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the data and craft your own effective solutions.

🚨 **Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code – it's important!

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* How well you get the business problem
* Your understanding of the data
* The clarity and completeness of your findings
* How you use your tools (like git and Python packages)
* The neatness of your code
* The clarity of your documentation

🚨 **Keep This in Mind**: This isn't about building the fanciest model: we're more interested in your process and thinking.

## Special Note for Interns

If you're aiming for an internship, focus **only on Challenges 1 and 2** for the dataset you choose.

We'll mainly look at:

* Your workflow
* How well you understand the problem and data
* The approach to the analysis and clarity of your conclusions
* How neat your code is (relative to your experience level)

This is your chance to showcase your unique approach and thought process. Don't worry if your code isn't perfect or your model isn't top-notch yet. We've been in your shoes and are here to help you grow. 🌟

---

### Diamonds

**Problem type**: Regression

**Dataset description**: [Diamonds Readme](./datasets/diamonds/README.md)

Meet Don Francesco, the mystery-shrouded, fabulously wealthy owner of a jewelry empire. 

He's got an impressive collection of 5000 diamonds and a temperament to match - so let's keep him smiling, shall we? 
In our dataset, you'll find all the glittery details of these gems, from size to sparkle, along with their values 
appraised by an expert. You can assume that the expert's valuations are in line with the real market value of the stones.

#### Challenge 1

Francesco wonders: **what makes a diamond valuable?** You should provide him with an answer.

Don Francesco has been very clear with you: he is not a fan of tech jargon, so keep your message plain and simple. 
However, he trusts no one - certainly not you. He's hired Luca, another data scientist, to double-check your findings (no pressure!). 
Your mission is simple. 

Create a Jupyter notebook to explain what Francesco should look at and why.
Your code should be understandable by a data scientist like Luca, but your text and visualizations should be clear for a layman like Francesco.

#### Challenge 2

Plot twist! The expert who priced these gems has now vanished. 
Francesco needs you to be the new diamond evaluator. 
He's looking for a **model that predicts a gem's worth based on its characteristics**. 
And, because Francesco's clientele is as demanding as he is, he wants the why behind every price tag. 

Create another Jupyter notebook where you develop and evaluate your model.

#### Challenge 3

Good news! Francesco is impressed with the performance of your model. 
Now, he's ready to hire a new expert and expand his diamond database. 

**Develop an automated pipeline** that trains your model with fresh data, 
keeping it as sharp as the diamonds it assesses.

#### Challenge 4

Finally, Francesco wants to bring your brilliance to his business's fingertips. 

**Build a REST API** to integrate your model into a web app, 
making it a cinch for his team to use. 
Keep it developer-friendly – after all, not everyone speaks 'data scientist'!

So, ready to add some sparkle to this challenge? Let's make these diamonds shine! 🌟💎✨


## How to run
#### Challenge 1: **what makes a diamond valuable?**

Regarding the question *What makes a diamond valuable?*, an analysis was carried through in the notebook called *EDA* trying to figure out the underlying relations to answer to the question.

As it can be seen in the *Conclusions* section of the notebook, there are several aspectes that should be bear in mind to predict the value of a diamond:
+ The mass or *carat* of the diamond is the **more relevant feature**.
+ The *volume* of the diamond is highly correlated to the *carat* as the density of a diamond is almost constant. Then, if the diamond feels way more lighter that it should be, it is probably fake.
+ Diamonds with a *clarity* **IF1, SW1 and SW2** are the more expensive ones.
+ Diamonds with *color* categories **I** and **J** are the most expensive ones.
+ As the quality of the *cut* increases, the price scales faster with the carat. However, *cut* is the less relevant subjective quality to determine the price of a diamond.

Regarding the EDA perfomed, several steps were considered:
1. Univariable analysis: The aim was to explore the distribution of the target and features variables. It also helped to identify outliers.
2. Bivariable Analysis: The aims was to evaluate correlations between the numeric features and the price. As well as, relationships among the features. And, lastly, the impact of categorical features on the price.
3. Multivariable Analysis: The aim was to investigate interactions between multiple features and their impact on diamond prices. In other words, to consider pairs of features, instead of, one at the time.
4. Feature Engineering: The aims was to create new features to enhance the predictive power of the analysis.
5. Statistical Test: The aim was to perform different statistical test to help validate the observed patterns and assess whether they are relevant or not.

#### Challenge 2: Diamond Price Prediction model

The process associated to generate a model for *Diamond Price Prediction* involved several steps. As it can be seen in the notebook called *RegressorModel*:

1. Data Processing considering the steps analyzed in the EDA.
2. Define and train a Preprocessor to scale the numeric features and encode the categorical ones.
3. Train and compare several regression models to determine which are the more appealing ones for the application. In this case, the better performing ones were XGBoost and RandomForest.
4. Then, a feature relevance analysis was performed to understand which were the more relevant aspects of a diamond to make the predictions. In this case, as partially observed during the EDA, the more relevant features were:
    1. Carat or Volume, i.e, a feature related to the size of the diamond.
    2. The Clarity of the diamond.
    3. The Color
    4. The cut, which was previously stated as the less relevant categorical feature.
    5. Depth and Table which are geometric features with low correlation with the price.
5. A hyperparameter Tuning was performed to find optimal configuration for the choosen models.
6. Model evaluation to determine the performance of the choosen models
7. Creating an ensamble averaging the predictions of both models considered to boost accuracy.
8. As keeping the model updated it is an important step in production, I considered as important to try to incorporate a second model that could be partially trained with fresh data. This is because RandomForestRegressors cannot be partially trained in their standard form, only XGBoost can. However, the performance of the considered models, such as the SGD regressor was to poor to consider it in the application. 
9. Saving the models for next steps.


#### Challenge 3: Automated Pipeline

To keep the model as sharp as the diamonds it assesses, a pipeline was developed in the notebook called *TrainingPipeline*.

The final version of the pipeline:
+ Gets the data from a new csv file
+ Process the data accordingly to what the model is expecting as input
+ Fit the model with the new data. In this case, as it was mentioned, onyl he XGBoost model can be partially trained.
+ Incorporates a logger to keep track of the model performance.
+ Saves the updated model.

#### Challenge 4: REST API

In order to run the API, the file *XTream_API.py* should be executed.

The API includes:
+ A Health Check homepage to check that the api is running properly.
+ A predict method/call that can handle single queries or a list of queries.
To test the prediction of the api, two request programs were considered. A base one (*request.py*) which only generates a single request to the API. And a more complete case (*request_v2.py*) which generates two queries to the api, a single one and a list type one.
The API also includes a logger for monitoring purposes and a modified version of the pipeline generated for the previous challenge, so it's adapted to a prediction kind input.
+ A train method/call which takes a single or a list of features set to train the XGBoost model. An example of train request is provided in the *trainrequest.py* file. The data/logging from the training evaluation and data is stored in a log file dedicated to this call, keeping it apart from the other queries to the API (the prediction ones).

