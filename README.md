About
==============

# JNCB ~ Jamaica Neural Network Credit Card Bank - Ai System

Using GUI "LAUNCH_JNCB_AI_FRAUD_ANALYSER.bat"
======


![alt text](GUI_SCREEN.png "Title Text")

Run batch launcher, use file icon to load transactions and generate predictions, and the plane icon to save predictions.
Big X Icon toggling, clears predictions. Predictions are saved to predictions.txt file.

You can load the following files for testing:
* data/_unseen_transactions_sample_1.csv (Unseen sample set 1, testing model generalization beyond training set)
* data/_unseen_transactions_sample_2.csv (Unseen sample set 2, testing model generalization beyond training set)
* data/_seen_unseen_mix_transaction_sample_1.csv (A mixture of seen and unseen samples unseen by ai in training, to test both memory and generalization)


Using PythonShelll "god_ai_credit_card_fraud_detection_load_pretrained.py"
======

![alt text]( https://github.com/g0dEngineer/Ai_CreditCardEtc_FraudDetection/blob/main/data/prediction%20on%20a%20single%20entry.png "Title Text")

# JNCB = Jamaica Neural Network Credit Card Bank - Ai System

This project seeks to supplement current Fraud Guard method with neural network based code/method that I prepared/wrote, for the goal/purpose of [improved fraud detection by > 50%](http://news.mit.edu/2018/machine-learning-financial-credit-card-fraud-0920).

The current method of many Jamaican banks' fraud detection though well structured, could be enhanced/supplemented by automatic means of credit card fraud detection, namely via the use of artificial intelligence. Artificial neural networks are quite general; there are neural networks that enable self-driving cars, while the same neural network types also enable disease diagnosis, language translation etc. 

The history of Ai has seen where expert systems with years of hand crafted rules/knowledge by experts, are enhanced considerably by automated systems that learn how to build rules. In some cases, hybrid systems have been constructed that make use of both learning ai, and rule-based ai.

Nowadays, most modern systems, including ones that [other banks like COK are using](https://www.fintechfutures.com/2018/10/smart-solution-gains-new-core-banking-tech-client-in-jamaica-cokcu/), make great use of the second wave of ai, namely statistical learning, or machine learning. The goal is to utilize the second wave of Ai, in tandem with current fraud guard systems, to greatly increase detection of frauds, while reducing the number of false positive detection.

As the bank gets more complex, we’ll reasonably need to use neural networks or some similar method to do fraud detection, because it is already hard for rule builders to keep up with fraud patterns with the current non-neural network based method, and neural network or similar methods capture more frauds, and minimizes the amount of transactions that are falsely detected as fraudulent, [by up to 54%;](http://news.mit.edu/2018/machine-learning-financial-credit-card-fraud-0920) potentially saving up to 37 million jmd.

**Tsys note**: In fact, [Tsys' latest Fico Fraud Management system utilizes artificial neural networks.](https://www.tsys.com/Assets/TSYS/downloads/br_tsys-fraud-mitigation.pdf)!

    *  •	The Fico system seems different from the apparently non-neural network based FraudGuard system that some banks currently employs. 
    I propose that some banks shall either seek to acquire Fico Fraud Management licensing, or integrate a neural net based pipeline, using the credit card artificial neural network code prepared by myself that this repository refers to or similar.

Quick explanation of how this neural network works
==============
1. Take some transaction data, in the form of a database row for each transaction from sample masked bank data.
2. Thousands of these transaction rows are labelled as 0 or 1 (not fraudulent or fraudulent) depending on fraudguard history. (Assumption: Data had been updated and properly labelled by Fraud Squad)
3. In training, expose the neural network to these labelled transactions, as the neural network learns what fraud and non-fraudulent looks like.
4. In testing aka inference (simulating when a single customer does a payment etc); expose neural network to unlabelled transactions. 
    * Neural network then produces a float value between 0 and 1 for each unlabelled transaction, where value closer to 1 indicates prediction of fraud, while closer to 0 indicates non-fraud.

See [this seminar lead by G.Bennett Bennett, concerning basic artificial neural networks](https://github.com/g0dEngineer/Ai_CreditCardEtc_FraudDetection/tree/main/GOD_AI_Seminar.pdf).



Original Code (55 lines)
==============
by Manuel on Kaggle: https://www.kaggle.com/manoloesparta/neural-network-accuracy-99-93



Modified Code (402 lines)
==============
by God Bennett. 

* This code achieves an accuracy of ~95% on Bank masked dev transaction data.

 
Code Modification Description
==============
G.Bennett wrote code to:

1. Perform individual testing of new transactions, (aka "online inferencing"), to simulate real-time processing of single transactions.
    * Note that online inferencing here, is not related to the internet. Online means per transaction database record neural network processing.
2. Perform crucial machine learning driven data imputation, since the bank transaction data tends to have missing values.
3. Perform data visualization, including histograms, ... and confusion matrices that reflect accuracy of the model.
4. Perform lecun_normal initialization, instead of default "uniform" found in original code, for accuracy improvement, as the original code initializers did not work well. See pool of initializers at documentation site: https://keras.io/initializers/


Requirements
==============
Uses python 3.6.3, Requires tensorflow 2.6.2, keras, matplotlib, pandas, sklearn, numpy installations to python.




Installation (With Admin Privileges)
==============

1. Download [my repository](https://github.com/g0dEngineer/Ai_CreditCardEtc_FraudDetection/tree/main).
2. Download  BANK [training_data.csv](https://drive.google.com/file/d/1jWUotyygDzeSN2uVtVjO9S2RX6OVqBDO/view?usp=sharing), copy to data/ folder.
3. Install all python modules seen in [Requirements](https://github.com/g0dEngineer/Ai_CreditCardEtc_FraudDetection/tree/main#requirements). (Open Cmd, set python36 path as PATH, then run "python -m pip install each_req_name" in same cmd)
4. Ensure your Python36 path and not mine is the LAUNCH_JNCB_AI_FRAUD_ANALYSER.bat path.

Installation (Without Admin Privileges Workaround)
==============

1. Download [Embeddable zip x86-64 zip file](https://www.python.org/ftp/python/3.6.3/python-3.6.3-embed-amd64.zip). (Unzip, no admin required for install)
2. Download [my cached copy of the non-embeddable pip installer](https://drive.google.com/file/d/19UgUkwO2HA97Tn2JKk3jbvrcUp6qYI9i/view?usp=sharing) taken from 3.6.8 which works in 3.6.3 embedded. (This is needed to still install python modules seen in requirements section.)
   * Note that this little workaround is needed because Embeddable does not support pip installer by default.
   * I also provided my cached copy of pip from the admin based install, because other instructions tend to have outdated data that doesn't work.
4. In your extracted (1) folder, create dir and subdir: lib/site-packages.
   * Copy (2)'s extracted zip such that each folder is directly under site-packages.
5. In notepad, open (1)/python36._pth, and add line at top right above "python36.zip".
6. In cmd, install [Requirements](https://github.com/g0dEngineer/Ai_CreditCardEtc_FraudDetection/tree/main#requirements) as usual by

   a. 1st setting path to (1)
   
   b. using python -m pip <each_requirement> as normal.
   
   * Eg: python -m pip install tensorflow==2.6.2 (Install other requirements, though by default tensorflow install should add some of the requirements.
7. Copy "god_ai_credit_card_fraud_detection_load_pretrained.py" and entire data folder, paste to (4)'s root which is the parent python embedded folder beside python exe.

8. Run LAUNCH_AI_FRAUD_ANALYSER.bat to launch application.



Usage
==============
There are two ways to use this artificial neural network system:

1. Training and running.
    * **Train** the neural network (in about 3 minutes on an i7 cpu typical corporate laptop) on the csv sample masked bank transaction dev data. 
        * Note that most columns in the data\...training_data.csv_** file are masked, and known to SASS payments team or other Prime related members only. You can request this information.
    * Run the trained neural network, and make some predictions.
        * **Training** is done by simply running the python file, and awaiting the neural network's processing for about 15 epochs.
            * A successful run will look [like this image](https://github.com/g0dEngineer/Ai_CreditCardEtc_FraudDetection/tree/main/data/95.66__bank_data_successful_run.png).
        * While making a prediction, take note of the "FRAUDFLAG" column, which lables each transaction in dataset as 1 or 0 (where 1=fraud, 0=not fraudulent):
            * There are 299,999 records in dataset csv, and of those, the training process used the first 70%.
            * To really test the neural network, means to expose it to a record it didn't see in training.
            * Copy any record after cell 210,000 **(except for the last column which is the label)**. Records after 210,000 are outside of the "70%" training set.
            * Paste the copied record into python shell after neural network training ran, as parameter "newTransactionRecordString" from function [god_ai_credit_card_fraud_detection_train_from_scratch.py](https://github.com/g0dEngineer/Ai_CreditCardEtc_FraudDetection/tree/main/god_ai_credit_card_fraud_detection_train_from_scratch.py)/ doOnlineInferenceOnRawRecord ( newTransactionRecordString ).
            * Take note of the result.
                * Eg a: Record A223999 is labelled 0, and neural net prediction is accurate at 0.029. (Closer to 0) See **data/notFraudulent_onlineInferenceOnRecord_sample.png**.
				
                * Eg b: Record AY224046 is labelled 1, and [neural net prediction is accurate at 0.3381. (Closer to 1)  See **data/fraudulent_onlineInferenceOnRecord_sample.png**.
             

2. The quicker way: Running a pretrained model prepared by myself, that doesn't require training.
    * Run [god_ai_credit_card_fraud_detection_load_pretrained.py](https://github.com/g0dEngineer/Ai_CreditCardEtc_FraudDetection/tree/main/god_ai_credit_card_fraud_detection_load_pretrained.py) from the data folder in this repository.
    * Ensure [95.66%_saved_bank_neural_network_weights.h5](https://github.com/g0dEngineer/Ai_CreditCardEtc_FraudDetection/tree/main/data/95.66__saved_bank_neural_network_weights.h5) is in the data folder of this repository.
    * Ensure the files above are in the directory of the csv file from the "[Installation](https://github.com/g0dEngineer/Ai_CreditCardEtc_FraudDetection/tree/main#Installation)" step of this repository.
    * To make predictions, do the same steps done in the training phase above, **particularly, starting from** "Run the trained neural network...".


Model accuracy in terms of confusion matrix
=============

It is important to guage how well an ai model is doing, in ai research and implementation.
"Confusion matrix" is a standard term in machine learning/artificial intelligence, that describes in this case:
1. The number of true positives aka correctly made predictions of detected fraud.
2. The number of true negatives aka correctly made predictions that indicate no fraud.
3. The number of false positives aka incorrectly made predictions that falsely indicate fraud.
4. The number of false negatives aka incorrectly made predictions that falsely indicate no fraud.
5. Overall accuray, as a function of the 4 items above.
    * **Total transactions** = false positives + false negatives + true positives + true negatives = (3708 + 197 + 533 + 85562) = 90,000
    * **Total correct predictions** or ‘true items’ = (true positives + true negatives )/Total transactions = (533 + 85562)/90,000 = 0.95661111111 ~ 95% accuracy
    * **Total incorrect predictions** or MME ~ Mean Misclassification Error or ‘false items’ = (false positives + false negatives)/Total = (3708 + 197)/90000 = (3708 + 197)/90000 ~ 0.043% inaccuracy. This is better than the standard of around 0.1. [>= 0.1 may signify bad performance](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_4_PerformanceMetrics/ThresholdBased.html).


Invoking the function "showConfusionMatrix()" in [god_ai_credit_card_fraud_detection_train_from_scratch.py](https://github.com/g0dEngineer/Ai_CreditCardEtc_FraudDetection/tree/main/god_ai_credit_card_fraud_detection_train_from_scratch.py) reveals the confusion matrix:


![alt text](https://github.com/g0dEngineer/Ai_CreditCardEtc_FraudDetection/blob/main/data/95.66%25_bank_data_confusion_matrix.png "Title Text")




