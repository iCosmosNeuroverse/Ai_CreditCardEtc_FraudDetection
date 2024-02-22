#Author: God Bennett


#Original Code (55 lines) by Manuel on Kaggle: https://www.kaggle.com/manoloesparta/neural-network-accuracy-99-93


#Modified Code (402 lines) by God Bennett 

 
#God Bennett  wrote code to:
#1. Perform individual testing of new transactions, (aka "online inferencing"), to simulate real-time processing of single transactions.

#2. Perform crucial machine learning driven data imputation, since NCB transaction data may have missing values.

#3. Perform data visualization, including histograms, ... and confusion matrices that reflect accuracy of the model.

#4. Perform lecun_normal initialization, instead of default "uniform" found in original code, for accuracy improvement.  See pool of initializers at documentation site: https://keras.io/initializers/


# Note by God Bennett : Uses python 3.6, Requires keras, matplotlib, pandas, sklearn, numpy and tensorflow installations to python.
# Epochs end at Epoch 5/5.


# import my libraries

import pandas as pd
import numpy as np

###############################################
#begin God Bennett 
import datetime
print('start time '+str(datetime.datetime.now()))
#end God Bennett 
###############################################
print("...training neural network on jncb dev transaction data" );

# read data

data = pd.read_csv('data/export_300k_v2_maskedNcbColumns.csv') #Edited by God Bennett  to point to local creditcard directory


###############################################God Bennett _function_0
#begin God Bennett _code impute missing values using BayesianRidge
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

#Use BayesianRidge, because it seems to tend to report the best accuracies as seen in documentation: https://scikit-learn.org/stable/auto_examples/impute/plot_iterative_imputer_variants_comparison.html
def getImputation (value):
       return pd.DataFrame(IterativeImputer(BayesianRidge()).fit_transform(value))

#save original file data frame
data_default = data

#create imputed data frame
data = getImputation(data)

#establish imputed data frame in terms of default columns and indices from file
data.columns = data_default.columns
data.index = data_default.index
###############################################end God Bennett _function_0


###############################################God Bennett _function_1
#Begin God Bennett _code to render pie visualization of fraudulent transactions vs non-fraudlent transactions wrt training data
import matplotlib.pyplot as plt 
def plotFraudVsNonFraudVisualization_Histogram_WrtDataset ( ):
       fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))

       fig1.text( .13, .94, "Visualization of fraud vs non fraud, wrt dataset", fontsize=13)

       fig1.text( .50, .94, "(Please close Window to begin neural network training)", fontsize=11, color='green')

       bins = 40

       ax1.hist(data.AMOUNT[data.CARDFLAGFRAUD == 1.0], bins = bins, density = True, alpha = 0.75, color = 'red')
       ax1.set_title(('Frauds: ~ ' + str(len(data[data.CARDFLAGFRAUD == 1.0]))), color='red')

       ax2.hist(data.AMOUNT[data.CARDFLAGFRAUD == 0.0], bins = bins, density = True, alpha = 0.5, color = 'blue')
       ax2.set_title(('Non Frauds: ~ ' + str(len(data[data.CARDFLAGFRAUD == 0.0]))), color='blue')
       

       plt.xlabel('Amount')
       plt.ylabel('% of Transactions')
       plt.yscale('log')
       plt.show()

plotFraudVsNonFraudVisualization_Histogram_WrtDataset ( )
#End God Bennett _code to render pie visualization of fraudulent transactions vs non-fraudlent transactions wrt training data
###############################################end God Bennett _function_1


###############################################begin God Bennett _code to specify dimensions before training
inputDimensions = 65 ##Corresponds to number of non-class columns in ncb input data file.
##end God Bennett  code to specify dimension before training
epochCount = 15 ##default: 5
batchCount = 10 ##default: 10
###############################################begin original_code 
# define features and target

X = data.drop('CARDFLAGFRAUD',axis=1)
y = data['CARDFLAGFRAUD']




# scale my features

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)



# divide in train and test data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

# create neural network

# import more libraries for neural network

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout



# create the neural network

#God Bennett _edit: initializers changed from uniform to lecun_normal alternative kernel_initializer from pool of initializers at documentation site: https://keras.io/initializers/
from keras import initializers

clas = Sequential()
clas.add(Dense(units=14,kernel_initializer=initializers.lecun_normal(seed=None),activation='relu',input_dim=inputDimensions))
clas.add(Dropout(rate=0.1))
clas.add(Dense(units=14,kernel_initializer=initializers.lecun_normal(seed=None),activation='relu',input_dim=inputDimensions))
clas.add(Dropout(rate=0.1))
clas.add(Dense(units=1,kernel_initializer='random_uniform',activation='sigmoid'))
clas.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#end God Bennett  modification.

"""
default_neural network setup
clas = Sequential()
clas.add(Dense(units=9,kernel_initializer='uniform',activation='relu',input_dim=inputDimensions))
clas.add(Dropout(rate=0.1))
clas.add(Dense(units=9,kernel_initializer='uniform',activation='relu',input_dim=inputDimensions))
clas.add(Dropout(rate=0.1))
clas.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
clas.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
"""

# train neural netork

clas.fit(X_train,y_train,batch_size=batchCount,epochs=epochCount,verbose=2)

# evalute model

scores = clas.evaluate(X_test,y_test,verbose=0)

# print results

print('Error',scores[0],'\nAccuracy',scores[1])
###############################################end original_code



###############################################God Bennett _function_2
#Begin God Bennett _code to render pie visualization of fraudulent transactions vs non-fraudlent transactions wrt trained neural network
predictions_tr = clas.predict_classes ( X_train, verbose=0)
predictions_te = clas.predict_classes ( X_test, verbose=0)
predictions = np.append(predictions_tr, predictions_te)

#see all predictions in excel file. (Observed in excel is ~462 detections of fraud, and the rest non-fraud (284807-  ~462). User needs to filter excel output on Prediction column.)
predictions = pd.DataFrame({'Prediction':predictions});
predictions.to_csv("predictions.csv") #store predictions

"""
#includes ground truth data, that is what is predicted versus what actually is true of the training data
#may not make sense, because the ground truths may be ordered differently from the prediction set.
groundTruths = np.array(data['CARDFLAGFRAUD'])
groundTruthDataFrame = pd.DataFrame({'Prediction':groundTruths})
predictionsPlusGroundTruth = pd.concat([predictions,groundTruthDataFrame], axis=1)
predictionsPlusGroundTruth.to_csv("predictionsPlusGroundTruth.csv") #store predictions
"""

#generate visualization of predictions, in the form of a pie chart
# "predictions[predictions.Prediction == 1]" gets all predictions that positively detect fraud aka predictions that signify fraud
def plotFraudVsNonFraudVisualization_PieChart_WrtTrainedNeuralNetwork():
    numberOfFrauds = len(predictions[predictions.Prediction > 0.3]) 

    numberOfNonFrauds = len(predictions[predictions.Prediction < 0.1]) 

    numberOfTotalTransactions = numberOfFrauds + numberOfNonFrauds
                     
    labels = ('Fraudulent', 'Non-Fraudulent')

    sizes = [numberOfFrauds, numberOfNonFrauds]
	
    explode = (0.1, 0)  # only explode the 1st slice
    
    colors = ["red", "green"] #red for fraudulent transactions, and green for normal transactions

    fig1, ax1 = plt.subplots()

    plt.suptitle(str(numberOfFrauds) + ' frauds detected out of ' + str ( numberOfTotalTransactions ) + " transactions ", fontsize=16)

    plt.title("(Please close Window to see next figure)", fontsize=14, color='blue')
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%19.1f%%', shadow=True, startangle=25)
    
    ax1.axis('equal') 

    plt.show()
plotFraudVsNonFraudVisualization_PieChart_WrtTrainedNeuralNetwork ( )
#End God Bennett _code to render pie visualization of fraudulent transactions vs non-fraudlent transactions wrt trained neural network
###############################################end God Bennett _function_2



###############################################God Bennett _function_3
#God Bennett  function 3 to add online inference or real time test capability to the model.
"""
This function does a prediction from test data that has been converted/transformed to a separate form that
differs from their form in the excel input file "creditcard.csv".
The function shows expected aka actual outcome versus predicted outcome.
"""
#arguments i = index of training data record.
#Note: The target training record is located at the excel line number - 2, based on physical count of items in dataset creditcard.csv.
def doOnlineInference(i):
	testX, testy = X[i], y[i]
	testX = testX.reshape(1, inputDimensions)
	yhat = clas.predict(testX, batch_size=1)
	print('------>Training Record' + str (np.array(sc.inverse_transform(X[i]), dtype="O"))) 
	print('>>>>>>>>>>>>>>')
	print('------------->Expected=%.1f, Predicted=%.1f' % (testy, yhat))
##############################################end God Bennett _function_3



	
###############################################God Bennett _function_4
####God Bennett  function 4 to add online inference or real time test capability to the model.
"""
This function does a prediction from test data that has been taken directly from excel input file "creditcard.csv".
(before which it is sorrounded by array braces, and written with commas)
Function is designed to simply return the predicted outcome. 
"""

#record has been preorganized by copying excel row, replacing spaces with commas, then putting in the form of np array.
#Note that "[[<value>]]" works as well without the np.array.
#Expected result from this record is fraudulent transaction, or 0.1+
examplePreOrganizedRecord=np.array([[596721,0,98,0,0,99,0,1,1,111012.41,-308492.99,1500000,-419505.4,-12339.72,-12339.72,0,0,-111012.41,-111012.41,0,0,0,0,0,0,0,0,0,0,0,0,293144,1174,13,59369066,13766,13256,938,502,198,201,867129,820134,4.51E+15,100,180,-6000,0,388,388,6000,5541,163153249,668987,707535,98,141,159,850,593,1,2048,1,1.15E+18,0]], dtype="O")[0:1]

#The example record above is line 180810 from creditcard.csv (aka entry 180808 based on physical count of items in creditcard.csv file)
def doOnlineInferenceOnPreOrganizedRecord ( newTransactionRecord ):
	#create copy of input set X
	X_copy = data.drop('CARDFLAGFRAUD',axis=1)
	#grow copy by new newTransactionRecord
	X_grown = np.append ( X_copy, newTransactionRecord, axis=0)
	#refit copy (aka X_grown) with respect to rest of input set
	X_grown = sc.fit_transform ( X_grown )
	#confirm that the new record at end of grown training set is the one supplied as input, i.e. whether function is
	#predicting on new record
	print('------>Training Record at end of training set = ' + str (np.array(sc.inverse_transform(X_grown[len(X_grown)-1]), dtype="O"))) 
	print('>>>>>>>>>>>>>>')
	#now that new record is added to end of list, the last entry of X_grown will be the new record, however in a format
	#ready for ai prediction processing.
	#the last entry is reshaped into a 1x(inputDimensions) object, which class.<predict> func accepts.
	#<predict> naturally returns a 2 dimensional array, of a single item, hence why I wrote the [0][0] indexing below.
	return clas.predict (X_grown[len(X_grown)-1].reshape (1,inputDimensions), batch_size=1) [0][0]

"""
This function does a prediction from test data that has been taken directly from excel input file "creditcard.csv". 
Function is designed to simply return the predicted outcome.

"""

#record has not been pre-organized from excel file by hand, except for the quotes at the beggining and end. Preprocessing is small, so
#fair to call it raw. Expected result from this record is abnormal transaction, or 0.1000+....
exampleRawRecord="177027	0	364	0	0	99	0	1	1	359.93	-449.47	17400	-792.84	-17.98	-9.7	0	8.28	-351.65	-351.65	0	0	0	0	0	0	0	0	0	0	0	0	39511	1160	15	59398847	13766	13386	13374	13370	13378	201	86750	815867	5.43E+15	100	80.17	-80.17	0	840	840	80.17	5542	162722199	50638	653386	104	141	159	156	604	1	2048	1	1.15E+18	0"

import re #use regular expression to convert input excel row to form with commas, for use in function.
import decimal #for use of converting exponent result to easily readable format
def doOnlineInferenceOnRawRecord ( newTransactionRecordString ):
       #remove spaces, add commas
       newTransactionRecordString_commaFormat = re.sub("\s+", ",", newTransactionRecordString.strip())
       #convert string to array for use in single observation prediction
       newTransactionRecord = [[float(item) for item in newTransactionRecordString_commaFormat.split ( "," )]]
       #create copy of input set X
       X_copy = data.drop('CARDFLAGFRAUD',axis=1)
       #grow copy by new newTransactionRecord
       X_grown = np.append ( X_copy, newTransactionRecord, axis=0)
       #refit copy (aka X_grown) with respect to rest of input set
       X_grown = sc.fit_transform ( X_grown )
       #confirm that the new record at end of grown training set is the one supplied as input, i.e. whether function is
       #predicting on new record
       print('------>Training Record at end of training set = ' + str (np.array(sc.inverse_transform(X_grown[len(X_grown)-1]), dtype="O"))) 
       print('>>>>>>>>>>>>>>')
       #now that new record is added to end of list, the last entry of X_grown will be the new record, however in a format
       #ready for ai prediction processing.
       #the last entry is reshaped into a 1x(inputDimensions) object, which class.<predict> func accepts.
       #<predict> naturally returns a 2 dimensional array, of a single item, hence why I wrote the [0][0] indexing below.
       #finally, float and round the resulting string value that emerges from "remove_exponent" function.
       return round(float(remove_exponent(clas.predict (X_grown[len(X_grown)-1].reshape (1,inputDimensions), batch_size=1) [0][0])),4)

###############################################God Bennett _function_4

       
       
###############################################God Bennett _function_5
####God Bennett  function 5 to produce confusion matrix based on the neural network output.
"""
~~forward slash aka diagnoal shows true positives and true negtives, all other cells show wrong predictions. 

~~matrix title normally shows ground truth

~~note for skewed data sets, 99.8% accuracy can be achieved even if model describes all to be truly normal,
while failing to detect any of the small actual abnormalities. Because of this it is important to do online inference against the
input/dataset, to ensure that the model has learnt the task. Tests with the 2 new onlineInference functions above prove that the
model learns well.

~~confusion matrices are not very good in skewed settings, although the code below is made for manuel's nn detection code.
"""
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def showConfusionMatrix():
       y_pred=clas.predict_classes(X_test)
       cm = confusion_matrix(y_test,y_pred)
       print('Confusion matrix: \n',cm)
       print('Classification report: \n',classification_report(y_test,y_pred))
       sns.heatmap(cm, annot=True, fmt='.0f', cmap='cividis_r')
       plt.show()
###############################################God Bennett _function_5

"""
###############################################God Bennett _function_6
#Begin God Bennett _code to render tSNE visualization of fraudulent transactions vs non-fraudlent transactions wrt trained neural network
#Note this code works, but it takes a 3 minutes to render. The pie char should give a good representation of the outcome, albeit.

from sklearn.manifold import TSNE

def plotFraudVsNonFraudVisualization_TDistributedStochasticNeighbourEmbedding_WrtTrainedNeuralNetwork ( ):
	Fraud = predictions[predictions.Prediction == 1]
	Normal = predictions[predictions.Prediction == 0]
	
	#Set df_used to the fraudulent transactions' dataset.
	df_used = Fraud

	#Add 10,000 normal transactions to df_used. (all 284807 transactions will take too long to print)
	df_used = pd.concat([df_used, Normal.sample(n = 10000)], axis = 0)


	#Scale features to improve the training ability of TSNE.
	standard_scaler = StandardScaler()
	df_used_std = standard_scaler.fit_transform(df_used)

	#Set y_used equal to the target values.
	y_used = df_used.ix[:,-1].values


	tsne = TSNE(n_components=2, random_state=0)
	x_test_2d_used = tsne.fit_transform(df_used_std)


	color_map = {1:'red', 0:'blue'}
	plt.figure()
	for idx, cl in enumerate(np.unique(y_used)):
		plt.scatter(x=x_test_2d_used[y_used==cl,0], 
					y=x_test_2d_used[y_used==cl,1], 
					c=color_map[idx], 
					label=cl)
	plt.xlabel('X in t-SNE')
	plt.ylabel('Y in t-SNE')
	plt.legend(loc='upper left')
	plt.title('t-SNE visualization of test data')
	plt.show()
plotFraudVsNonFraudVisualization_TDistributedStochasticNeighbourEmbedding_WrtTrainedNeuralNetwork ( )
#End God Bennett _code to render tSNE visualization of fraudulent transactions vs non-fraudlent transactions wrt trained neural network
###############################################end God Bennett _function_6
"""


###############################################God Bennett _function_7 save and load model
from keras.models import load_model

def saveModel ( directory ):
       clas.save( directory )

def loadModel ( directory ):
       clas.load_model ( directory )

###############################################end God Bennett _function_7


"""function used to remove exponent from value, to a more readable format
Source: https://stackoverflow.com/questions/9195800/converting-exponential-to-float

example value: prediction=2.34344e-5
output: 0.00002...
"""
def remove_exponent(value):
    """
       >>>(Decimal('5E+3'))
       Decimal('5000.00000000')
    """
    decimal_places = 8
    max_digits = 16

    if isinstance(value, decimal.Decimal):
        context = decimal.getcontext().copy()
        context.prec = max_digits
        return "{0:f}".format(value.quantize(decimal.Decimal(".1") ** decimal_places, context=context))
    else:
        return "%.*f" % (decimal_places, value)
       

###############################################
#begin God Bennett _code to report end time
print('end time '+str(datetime.datetime.now()))
#end God Bennett _code to report end time
###############################################




