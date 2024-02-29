#Author God Bennett. 
#Simple test to load saved model.
#To do further testing one may simply copy the onlineInference functions doOnlineInferenceOnRawRecord
from keras.models import load_model
import pandas as pd
import numpy as np


inputDimensions = 65 ##Corresponds to number of non-class columns in ncb input data file.

# read dataset
data = pd.read_csv('data/training_data.csv') 

# create model based on saved weights
model = load_model('data/95.66%_saved_BANK_neural_network_weights.h5')

# split data into labels and training 
X = data.drop('FRAUDFLAG',axis=1)
y = data['FRAUDFLAG']


# scale my features
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)


# divide in train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)





#show confusion matrix to confirm model accuracy
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
def showConfusionMatrix():
       y_pred=model.predict_classes(X_test)
       cm = confusion_matrix(y_test,y_pred)
       print('Confusion matrix: \n',cm)
       print('Classification report: \n',classification_report(y_test,y_pred))
       sns.heatmap(cm, annot=True, fmt='.0f', cmap='cividis_r')
       plt.show()


#online inference functions for precise testing
exampleRawRecord = "177027	0	364	0	0	99	0	1	1	359.93	-449.47	17400	-792.84	-17.98	-9.7	0	8.28	-351.65	-351.65	0	0	0	0	0	0	0	0	0	0	0	0	39511	1160	15	59398847	13766	13386	13374	13370	13378	201	86750	815867	5.43E+15	100	80.17	-80.17	0	840	840	80.17	5542	162722199	50638	653386	104	141	159	156	604	1	2048	1	1.15E+18	0";

###############################################God Bennett. _function_4
####God Bennett. function 4 to add online inference or real time test capability to the model.
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
	X_copy = data.drop('FRAUDFLAG',axis=1)
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
	return model.predict (X_grown[len(X_grown)-1].reshape (1,inputDimensions), batch_size=1) [0][0]

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
       X_copy = data.drop('FRAUDFLAG',axis=1)
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
       return round(float(remove_exponent(model.predict (X_grown[len(X_grown)-1].reshape (1,inputDimensions), batch_size=1) [0][0])),4)

###############################################God Bennett. (My name was legally changed from Jordan Bennett)_function_4

#Example raw record process
#doOnlineInferenceOnRawRecord(exampleRawRecord)
#doOnlineInferenceOnPreOrganizedRecord(examplePreOrganizedRecord)


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


def stn(string):
    # Split the string by whitespace
    string_values = string.split()

    # Convert each string value to a numeric type
    numeric_values = [float(value) if '.' in value else int(value) for value in string_values]

    return numeric_values



##################################
# read unseen dataset, do predictions on resulting variable, where each index corresponds to each row
uData = pd.read_csv('data/unseen_transactions_csv.csv')
uX = uData.drop('FRAUDFLAG',axis=1)
#uX = sc.fit_transform(uX) #issue with this is that, it seems to destroy the feature set. Seems to render nn prediction as all 0s
#alternative is to do imputation on the non fitted uX data, which is possible although it's not directly iterable without being fitted, then
#conver the resulting imputed array as a string, followed by consumption by RawRecord inference "doOnlineInferenceOnRawRecord"

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#Removes "nan" numbers replaces with appropriate mean of surrounding values

# Create an imputer object with strategy 'mean'
imputer = IterativeImputer()

# Fit the imputer to the data
imputer.fit(uX)

# Transform the data
iuX = imputer.transform(uX)


##################################
# funct form read unseen dataset, do predictions on resulting variable, where each index corresponds to each row
# expects field CARDFLAGFRAUD as fraud column. Can be dynamic, but left as a standard for now.
# eg: dat=PROCESS_DATA( "data/unseen_transactions_csv.csv")
def PROCESS_DATA(transaction_file_loc):
       uData = pd.read_csv(transaction_file_loc)
       uX = uData.drop('FRAUDFLAG',axis=1)
       #uX = sc.fit_transform(uX) #issue with this is that, it seems to destroy the feature set. Seems to render nn prediction as all 0s
       #alternative is to do imputation on the non fitted uX data, which is possible although it's not directly iterable without being fitted, then
       #conver the resulting imputed array as a string, followed by consumption by RawRecord inference "doOnlineInferenceOnRawRecord"


       #Removes "nan" numbers replaces with appropriate mean of surrounding values

       # Create an imputer object with strategy 'mean'
       imputer = IterativeImputer()

       # Fit the imputer to the data
       imputer.fit(uX)

       # Transform the data
       iuX = imputer.transform(uX)

       return iuX


##converts imputed excel data
##eg iuX[0] is  valid input for consumable function
#Input to doOnlineInferenceOnRawRecord
#getC - get Ai consumable
#eg from unseen_data csv file
#doOnlineInferenceOnRawRecord(getCn(iuX[0]))   --- expected close to 0.1 to 1, fraudulent
#doOnlineInferenceOnRawRecord(getCn(iuX[1]))   --- expected close to 0.1 to 1, fraudulent
#doOnlineInferenceOnRawRecord(getCn(iuX[2]))   --- expected close to 0, non-fraudulent
#doOnlineInferenceOnRawRecord(getCn(iuX[3]))   --- expected close to 0, non-fraudulent
#doOnlineInferenceOnRawRecord(getCn(iuX[4]))   --- expected close to 0, non-fraudulent
def getCn ( array ):
       return ' '.join(str(x) for x in array)

#getC - get Ai consumable
#Input to "doOnlineInferenceOnPreOrganizedRecord"
def ________getCn ( array ):
       return  np.array([array], dtype="O")[0:1]
