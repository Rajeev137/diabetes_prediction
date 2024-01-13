import numpy as np
import pickle
#loading the saved model
loaded_model = pickle.load(open('/Users/rajeevsharma/Downloads/project/Diabetes prediction/trained_model.sav', 'rb'))
input_data = (13,145,82,19,110,22.2,0.245,57)

#changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting only for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')