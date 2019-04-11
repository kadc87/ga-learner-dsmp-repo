# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 



# code starts here
bank = pd.read_csv(path)

categorical_var = bank.select_dtypes(include = 'object')
print(categorical_var)

numerical_var = bank.select_dtypes(include = 'number')
print(numerical_var)
# code ends here


# --------------
# code starts here
banks = bank.drop(columns = ['Loan_ID'], axis = 1)
print(banks.isnull().sum())

bank_mode = banks.mode()
print(bank_mode)

#banks = banks.fillna(bank_mode.iloc[0], inplace = True)
banks=banks.replace(np.nan, bank_mode)

print(banks)
#code ends here


# --------------
# Code starts here

avg_loan_amount = pd.pivot_table(banks, index = ['Gender', 'Married', 'Self_Employed'], values = 'LoanAmount', aggfunc = 'mean')


# code ends here



# --------------
# code starts here
loan_approved_se = banks[(banks['Self_Employed'] == 'Yes') & (banks['Loan_Status'] == 'Y')].shape[0]
print(loan_approved_se)



loan_approved_nse = banks[(banks['Self_Employed'] == 'No') & (banks['Loan_Status'] == 'Y')].shape[0]
print(loan_approved_nse)

percentage_se = (loan_approved_se * 100) / 614

percentage_nse = (loan_approved_nse * 100) / 614

# code ends here


# --------------
# code starts here

loan_term = banks['Loan_Amount_Term'].apply(lambda x: x/12)
big_loan_term = banks[loan_term >=25].shape[0]



# code ends here


# --------------
# code starts here
loan_groupby = bank.groupby(['Loan_Status'])[['ApplicantIncome', 'Credit_History']]

mean_values = loan_groupby.mean()

print(mean_values)


# code ends here


