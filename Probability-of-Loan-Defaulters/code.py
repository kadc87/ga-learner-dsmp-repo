# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#code starts here

df = pd.read_csv(path)
total = df.shape[0]
pa_num = df[df['fico'] > 700].shape[0]
p_a =  pa_num / total
print(p_a)

pb_num = df[df['purpose'] == 'debt_consolidation'].shape[0]
df1= df[df['purpose'] == 'debt_consolidation']
p_b = pb_num / total
print(p_b)

pab=(df[(df['purpose']=='debt_consolidation') & (df['fico']>700)]).shape[0]
p_a_b=pab/total
print(p_a_b)

result=p_a_b==p_a
print(result)

# code ends here


# --------------
# code starts here

total = df.shape[0]

problp_num = df[df['paid.back.loan'] == 'Yes'].shape[0]
prob_lp = problp_num / total
print(prob_lp)

probcs_num = df[df['credit.policy'] == 'Yes'].shape[0]
prob_cs = probcs_num / total
print(prob_cs)

new_df = df[df['paid.back.loan'] == 'Yes']

probpdcs_num = (df[(df['paid.back.loan'] == 'Yes') & (df['credit.policy'] == 'Yes')]).shape[0]
#prob_pd_cs = probpdcs_num / total
#print(prob_pd_cs)
prob_pd_cs = 0.8323182100683655
bayes = (prob_pd_cs * prob_lp) / prob_cs
print(bayes)
# code ends here


# --------------
# code starts here
plt.bar(df['purpose'], 0.5)
df1 = df[df['paid.back.loan'] == 'No']
plt.bar(df1['purpose'], 0.5)
# code ends here


# --------------
# code starts here
inst_median = df['installment'].median()
inst_mean = df['installment'].mean()
plt.hist(df['installment'])
plt.show()
plt.hist(df['log.annual.inc'])
plt.show()
# code ends here


