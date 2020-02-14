# timeit

# Student Name : Paolo Schirru
# Cohort       : 1

# Note: You are only allowed to submit ONE final model for this assignment.


################################################################################
# Import Packages
################################################################################

# use this space for all of your package imports
import pandas as pd             # data science essentials
import matplotlib.pyplot as plt # essential graphical output
import seaborn as sns           # enhanced graphical output
from sklearn.model_selection import train_test_split #splits to training/test
import statsmodels.formula.api as smf #basic analysis
from sklearn.linear_model import LogisticRegression  # logistic regression
from sklearn.metrics import confusion_matrix         # confusion matrix
from sklearn.metrics import roc_auc_score            # auc score
from sklearn.neighbors import KNeighborsClassifier   # KNN for classification
from sklearn.tree import DecisionTreeClassifier      # classification trees
from sklearn.metrics import mean_squared_error as MSE# Mean squared error
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score  # cross validation
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler     # standard scaler
from sklearn.ensemble import GradientBoostingClassifier #Ensemble Modeling
from sklearn.ensemble import RandomForestClassifier    #Ensemble Modeling

# CART model packages
from sklearn.tree import DecisionTreeClassifier      # classification trees
from sklearn.tree import export_graphviz             # exports graphics
from sklearn.externals.six import StringIO           # saves objects in memory
from IPython.display import Image                    # displays on frontend
import pydotplus                                     # interprets dot objects

from sklearn.model_selection import GridSearchCV     # hyperparameter tuning
from sklearn.metrics import make_scorer              # customizable scorer






################################################################################
# Load Data
################################################################################

# use this space to load the original dataset
# MAKE SURE TO SAVE THE ORIGINAL FILE AS original_df
# Example: original_df = pd.read_excel('Apprentice Chef Dataset.xlsx')

original_df = pd.read_excel('Apprentice_Chef_Dataset.xlsx')

################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

# use this space for all of the feature engineering that is required for your
# final model

# if your final model requires dataset standardization, do this here as well

#Flagging the missing values in the Family Name
if original_df['FAMILY_NAME'].isnull().astype(int).sum() > 0:
    original_df['m_'+'FAMILY_NAME'] = original_df['FAMILY_NAME'].isnull().astype(int)


#this creates a new columns that it will be used in the model to understand
#if not having the family name has an impact on cross-sell success

#Creating dummy variables for the email domains
#The dummy variables are used in the model to understand if the domain of the
#email that is given has an impact on the promotion choice


# placeholder list
placeholder_lst = []

# looping over each email address
for index, col in original_df.iterrows():

    # splitting email domain at '@'
    split_email = original_df.loc[index, 'EMAIL'].split(sep = '@')

    # appending placeholder_lst with the results
    placeholder_lst.append(split_email)

# converting placeholder_lst into a DataFrame
email_df = pd.DataFrame(placeholder_lst)
del email_df[0]

# email domain types
personal_email_ = ['@gmail.com', '@yahoo.com','@protonmail.com']
profess_email   = ['@mmm.com','@amex.com','@apple.com','@boeing.com',
                   '@caterpillar.com','@chevron.com', '@cisco.com',
                   '@cocacola.com','@disney.com','@dupont.com','@exxon.com',
                   '@ge.org', '@goldmansacs.com', '@homedepot.com', '@ibm.com',
                   '@intel.com', '@jnj.com', '@jpmorgan.com', '@mcdonalds.com',
                   '@merck.com', '@microsoft.com', '@nike.com', '@pfizer.com',
                   '@pg.com', '@travelers.com', '@unitedtech.com',
                   '@unitedhealth.com', '@verizon.com', '@visa.com',
                   '@walmart.com']
junk_email      = ['@me.com','@aol.com','@live.com','@msn.com',
                  '@passport.com','@hotmail.com']

#domain list
domains_lst = []

# looping to group observations by domain type
for domain in email_df[1]:

    if '@' + domain in personal_email_:
        domains_lst.append('personal')
    elif  '@' + domain in profess_email:
        domains_lst.append('professional')
    elif  '@' + domain in junk_email:
        domains_lst.append('junk')
    else:
        print("There was an error")

# concatenating with original DataFrame
original_df['email_dom'] = pd.Series(domains_lst)

#Creating dummies for Emails
dummies = pd.get_dummies(original_df['email_dom'])

#Dropping EMAIL and email_dom from original_df
original_df = original_df.drop('EMAIL', axis = 1)
original_df = original_df.drop('email_dom', axis = 1)

#Joining the dummies to original_df
original_df = original_df.join([dummies])


# setting outlier thresholds
#These points are considered outliers in the data
#We are telling the model to consider these thresholds and
#to behave differently


REVENUE_HI                     = 3500
TOTAL_MEALS_ORDERED_HI         = 220
UNIQUE_MEALS_PURCH_HI          = 9
CONTACTS_W_CUSTOMER_SERVICE_LO = 3
CONTACTS_W_CUSTOMER_SERVICE_HI = 12.5
PRODUCT_CATEGORIES_VIEWED_LO   = 1
PRODUCT_CATEGORIES_VIEWED_HI   = 10
AVG_TIME_PER_SITE_VISIT_HI     = 250
CANCELLATIONS_BEFORE_NOON_HI   = 7
CANCELLATIONS_AFTER_NOON_HI    = 2.0
MOBILE_LOGINS_LO               = 5
MOBILE_LOGINS_HI               = 6
PC_LOGINS_LO                   = 1
PC_LOGINS_HI                   = 2
WEEKLY_PLAN_HI                 = 50
EARLY_DELIVERIES_HI            = 5
LATE_DELIVERIES_HI             = 10
FOLLOWED_RECOMMENDATIONS_PCT_HI= 90
AVG_PREP_VID_TIME_HI           = 280
LARGEST_ORDER_SIZE_LO          = 2
LARGEST_ORDER_SIZE_HI          = 8
MEDIAN_MEAL_RATING_LO          = 2
MEDIAN_MEAL_RATING_HI          = 4
AVG_CLICKS_PER_VISIT_LO        = 7.5
AVG_CLICKS_PER_VISIT_HI        = 17.5
TOTAL_PHOTOS_VIEWED_HI         = 450

##############################################################################
## Feature Engineering (outlier thresholds)                                 ##
##############################################################################

# developing features (columns) for outliers
#REVENUE
original_df['out_rev'] = 0
condition_hi = original_df.loc[0:,'out_rev'][original_df['REVENUE'] > REVENUE_HI]

original_df['out_rev'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)
#TOTAL MEALS ORDERED
original_df['out_total_meals'] = 0
condition_hi = original_df.loc[0:,'out_total_meals'][original_df['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_HI]

original_df['out_total_meals'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)
#UNIQUE_MEALS_PURCHASED
original_df['out_unique_meals'] = 0
condition_hi = original_df.loc[0:,'out_unique_meals'][original_df['UNIQUE_MEALS_PURCH'] > UNIQUE_MEALS_PURCH_HI]

original_df['out_unique_meals'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)
#CONTACTS_W_CUSTOMER_SERVICE
original_df['out_contacts_c_s'] = 0
condition_hi = original_df.loc[0:,'out_contacts_c_s'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_HI]
condition_lo = original_df.loc[0:,'out_contacts_c_s'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] < CONTACTS_W_CUSTOMER_SERVICE_LO]

original_df['out_contacts_c_s'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

original_df['out_contacts_c_s'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

#PRODUCT_CATEGORIES_VIEWED
original_df['out_prod_cat_view'] = 0
condition_hi = original_df.loc[0:,'out_prod_cat_view'][original_df['PRODUCT_CATEGORIES_VIEWED'] > PRODUCT_CATEGORIES_VIEWED_HI]
condition_lo = original_df.loc[0:,'out_prod_cat_view'][original_df['PRODUCT_CATEGORIES_VIEWED'] < PRODUCT_CATEGORIES_VIEWED_LO]

original_df['out_prod_cat_view'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)
original_df['out_prod_cat_view'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)
#AVG_TIME_PER_SITE_VISIT
original_df['out_avg_time_x_site_visit'] = 0
condition_hi = original_df.loc[0:,'out_avg_time_x_site_visit'][original_df['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_HI]

original_df['out_avg_time_x_site_visit'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)
#CANCELLATIONS_BEFORE_NOON
original_df['out_canc_bef_noon'] = 0
condition_hi = original_df.loc[0:,'out_canc_bef_noon'][original_df['CANCELLATIONS_BEFORE_NOON'] > CANCELLATIONS_BEFORE_NOON_HI]

original_df['out_canc_bef_noon'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)
#CANCELLATIONS_AFTER_NOON
original_df['out_canc_af_noon'] = 0
condition_hi = original_df.loc[0:,'out_canc_af_noon'][original_df['CANCELLATIONS_BEFORE_NOON'] > CANCELLATIONS_AFTER_NOON_HI]

original_df['out_canc_af_noon'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)
#MOBILE_LOGINS
original_df['out_mob_log'] = 0
condition_hi = original_df.loc[0:,'out_mob_log'][original_df['MOBILE_LOGINS'] > MOBILE_LOGINS_HI]
condition_lo = original_df.loc[0:,'out_mob_log'][original_df['MOBILE_LOGINS'] < MOBILE_LOGINS_LO]

original_df['out_mob_log'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

original_df['out_mob_log'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)
#PC_LOGINS
original_df['out_pc_log'] = 0
condition_hi = original_df.loc[0:,'out_pc_log'][original_df['PC_LOGINS'] > PC_LOGINS_HI]
condition_lo = original_df.loc[0:,'out_pc_log'][original_df['PC_LOGINS'] < PC_LOGINS_LO]

original_df['out_pc_log'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

original_df['out_pc_log'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)
#WEEKLY_PLAN
original_df['out_week_pl'] = 0
condition_hi = original_df.loc[0:,'out_week_pl'][original_df['WEEKLY_PLAN'] > WEEKLY_PLAN_HI]

original_df['out_week_pl'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)
#EARLY_DELIVERIES
original_df['out_early_del'] = 0
condition_hi = original_df.loc[0:,'out_early_del'][original_df['EARLY_DELIVERIES'] > EARLY_DELIVERIES_HI]

original_df['out_early_del'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)
#LATE_DELIVERIES
original_df['out_late_del'] = 0
condition_hi = original_df.loc[0:,'out_late_del'][original_df['LATE_DELIVERIES'] > LATE_DELIVERIES_HI]

original_df['out_late_del'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)
#FOLLOWED_RECOMMENDATIONS_PCT
original_df['out_fl_rec_pct'] = 0
condition_hi = original_df.loc[0:,'out_fl_rec_pct'][original_df['FOLLOWED_RECOMMENDATIONS_PCT'] > FOLLOWED_RECOMMENDATIONS_PCT_HI]

original_df['out_fl_rec_pct'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)
#AVG_PREP_VID_TIME
original_df['out_avg_prep_vtm'] = 0
condition_hi = original_df.loc[0:,'out_avg_prep_vtm'][original_df['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_HI]

original_df['out_avg_prep_vtm'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)
#LARGEST_ORDER_SIZE
original_df['out_lst_ord_size'] = 0
condition_hi = original_df.loc[0:,'out_lst_ord_size'][original_df['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_HI]
condition_lo = original_df.loc[0:,'out_lst_ord_size'][original_df['LARGEST_ORDER_SIZE'] < LARGEST_ORDER_SIZE_LO]

original_df['out_lst_ord_size'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

original_df['out_lst_ord_size'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)
"""#MASTER_CLASSES_ATTENDED
original_df['out_mst_cl_att'] = 0
condition_hi = original_df.loc[0:,'out_mst_cl_att'][original_df['MASTER_CLASSES_ATTENDED'] > MASTER_CLASSES_ATTENDED_HI]

original_df['out_mst_cl_att'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)"""
#MEDIAN_MEAL_RATING
original_df['out_meal_rt'] = 0
condition_hi = original_df.loc[0:,'out_meal_rt'][original_df['MEDIAN_MEAL_RATING'] > MEDIAN_MEAL_RATING_HI]
condition_lo = original_df.loc[0:,'out_meal_rt'][original_df['MEDIAN_MEAL_RATING'] < MEDIAN_MEAL_RATING_LO]

original_df['out_meal_rt'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

original_df['out_meal_rt'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)
#AVG_CLICKS_PER_VISIT
original_df['out_avg_cks_xvisit'] = 0
condition_hi = original_df.loc[0:,'out_avg_cks_xvisit'][original_df['AVG_CLICKS_PER_VISIT'] > AVG_CLICKS_PER_VISIT_HI]
condition_lo = original_df.loc[0:,'out_avg_cks_xvisit'][original_df['AVG_CLICKS_PER_VISIT'] < AVG_CLICKS_PER_VISIT_LO]

original_df['out_avg_cks_xvisit'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

original_df['out_avg_cks_xvisit'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)
#TOTAL_PHOTOS_VIEWED
original_df['out_tot_ph_views'] = 0
condition_hi = original_df.loc[0:,'out_tot_ph_views'][original_df['TOTAL_PHOTOS_VIEWED'] > TOTAL_PHOTOS_VIEWED_HI]

original_df['out_tot_ph_views'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)






################################################################################
# Train/Test Split
################################################################################

# use this space to set up testing and validation sets using train/test split

# Note: Be sure to set test_size = 0.25

#Scaling the data helps to have better predictions

scaling_data = original_df.drop(['NAME','FIRST_NAME',
                              'FAMILY_NAME'], axis = 1)

# INSTANTIATING a StandardScaler() object
scaler = StandardScaler()

# FITTING the scaler with the data
scaler.fit(scaling_data)

# TRANSFORMING our data after fit
scaled_data = scaler.transform(scaling_data)

# converting scaled data into a DataFrame
scaled_data_df = pd.DataFrame(scaled_data)

# adding labels to the scaled DataFrame
scaled_data_df.columns = scaling_data.columns

# checking the results
scaled_data_df.describe().round(2)

# declaring response variable
chef_target = original_df.loc[ : , 'CROSS_SELL_SUCCESS']

# declaring explanatory variables
chef_data_2 = scaled_data_df.loc[:,['FOLLOWED_RECOMMENDATIONS_PCT','MOBILE_NUMBER',

                                    ]]

# train-test split with stratification
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
            chef_data_2,
            chef_target,
            test_size = 0.25,
            random_state = 222,
            stratify = chef_target)




################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# use this space to instantiate, fit, and predict on your final model
# INSTANTIATING the model object without hyperparameters
full_gbm_default = GradientBoostingClassifier(
                                              random_state  = 222)


# FIT step is needed as we are not using .best_estimator
full_gbm_default_fit = full_gbm_default.fit(X_train_2, y_train_2)


# PREDICTING based on the testing set
full_gbm_default_pred = full_gbm_default_fit.predict(X_test_2)


# SCORING the results
print('Training ACCURACY:', full_gbm_default_fit.score(X_train_2, y_train_2).round(4))
print('Testing ACCURACY :', full_gbm_default_fit.score(X_test_2, y_test_2).round(4))
print('AUC Score        :', roc_auc_score(y_true  = y_test_2,
                                          y_score = full_gbm_default_pred).round(4))




################################################################################
# Final Model Score (score)
################################################################################

# use this space to score your final model on the testing set
# MAKE SURE TO SAVE YOUR TEST SCORE AS test_score
# Example: test_score = final_model.score(X_test, y_test)

test_score = roc_auc_score(y_true  = y_test_2,
                                          y_score = full_gbm_default_pred).round(4)
