import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import streamlit as st
import plotly.express as px
from scipy.stats import ttest_ind, f_oneway, mannwhitneyu
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
#Header
st.title("Weightlifting and Powerlifting Data")
st.write("""
By Tyler Matson
""")


# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('sampled_openpowerlifting.csv')[['Age', 'BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'Wilks', 'Sex', 'Equipment', 'WeightClassKg']]
    
@st.cache_data
def load_weightlifting_data():
    df = pd.read_csv('weightlifting.csv')
    df.columns = df.columns.str.replace(' ', '_')
    return df[['Bodyweight_(kg)', 'Snatch_(kg)', 'Clean_&_Jerk_(kg)']]

#Call the data
df = load_data()
df_weightlifting = load_weightlifting_data()

#Do necessary calculations (I'm doing it all up here so it doesn't have to do it every page, but I'm using a different data set than the one I'm starting with
    #so I can still show the IDA)
# Encode 'Sex' column before applying KNNImputer because it doesn't like it when I do it after
df2=load_data()
df2['Sex_Encoded'] = LabelEncoder().fit_transform(df['Sex'])
get_cols = ['Age', 'BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'Wilks', 'Sex_Encoded']
knn = KNNImputer(n_neighbors=5)
df_knn = pd.DataFrame(knn.fit_transform(df2[get_cols]), columns=get_cols)
df2.update(df_knn)
df_new_powerlift = df2.dropna()


# DOTS Calculator
def calculate_dots(row):
    if row['Sex'] == 'M':
        denominator = -0.000001093 * row['BodyweightKg']**4 + 0.0007391293 * row['BodyweightKg']**3 + -0.1918759221 * row['BodyweightKg']**2 + 24.0900756 * row['BodyweightKg'] + -307.75076
    else:
        denominator = -0.0000010706 * row['BodyweightKg']**4 + 0.0005158568 * row['BodyweightKg']**3 + -0.1126655495 * row['BodyweightKg']**2 + 13.6175032 * row['BodyweightKg'] + -57.96288
    
    
    return (row['Best3SquatKg'] + row['Best3BenchKg'] + row['Best3DeadliftKg']) * 500 / denominator

# Apply Function
df_new_powerlift['DOTS_Score'] = df_new_powerlift.apply(calculate_dots, axis=1)

#Tab Selector
tab = st.sidebar.selectbox('Choose a tab',["Introduction and Data", "Imputation Methods", "Distributions of Lifters", "DOTS and Wilks", "DOTS and Wilks Bias Analysis","Polynomial Regression with Body Weight and Weight Lifted", "Lift Prediction Calculator"], index=0)

if tab == "Introduction and Data":
    st.subheader("Introduction and Data")
    st.write("""
    The goal of this project is to investigate biases and correlations within the sport of powerlifting. Certain scoring metrics may favor certain demographics of contestants more. This project also aims to see how powerlifting compares with olympic weightlifting and strongman when it comes to the correlation of body weight. 
    """)
    st.write("""
    This dataset was actually originally 300,000 data points, and I found a more updated version with 800,000, however because I'm not using a supercomputer and because this is in Streamlit, I felt it necessary to initially trim the dataset down to 10,000 points. 
    """)
    
    # Display initial dataset
    st.subheader("Initial Dataset")
    st.write(df)


if tab == "Imputation Methods":
    st.write("""
    The following dataset is the one for all of the powerlifting statistics used in this project. Very clearly, it is missing many data points. In order to fix this, I plan to impute using either K-Nearest Neighbors (KNN), or Multivariate Imputation by Chained Equations (MICE). In order to determine which method is better, I will calculate the MSE of each one.  
    """)
    
    st.write("""
    A problem was already encountered, which is that the built in imputation methods do not like the rows that contain mostly empty values, which there are 89. So, in order to fix this, I dropped the rows with more than 50% of their values missing. Then, I can calculate the MSE of each method to determine which one is better and use it.
    """)
    
    # Check cols for negative values because of weird formatting that also gave me a headache
    cols = ['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']
    df[cols] = df[cols].applymap(lambda x: np.nan if x < 0 else x)

    #THESE 89 VALUES CAUSED ME SO MUCH STRESS LIKE ACTUALLY 1.5 HOURS OF DEBUGGING
    
    # Imputing
    impute = df[['Age', 'BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'Wilks']]
    new_df = impute[impute.isnull().sum(axis=1) <= impute.shape[1] / 2]
    st.subheader("Dataset After Removing Rows with 50% or More Missing Values")
    st.write(new_df)
    
    #KNN Imputation
    knn = KNNImputer(n_neighbors=5)
    df_knn = pd.DataFrame(knn.fit_transform(new_df), columns=new_df.columns)
    
    #MICE
    mice = IterativeImputer(random_state=42)
    df_mice= pd.DataFrame(mice.fit_transform(new_df), columns=new_df.columns)
    
    
    # Compare new values with original dataset values
    original = new_df[~new_df.isnull()].dropna(how='all')
    knn = df_knn[~new_df.isnull()].dropna(how='all')
    mice = df_mice[~new_df.isnull()].dropna(how='all')
    
    # Reset index to align data
    original_align = original.loc[original.index.intersection(knn.index)].dropna(how='any').reset_index(drop=True)
    knn_align = knn.loc[original.index.intersection(knn.index)].dropna(how='any').reset_index(drop=True)
    mice_align = mice.loc[original.index.intersection(knn.index)].dropna(how='any').reset_index(drop=True)
    
    mse_knn = mean_squared_error(original_align, knn_align)
    mse_mice = mean_squared_error(original_align, mice_align)
    
    # Display
    st.subheader("MSE Comparison of Imputation Methods")
    st.write("MSE for KNN Imputation: " + str(mse_knn))
    st.write("MSE for MICE Imputation: " + str(mse_mice))
    
    st.write("""
    Now, I can see that it really doesn't matter which method is, so I will just use KNN because it is slightly better. 
    """)


    #RESETTING MY KNN IMPUTATION BECAUSE THESE 89 VALUES CAUSED ME SO MUCH STRESS IN FIGURING OUT WHY MY STUFF WASN'T WORKING, AND THIS WAY WORKS SO I'M NOT TOUCHING IT AGAIN
    
    #Do necessary Calculations
    get_cols = ['Age', 'BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'Wilks']
    knn = KNNImputer(n_neighbors=5)
    df_knn = pd.DataFrame(knn.fit_transform(df[get_cols]), columns=get_cols)
    df.update(df_knn)
    df_new_powerlift = df.dropna()
    df_new_powerlift['Sex_Encoded'] = LabelEncoder().fit_transform(df_new_powerlift['Sex'])
    
    st.write("""
    Additionally, I am going to label encoded the Sex column for processing purposes.
    """)#I ended up not actually using this
    
    # Display the final dataset
    st.subheader("Final Dataset with Label Encoding")
    st.write(df_new_powerlift)


if tab == "Distributions of Lifters":

    st.title("Powerlifting Competitor Analysis")
    
    st.write("""
    Powerlifting is a sport that has many different sub categories and classes, so I wanted to visualize them to show someone who is not familiar with powerlifting. Something to note for the weight classes is the most popular weight classes are the ones featured in large organizations, such as the USAPL, USPA, and IPF.
    """)
    

    
    # Men and Women Bar Chart
    st.subheader("Number of Men and Women Category Competitors")
    sex_count = df_new_powerlift['Sex'].value_counts()
    plot = px.bar(sex_count, x=sex_count.index, y=sex_count.values, labels={'y':'Number of Competitors', 'index':'Sex'}, title="Number of Men and Women Category Competitors")
    st.plotly_chart(plot)
    
    # Equipment Bar Chart
    st.subheader("Number of Competitors in Each Equipment Category")
    equipment_count = df_new_powerlift['Equipment'].value_counts()
    plot = px.bar(equipment_count, x=equipment_count.index, y=equipment_count.values, labels={'y':'Number of Competitors', 'index':'Equipment'}, title="Number of Competitors in Each Equipment Category")
    st.plotly_chart(plot)
    
    #Men's Weight Classes
    st.subheader("Male Category Competitors by Weight Class")
    men_count = df_new_powerlift[df_new_powerlift['Sex'] == 'M']['WeightClassKg'].value_counts().sort_index()
    plot = px.bar(men_count, x=men_count.index, y=men_count.values, labels={'y':'Number of Competitors', 'index':'Weight Class (Kg)'}, title="Number of Male Category Competitors in Each Weight Class")
    plot.update_layout(bargap=0.1) 
    st.plotly_chart(plot)
    
    #Women's Weight Classes
    st.subheader("Female Category Competitors by Weight Class")
    women_count = df_new_powerlift[df_new_powerlift['Sex'] == 'F']['WeightClassKg'].value_counts().sort_index()
    plot = px.bar(women_count, x=women_count.index, y=women_count.values, labels={'y':'Number of Competitors', 'index':'Weight Class (Kg)'}, title="Number of Female Category Competitors in Each Weight Class")
    plot.update_layout(bargap=0.1)  
    st.plotly_chart(plot)

if tab == "DOTS and Wilks":
    
    # Display head of DOTS-added df
    st.subheader("Dataset with DOTS Score")
    st.write(df_new_powerlift[['Sex', 'BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'DOTS_Score']].head())
    
    st.subheader("Comparison of DOTS and Wilks Scores")
    
    # Downsampling because too much data for Streamlit
    def downsample_data(df, y, x, rate=0.1):
        final = []
        grouped = df.groupby(x)
        for name, group in grouped:
            sorted = np.sort(group[y].values)
            if len(sorted) > 2:
                min = sorted[0]
                max = sorted[-1]
                returned_index = np.linspace(1, len(sorted) - 2, int(len(sorted) * rate), dtype=int)
                returned_values = sorted[returned_index]
                returned_group = pd.DataFrame({x: name, y: np.concatenate([[min], returned_values, [max]])})
            else:
                returned_group = group[[x, y]]
            final.append(returned_group)
        return pd.concat(final)
    
    
    st.write("""
    Something I wanted to look into was the score distributions across different categories. There are 2 popular scoring methods, and these are DOTS and WILKS. DOTS was not featured in the dataset, so I had to go find the formula online to implement it. Both of these scores are calculated using the sex and bodyweight of the lifter. I wanted to see if either scoring method favored one sex over the other, or certain weight classes over the other. Additionally, due to the technical constraints of Streamlit, the process slows greatly due to the size of the data, so my solution was to undersample while keeping the highest and lowest points. There are a significant amount of points, so this program is plotting 1 out of every 100 points with the downsample. 
    """)
    
    #DOTS Sex
    st.subheader("DOTS Score Comparison: Men's Category vs Women's Category")
    down = downsample_data(df_new_powerlift, 'DOTS_Score', 'Sex', rate=0.01)
    plot = px.box(down, x='Sex', y='DOTS_Score', points="all", title="DOTS Score Comparison Between Men's Category vs Women's Category", labels={'Sex': 'Sex', 'DOTS_Score': 'DOTS Score'})
    st.plotly_chart(plot)
    
    #Wilks Sex
    st.subheader("Wilks Score Comparison: Men's Category vs Women's Category")
    down = downsample_data(df_new_powerlift, 'Wilks', 'Sex', rate=0.01)
    plot = px.box(down, x='Sex', y='Wilks', points="all", title="Wilks Score Comparison Between Men's Category vs Women's Category", labels={'Sex': 'Sex', 'Wilks': 'Wilks Score'})
    st.plotly_chart(plot)
    
    #DOTS Across Weight 
    st.subheader("DOTS Score Comparison Across Weight Classes")
    down = downsample_data(df_new_powerlift, 'DOTS_Score', 'WeightClassKg', rate=0.01)
    plot = px.box(down, x='WeightClassKg', y='DOTS_Score', points="all", title="DOTS Score Across Weight Classes", labels={'WeightClassKg': 'Weight Class (Kg)', 'DOTS_Score': 'DOTS Score'})
    plot.update_xaxes(categoryorder="category ascending")
    st.plotly_chart(plot)
    
    #Wilks Across Weight
    st.subheader("Wilks Score Comparison Across Weight Classes")
    down = downsample_data(df_new_powerlift, 'Wilks', 'WeightClassKg', rate=0.01)
    plot = px.box(down, x='WeightClassKg', y='Wilks', points="all", title="Wilks Score Across Weight Classes", labels={'WeightClassKg': 'Weight Class (Kg)', 'Wilks': 'Wilks Score'})
    plot.update_xaxes(categoryorder="category ascending")
    st.plotly_chart(plot)
    
    #DOTS Across Equipment
    st.subheader("DOTS Score Comparison Across Equipment Categories")
    down = downsample_data(df_new_powerlift, 'DOTS_Score', 'Equipment', rate=0.01)
    plot = px.box(down, x='Equipment', y='DOTS_Score', points="all", title="DOTS Score Across Equipment Categories", labels={'Equipment': 'Equipment', 'DOTS_Score': 'DOTS Score'})
    st.plotly_chart(plot)
    
    #Wilks Across Equipment
    st.subheader("Wilks Score Comparison Across Equipment Categories")
    down = downsample_data(df_new_powerlift, 'Wilks', 'Equipment', rate=0.01)
    plot = px.box(down, x='Equipment', y='Wilks', points="all", title="Wilks Score Across Equipment Categories", labels={'Equipment': 'Equipment', 'Wilks': 'Wilks Score'})
    st.plotly_chart(plot)
    
if tab == "DOTS and Wilks Bias Analysis":

    st.title("DOTS and Wilks Bias Analysis")

    
    # Mapping Weight Classes (I hate this, this sucked)
    common_weight = {
        'M': [52, 56, 60, 67.5, 75, 82.5, 90, 100, 110, 125, 140, float('inf')],
        'F': [44, 48, 52, 56, 60, 67.5, 75, 82.5, 90, 100, float('inf')]
    }
    common_weight_labels = {
        'M': ['52kg', '56kg', '60kg', '67.5kg', '75kg', '82.5kg', '90kg', '100kg', '110kg', '125kg', '140kg', '140+kg'],
        'F': ['44kg', '48kg', '52kg', '56kg', '60kg', '67.5kg', '75kg', '82.5kg', '90kg', '100kg', '100+kg']
    }
    
    # Map weight class
    df_new_powerlift['USAPL_WeightClass'] = df_new_powerlift.apply(
        lambda row: next(
            (common_weight_labels[row['Sex']][i] for i, limit in enumerate(common_weight[row['Sex']]) 
             if row['BodyweightKg'] <= limit), None
        ), axis=1
    )
    
    # Replace the +'s for numerical sorting
    weight_class_numeric = {'52kg': 52, '56kg': 56, '60kg': 60, '67.5kg': 67.5, '75kg': 75,
                            '82.5kg': 82.5, '90kg': 90, '100kg': 100, '110kg': 110, '125kg': 125,
                            '140kg': 140, '140+kg': 141, '44kg': 44, '48kg': 48, '100+kg': 101}
    df_new_powerlift['USAPL_WeightClass'] = df_new_powerlift['USAPL_WeightClass'].map(weight_class_numeric)
    
    # ANOVA Analysis
    st.subheader("ANOVA Analysis")
    st.write("""
    Besides eyeballing graphs, doing a simple Analysis of Variance test on the DOTS on Weight Class and Wilks shows significant p-values that can be explored further with Tukey's HSD.
    """)
    
    results = f_oneway(*[
        df_new_powerlift[df_new_powerlift['WeightClassKg'] == wc]['DOTS_Score']
        for wc in df_new_powerlift['WeightClassKg'].unique()
    ])
    st.write("ANOVA for DOTS by Weight Class")
    st.write("F-statistic: ", results.statistic, "P-value: ", results.pvalue)
    
    results = f_oneway(*[
        df_new_powerlift[df_new_powerlift['Sex'] == sex]['DOTS_Score']
        for sex in df_new_powerlift['Sex'].unique()
    ])
    st.write("ANOVA for DOTS by Sex")
    st.write("F-statistic: ", results.statistic, "P-value: ", results.pvalue)
    
    # Tukey Analysis
    st.subheader("Tukey HSD Testing")
    st.write("""
    By using Tukey's Honestly Signifcant Difference Test, we can analyze the Weight Classes and Sex for bias in the data. The library we are using, statsmodel, will automatically apply the Tukey-Kramer adjustment to account for the unequal sample sizes. By analyzing the plots generated using the Tukey HSD, we can see the differences in the groups
    """)
    
    male_lifters = df_new_powerlift[df_new_powerlift['Sex'] == 'M']
    female_lifters = df_new_powerlift[df_new_powerlift['Sex'] == 'F']
    
    # Tukey for DOTS (Male and Female)
    tukey_male_dots = pairwise_tukeyhsd(endog=male_lifters['DOTS_Score'], groups=male_lifters['USAPL_WeightClass'])
    tukey_female_dots = pairwise_tukeyhsd(endog=female_lifters['DOTS_Score'], groups=female_lifters['USAPL_WeightClass'])
    
    # Plot results for DOTS (Male and Female)
    st.subheader("DOTS Score by Weight Class (Male)")
    fig_male = tukey_male_dots.plot_simultaneous(xlabel="DOTS Score", ylabel="Weight Class")
    st.pyplot(fig_male)
    
    st.subheader("DOTS Score by Weight Class (Female)")
    fig_female = tukey_female_dots.plot_simultaneous(xlabel="DOTS Score", ylabel="Weight Class")
    st.pyplot(fig_female)
    
    # Tukey for Wilks (Male and Female)
    tukey_male_wilks = pairwise_tukeyhsd(endog=male_lifters['Wilks'], groups=male_lifters['USAPL_WeightClass'])
    tukey_female_wilks = pairwise_tukeyhsd(endog=female_lifters['Wilks'], groups=female_lifters['USAPL_WeightClass'])
    
    # Plot results for Wilks (Male and Female)
    st.subheader("Wilks Score by Weight Class (Male)")
    fig_male_wilks = tukey_male_wilks.plot_simultaneous(xlabel="Wilks Score", ylabel="Weight Class")
    st.pyplot(fig_male_wilks)
    
    st.subheader("Wilks Score by Weight Class (Female)")
    fig_female_wilks = tukey_female_wilks.plot_simultaneous(xlabel="Wilks Score", ylabel="Weight Class")
    st.pyplot(fig_female_wilks)

    st.subheader("Tukey HSD Analysis")
    st.write("""
    In these plots, we can clearly see biases towards certain weight classes. In the male plots, these plots appear to clearly show biases towards the heavier weight classes, and in the female plots, we can see a bias towards the lighter lifters. It is important that all confounding variables are considered, and a thorough understanding of the sport of powerlifting is utlilized. 
    
    In the case of the men's plots, we can see an almost constant increase in the DOTS and Wilks scores as weight increases, and this can be explained with the fact that as men go to the gym more consistently, they will most likely gain more weight in muscle. Beginner lifters are more likely to be lighter, and can explain why on average, the lighter lifters are scoring lower. We are seeing selection bias as this plot can be explained by experience levels, which we do not have the data to account for. 

    However, in the case of the women's plots, we can see higher scores in the lighter weight divisions, at 52 kg and 56 kg. I currently do not have an explanation for why this is, as while women won't put on as much weight as men, they should still gain weight as they gain muscle and experience in the sport. This data could be an indicator that Wilks and DOTS are biased towards certain women's weight classes, however without a much more thorough investigation, it would be irresponsible to make a claim like that. Perhaps interviews with experts in the sport could reveal an unknown variable. 
    """)


    st.subheader("Tukey HSD Testing (cont.)")
    # Tukey for DOTS and Wilks (Sex)
    for score, column in zip(["DOTS", "Wilks"], ['DOTS_Score', 'Wilks']):
        tukey_sex = pairwise_tukeyhsd(
            endog=df_new_powerlift[column], 
            groups=df_new_powerlift['Sex']
        )
        st.write(f"Tukey HSD Summary for {score} by Sex")
        st.text(tukey_sex.summary())
        
        fig_sex = tukey_sex.plot_simultaneous(xlabel={score}, ylabel='Sex')
        st.pyplot(fig_sex)

    st.subheader("Tukey HSD Analysis (cont.)")
    st.write("""
    These plots show a clear bias towards male lifters, however it is still important to consider outside variables before making a huge claim. To further investigate, I created a new dataset of solely the lifters who have scored 600+ DOTS, which is an extremely high score, and made some plots of the distributions.
    """)

    op83 = pd.read_csv('openpowerlifting_first_83.csv')
    
    # Function to map weight classes
    def map_classes(row):
        weight_class = str(row['Class']).strip()
        if '+' in weight_class:  # Handle "140+kg" type cases
            return common_weight_labels[row['Sex']][-1]
        else:
            weight = float(weight_class)
            divisions = common_weight[row['Sex']]
            labels = common_weight_labels[row['Sex']]
            for i, limit in enumerate(divisions):
                if weight <= limit * 2.20462:  # Convert to pounds
                    return labels[i]
    
    # Apply the mapping function
    op83['USAPL_WeightClass'] = op83.apply(map_classes, axis=1)
    
    # Group the data by the new USAPL weight classes and sex
    data83 = op83.groupby(['USAPL_WeightClass', 'Sex']).size().unstack(fill_value=0)

    # Separate male and female data
    female_labels = common_weight_labels['F']
    male_labels = common_weight_labels['M']

    # Female lifters
    female_data = data83['F'].reindex(female_labels, fill_value=0).reset_index()
    female_data.columns = ['Weight Class', 'Count']
    fig_female = px.bar(
        female_data,
        x='Weight Class',
        y='Count',
        title='Powerlifters by Weight Class (Female)',
        labels={'Weight Class': 'Weight Class', 'Count': 'Number of Lifters'},
    )
    st.plotly_chart(fig_female)

    # Male lifters
    male_data = data83['M'].reindex(male_labels, fill_value=0).reset_index()
    male_data.columns = ['Weight Class', 'Count']
    fig_male = px.bar(
        male_data,
        x='Weight Class',
        y='Count',
        title='Powerlifters by Weight Class (Male)',
        labels={'Weight Class': 'Weight Class', 'Count': 'Number of Lifters'},
    )
    st.plotly_chart(fig_male)

    lifter_counts = op83['Sex'].value_counts().reset_index()
    lifter_counts.columns = ['Sex', 'Count']
    fig = px.bar(
        lifter_counts,
        x='Sex',
        y='Count',
        title='Distribution of Lifters by Sex',
        labels={'Sex': 'Sex', 'Count': 'Count'},
    )
    fig.update_layout(xaxis={'categoryorder': 'total descending'}, width=800, height=500)
    st.plotly_chart(fig)

    st.write("""
        Interestingly enough, when analyzing the best lifters, women outnumber men by a decent amount, despite what seems like a bias favoring men. The distribution still seems to favor lighter weight women competitors over heavier weight women competitors, but again, further analysis would be required in order to definitively prove that the score metrics favor the lighter women. 
        """)
    
if tab == "Polynomial Regression with Body Weight and Weight Lifted":

    # Load strongman dataset
    strongman = pd.read_csv('Strongman.csv')
    strongman = strongman.dropna()  # Drop rows with missing values

    # Perform necessary preprocessing
    strongman['Weight'] = strongman['Weight'].str.replace('kg', '').astype(float)
    strongman['Log'] = strongman['Log'].str.replace('kg', '').astype(float)
    strongman['Yoke'] = strongman['Yoke'].str.replace('kg', '').astype(float)
    strongman['Deadlift'] = strongman['Deadlift'].str.replace('kg', '').astype(float)


    df_weightlifting = df_weightlifting[(df_weightlifting['Snatch_(kg)'] > 0) & (df_weightlifting['Clean_&_Jerk_(kg)'] > 0)]
  
    # Do polynomial regression line
    def poly_reg(X, y, degree, reg_rate = 1.0):

        n_samples = int(len(X) * reg_rate)
        sampled_indices = np.random.choice(len(X), n_samples, replace=False)
        X = X[sampled_indices]
        y = y[sampled_indices]
        
        poly = PolynomialFeatures(degree=degree)
        X_fit = poly.fit_transform(X)
        lin = LinearRegression()
        lin.fit(X_fit, y)
        y_pred = lin.predict(X_fit)
        correlation = np.corrcoef(X.flatten(), y)[0, 1]
        return X.flatten()[np.argsort(X.flatten())], y_pred[np.argsort(X.flatten())], r2_score(y, y_pred), correlation
    
    # Function to plot because this was getting messy
    def plot_poly(x, y, degree, title, y_label, rate=0.1, reg_rate = 1.0):
        fig, ax = plt.subplots(figsize=(6, 4))
    
        # Compute polynomial regression 
        x_sorted, y_pred_sorted, r2, correlation = poly_reg(x, y, degree, reg_rate)

        #Get random downsample 
        x_down = x[np.random.choice(len(x), size=int(len(x) * rate), replace=False)]
        y_down = y[np.random.choice(len(x), size=int(len(x) * rate), replace=False)]
    
        # Plot scatter points and line
        sns.scatterplot(x=x_down.flatten(), y=y_down, ax=ax)
        sns.lineplot(x=x_sorted, y=y_pred_sorted, color='red', ax=ax)


        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    
        # Plot title and labels
        ax.set_title(title + '\nPolynomial Degree: ' + str(degree) + ', R^2: ' + str(round(r2, 2))+ " Correlation: "+ str(round(correlation, 2)))
        ax.set_xlabel('Bodyweight (kg)')
        ax.set_ylabel(y_label + ' (kg)')
    
        return fig
    
    st.title("Polynomial Regression with Body Weight and Weight Lifted")
    
    st.write("""
    Next, I wanted to see how bodyweight affected each lift, including other strength sports' lifts: Olympic Lifting and Strongman. The heavier you are, the more fat and muscle you can have to lift more weight, but I wanted to see which lifts had a lower or higher correlation, to see how much it matters by lift. Surprisingly, olympic lifts had a higher correlation.
    """)

    st.write("""
        The sliders allow you to select how much data is used for the regression line. Because of the large amount of data for powerlifting used, using 100% of the data for the regression will take a long time. 
        """)
    # Set the downsample rate for the data for the regression (yeah this takes a long time for higher values)
    reg_rate = st.slider("Select Regression Downsampling Rate", min_value=0.1, max_value=1.0, value=.3, step=0.1)
    
    
    #Bodyweight vs Squat
    plot = plot_poly(df_new_powerlift[['BodyweightKg']].values, df_new_powerlift['Best3SquatKg'].values, 2, 'Bodyweight vs. Squat', 'Best3SquatKg', rate=0.05, reg_rate=reg_rate)
    st.pyplot(plot)
    
    #Bodyweight vs Bench
    plot = plot_poly(df_new_powerlift[['BodyweightKg']].values, df_new_powerlift['Best3BenchKg'].values, 2, 'Bodyweight vs. Bench', 'Best3BenchKg', rate=0.05,reg_rate=reg_rate)
    st.pyplot(plot)
    
    #Bodyweight vs Deadlift
    plot = plot_poly(df_new_powerlift[['BodyweightKg']].values, df_new_powerlift['Best3DeadliftKg'].values, 2, 'Bodyweight vs Deadlift', 'Best3DeadliftKg', rate=0.05, reg_rate=reg_rate)
    st.pyplot(plot)
    
    st.subheader("Weightlifting: Snatch and Clean & Jerk")

    # Set the downsample rate for the data for the regression (yeah this takes a long time for higher values)
    reg_rate = st.slider("Select Regression Downsampling Rate", min_value=0.1, max_value=1.0, value=.8, step=0.1)
       
    #Bodyweight vs Snatch
    plot = plot_poly(df_weightlifting[['Bodyweight_(kg)']].values, df_weightlifting['Snatch_(kg)'].values, 2, 'Bodyweight vs. Snatch', 'Snatch_(kg)', rate=0.4,reg_rate=reg_rate)
    st.pyplot(plot)
    
    #Bodyweight vs Clean and Jerk
    plot = plot_poly(df_weightlifting[['Bodyweight_(kg)']].values, df_weightlifting['Clean_&_Jerk_(kg)'].values, 2, 'Bodyweight vs. Clean & Jerk', 'Clean_&_Jerk_(kg)', rate=0.4, reg_rate=reg_rate)
    st.pyplot(plot)


    st.subheader("Strongman: Log and Yoke")

    # Set the downsample rate for the data for the regression (yeah this takes a long time for higher values)
    reg_rate = st.slider("Select Regression Downsampling Rate", min_value=0.1, max_value=1.0, value=1.0, step=0.1)

    # Strongman: Log
    plot = plot_poly(strongman[['Weight']].values, strongman['Log'].values, 2, 'Weight vs. Log', 'Log (kg)', rate=1, reg_rate=reg_rate)
    st.pyplot(plot)

    # Strongman: Yoke
    plot = plot_poly(strongman[['Weight']].values, strongman['Yoke'].values, 2, 'Weight vs. Yoke', 'Yoke (kg)', rate=1, reg_rate=reg_rate)
    st.pyplot(plot)

    # Strongman: Deadlift
    plot = plot_poly(strongman[['Weight']].values, strongman['Deadlift'].values, 2, 'Weight vs. Deadlift', 'Deadlift (kg)', rate=1, reg_rate=reg_rate)
    st.pyplot(plot)

if tab == "Lift Prediction Calculator":
    st.subheader("Predict Your Missing Lift")
    st.write("""
    Use this calculator to predict one of your lifts (Squat, Bench, or Deadlift) based on your body weight, sex, and the other two lifts. This can be useful if you want to predict where you should be on a lift, or to identify which of your lifts is the weakest.
    """)

    # Prepare data for training
    df_data = df_new_powerlift[['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'BodyweightKg', 'Sex_Encoded']].dropna()

    # Split data into predictors and target
    def data_split(target):
        X = df_data.drop(columns=[target])
        y = df_data[target]
        return X, y

    # Dropdown to select the model type
    model_type = st.selectbox("Select the model type:", ["LightGBM", "MLP", "Random Forest", "KNN"])

    # Create models for the lift based on the model
    models = {}
    for target in ['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']:
        X, y = data_split(target)
        if model_type == "LightGBM":
            model = LGBMRegressor(n_estimators=100, learning_rate=0.1)
        elif model_type == "MLP":
            model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500)
        elif model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100)
        elif model_type == "KNN":
            model = KNeighborsRegressor(n_neighbors=5)

        model.fit(X, y)
        models[target] = model

    # User Input
    input_lift = st.selectbox("Which lift do you want to predict?", ["Squat", "Bench", "Deadlift"])
    bodyweight = st.number_input("Enter your body weight (kg):", min_value=20.0, max_value=200.0, step=0.1)
    sex = st.selectbox("Select your sex:", ["Male", "Female"])
    sex_encoded = 0 if sex == "Male" else 1


    # Execute ~
    if input_lift == "Squat":
        bench = st.number_input("Enter your Bench (kg):", min_value=20.0, max_value=600.0, step=0.1)
        deadlift = st.number_input("Enter your Deadlift (kg):", min_value=20.0, max_value=600.0, step=0.1)
        if st.button("Predict Squat"):
            predicted_squat = models['Best3SquatKg'].predict([[bench, deadlift, bodyweight, sex_encoded]])[0]
            st.write("Predicted Squat:", round(predicted_squat, 2), "kg")

    elif input_lift == "Bench":
        squat = st.number_input("Enter your Squat (kg):", min_value=20.0, max_value=600.0, step=0.1)
        deadlift = st.number_input("Enter your Deadlift (kg):", min_value=20.0, max_value=600.0, step=0.1)
        if st.button("Predict Bench"):
            predicted_bench = models['Best3BenchKg'].predict([[squat, deadlift, bodyweight, sex_encoded]])[0]
            st.write("Predicted Bench:", round(predicted_bench, 2), "kg")

    elif input_lift == "Deadlift":
        squat = st.number_input("Enter your Squat (kg):", min_value=20.0, max_value=600.0, step=0.1)
        bench = st.number_input("Enter your Bench (kg):", min_value=20.0, max_value=600.0, step=0.1)
        if st.button("Predict Deadlift"):
            predicted_deadlift = models['Best3DeadliftKg'].predict([[squat, bench, bodyweight, sex_encoded]])[0]
            st.write("Predicted Deadlift:", round(predicted_deadlift, 2), "kg")
