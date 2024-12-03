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

#Header
st.title("Weightlifting and Powerlifting Data")
st.write("""
By Tyler Matson
""")


# Load Data (I tried to figure out this caching thing but I couldn't really get it to work well)
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


#Tab Selector
tab = st.sidebar.selectbox('Choose a tab',["Introduction and Data", "Imputation Methods", "Distributions of Lifters", "DOTS and Wilks", "Polynomial Regression with Body Weight and Weight Lifted"], index=0)

if tab == "Introduction and Data":
    st.subheader("Introduction and Data")
    st.write("""
    The goal of this project is to investigate biases and correlations within the sport of powerlifting. Certain scoring metrics may favor certain demographics of contestants more. This project also aims to see how powerlifting compares with olympic weightlifting when it comes to the correlation of body weight. 
    """)
    st.write("""
    This dataset was actually originally 300,000 data points, and I found a more updated version with 800,000, however because I'm not using the HPCC supercomputer on campus and because this is in Streamlit, I felt it necessary to initially trim the dataset down to 10,000 points. 
    """)
    
    # Display initial dataset
    st.subheader("Initial Dataset")
    st.write(df)


if tab == "Imputation Methods":
    st.write("""
    The following dataset is the one for all of the powerlifting statistics used in this project. Very clearly, it is missing many data points. In order to fix this, I plan to impute using either K-Nearest Neighbors (KNN), or Multivariate Imputation by Chained Equations (MICE). In order to determine which method is better, I will calculate the MSE of each one.  
    """)
    
    st.write("""
    BUT, I already ran into a problem, which is that the built in imputation methods do not like the rows that contain mostly empty values, which there are 89. So, in order to fix this, I dropped the rows with more than 50% of their values missing. Then, I can calculate the MSE of each method to determine which one is better and use it.
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
    Now, I can see that it really doesn't matter which method I use, so I will just use KNN because it is slightly better. 
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
    #Do necessary Calculations
    get_cols = ['Age', 'BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'Wilks']
    knn = KNNImputer(n_neighbors=5)
    df_knn = pd.DataFrame(knn.fit_transform(df[get_cols]), columns=get_cols)
    df.update(df_knn)
    df_new_powerlift = df.dropna()
    df_new_powerlift['Sex_Encoded'] = LabelEncoder().fit_transform(df_new_powerlift['Sex'])

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
    #Do necessary Calculations
    get_cols = ['Age', 'BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'Wilks']
    knn = KNNImputer(n_neighbors=5)
    df_knn = pd.DataFrame(knn.fit_transform(df[get_cols]), columns=get_cols)
    df.update(df_knn)
    df_new_powerlift = df.dropna()
    df_new_powerlift['Sex_Encoded'] = LabelEncoder().fit_transform(df_new_powerlift['Sex'])

    
    # DOTS Calculator
    def calculate_dots(row):
        if row['Sex'] == 'M':
            denominator = -0.000001093 * row['BodyweightKg']**4 + 0.0007391293 * row['BodyweightKg']**3 + -0.1918759221 * row['BodyweightKg']**2 + 24.0900756 * row['BodyweightKg'] + -307.75076
        else:
            denominator = -0.0000010706 * row['BodyweightKg']**4 + 0.0005158568 * row['BodyweightKg']**3 + -0.1126655495 * row['BodyweightKg']**2 + 13.6175032 * row['BodyweightKg'] + -57.96288
        
        
        return (row['Best3SquatKg'] + row['Best3BenchKg'] + row['Best3DeadliftKg']) * 500 / denominator
    
    # Apply Function
    df_new_powerlift['DOTS_Score'] = df_new_powerlift.apply(calculate_dots, axis=1)
    
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
    Something I wanted to look into was the score distributions across different categories. There are 2 popular scoring methods, and these are DOTS and WILKS. DOTS was not featured in the dataset, so I had to go find the formula online, (which funnily enough, the officially posted formula was wrong, and I had to find it from a calculator's github page). Both of these scores are calculated using the sex and bodyweight of the lifter. I wanted to see if either scoring method favored one sex over the other, or certain weight classes over the other. Additionally, I faced the consequences of my actions when my streamlit grinded to a halt in an attempt to plot 10,000 interactive points for each plot, so my solution was to undersample while keeping the highest and lowest points. There's a lot of plots, so I'm plotting 1 out of every 100 points with the downsample. 
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
    
if tab == "Polynomial Regression with Body Weight and Weight Lifted":

    # Load strongman dataset
    strongman = pd.read_csv('strongman.csv')
    strongman = strongman.dropna()  # Drop rows with missing values

    # Perform necessary preprocessing
    strongman['Weight'] = strongman['Weight'].str.replace('kg', '').astype(float)  # Convert weight to float
    strongman['Log'] = strongman['Log'].str.replace('kg', '').astype(float)  # Convert log to float
    strongman['Yoke'] = strongman['Yoke'].str.replace('kg', '').astype(float)  # Convert yoke to float
    
    #Do necessary Calculations
    get_cols = ['Age', 'BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'Wilks']
    knn = KNNImputer(n_neighbors=5)
    df_knn = pd.DataFrame(knn.fit_transform(df[get_cols]), columns=get_cols)
    df.update(df_knn)
    df_new_powerlift = df.dropna()
    df_new_powerlift['Sex_Encoded'] = LabelEncoder().fit_transform(df_new_powerlift['Sex'])
    
  
    # Do polynomial regression line
    def poly_reg(X, y, degree):
        poly = PolynomialFeatures(degree=degree)
        X_fit = poly.fit_transform(X)
        lin = LinearRegression()
        lin.fit(X_fit, y)
        y_pred = lin.predict(X_fit)
        return X.flatten()[np.argsort(X.flatten())], y_pred[np.argsort(X.flatten())], r2_score(y, y_pred)
    
    # Function to plot because this was getting messy
    def plot_poly(x, y, degree, title, y_label, rate=0.1):
        fig, ax = plt.subplots(figsize=(6, 4))
    
        # Compute polynomial regression 
        x_sorted, y_pred_sorted, r2 = poly_reg(x, y, degree)

        #Get random downsample 
        x_down = x[np.random.choice(len(x), size=int(len(x) * rate), replace=False)]
        y_down = y[np.random.choice(len(x), size=int(len(x) * rate), replace=False)]
    
        # Plot scatter points and line
        sns.scatterplot(x=x_down.flatten(), y=y_down, ax=ax)
        sns.lineplot(x=x_sorted, y=y_pred_sorted, color='red', ax=ax)
        
        # Plot title and labels
        ax.set_title(title + '\nPolynomial Degree: ' + str(degree) + ', R^2: ' + str(round(r2, 2)))
        ax.set_xlabel('Bodyweight (kg)')
        ax.set_ylabel(y_label + ' (kg)')
    
        return fig
    
    st.title("Polynomial Regression with Body Weight and Weight Lifted")
    
    st.write("""
    Next, I wanted to see how bodyweight affected each lift, including another strength sports' lifts: olympic lifting. Obviously, the heavier you are, the more fat and muscle you can have to lift more weight, but I wanted to see which lifts had a lower correlation, which then the argument could be made that there is more technicality in the lift as opposed to raw strength. Unsurprisingly, olympic lifts had a lower correlation, but it is interesting to see how much bodyweight correlates to the weight lifted. Additionally, strongman data was analyzed to see how the effects of bodyweight affected that sport.  
    """)
    
    # Set the polynomial degree slider (yeah this takes a long time for higher values)
    degree = st.slider("Select Polynomial Degree", min_value=1, max_value=5, value=2)
    
    
    #Bodyweight vs Squat
    plot = plot_poly(df_new_powerlift[['BodyweightKg']].values, df_new_powerlift['Best3SquatKg'].values, degree, 'Bodyweight vs. Squat', 'Best3SquatKg', rate=0.01)
    st.pyplot(plot)
    
    #Bodyweight vs Bench
    plot = plot_poly(df_new_powerlift[['BodyweightKg']].values, df_new_powerlift['Best3BenchKg'].values, degree, 'Bodyweight vs. Bench', 'Best3BenchKg', rate=0.01)
    st.pyplot(plot)
    
    #Bodyweight vs Deadlift
    plot = plot_poly(df_new_powerlift[['BodyweightKg']].values, df_new_powerlift['Best3DeadliftKg'].values, degree, 'Bodyweight vs Deadlift', 'Best3DeadliftKg', rate=0.01)
    st.pyplot(plot)
    
    st.subheader("Weightlifting: Snatch and Clean & Jerk")
       
    #Bodyweight vs Snatch
    plot = plot_poly(df_weightlifting[['Bodyweight_(kg)']].values, df_weightlifting['Snatch_(kg)'].values, degree, 'Bodyweight vs. Snatch', 'Snatch_(kg)', rate=0.1)
    st.pyplot(plot)
    
    #Bodyweight vs Clean and Jerk
    plot = plot_poly(df_weightlifting[['Bodyweight_(kg)']].values, df_weightlifting['Clean_&_Jerk_(kg)'].values, degree, 'Bodyweight vs. Clean & Jerk', 'Clean_&_Jerk_(kg)', rate=0.1)
    st.pyplot(plot)


    st.subheader("Strongman: Log and Yoke")

    # Strongman: Log
    plot = plot_poly(strongman[['Weight']].values, strongman['Log'].values, degree, 'Weight vs. Log', 'Log (kg)', rate=0.2)
    st.pyplot(plot)

    # Strongman: Yoke
    plot = plot_poly(strongman[['Weight']].values, strongman['Yoke'].values, degree, 'Weight vs. Yoke', 'Yoke (kg)', rate=0.2)
    st.pyplot(plot)

    # Strongman: Deadlift
    plot = plot_poly(strongman[['Weight']].values, strongman['Deadlift'].values, degree, 'Weight vs. Deadlift', 'Deadlift (kg)', rate=0.2)
    st.pyplot(plot)
