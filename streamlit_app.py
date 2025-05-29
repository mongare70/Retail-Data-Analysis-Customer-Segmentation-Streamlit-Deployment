import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.ensemble import IsolationForest
from scipy.stats import mode
import warnings

warnings.filterwarnings("ignore")


# Function to perform data cleaning and preprocessing
def preprocess_data(df):
    # Handle missing values
    numerical_cols = [
        "Age",
        "Total_Purchases",
        "Amount",
        "Total_Amount",
        "Ratings",
        "sentiment",
    ]
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Drop rows with missing values in specified columns
    subset_cols_to_drop_na = [
        "Transaction_ID",
        "Customer_ID",
        "Name",
        "Email",
        "Gender",
        "Date",
        "Time",
        "Address",
        "Zipcode",
        "State",
        "City",
        "Income",
        "Product_Brand",
        "Feedback",
        "Customer_Segment",
        "Product_Category",
    ]

    # Check if columns exist before dropping
    subset_cols_to_drop_na_existing = [
        col for col in subset_cols_to_drop_na if col in df.columns
    ]
    if subset_cols_to_drop_na_existing:
        df.dropna(subset=subset_cols_to_drop_na_existing, inplace=True)

    # Convert data types
    int_cols = ["Customer_ID", "Transaction_ID", "Age", "Total_Purchases", "Ratings"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Outlier removal using Isolation Forest
    if all(col in df.columns for col in numerical_cols):
        subset = df[numerical_cols]
        isolation_forest = IsolationForest(random_state=42)
        outlier_prediction = isolation_forest.fit_predict(subset)
        df = df[outlier_prediction == 1]

    st.write(f"Cleaned dataframe:")
    st.write(df.head())

    return df


# Function to perform feature engineering
def feature_engineer(df):
    # Create datetime column from Year and Time if they exist
    if "Date" in df.columns and "Time" in df.columns:
        try:
            df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
            df.rename(columns={"datetime": "Purchase_Date"}, inplace=True)
        except Exception as e:
            st.warning(f"Could not create Purchase_Date from Date and Time: {e}")
            df["Purchase_Date"] = pd.to_datetime(df["Date"])  # Fallback to just Date
    elif "Date" in df.columns:
        df["Purchase_Date"] = pd.to_datetime(df["Date"])

    # One-Hot Encoding for Gender column
    if "Gender" in df.columns:
        gender_dummies = pd.get_dummies(df["Gender"], prefix="Gender")
        df = pd.concat([df, gender_dummies], axis=1)

    # Ordinal Encoding for Income column
    if "Income" in df.columns:
        encoder = OrdinalEncoder(categories=[["Low", "Medium", "High"]])
        try:
            df["Income_encoded"] = encoder.fit_transform(df[["Income"]])
        except ValueError:
            st.warning(
                "Income column contains values other than 'Low', 'Medium', 'High'. Skipping encoding."
            )

    # Ordinal Encoding for Feedback column
    if "Feedback" in df.columns:
        encoder = OrdinalEncoder(categories=[["Bad", "Average", "Good", "Excellent"]])
        try:
            df["Feedback_encoded"] = encoder.fit_transform(df[["Feedback"]])
        except ValueError:
            st.warning(
                "Feedback column contains values other than 'Bad', 'Average', 'Good', 'Excellent'. Skipping encoding."
            )

    # Ordinal Encoding for Customer Segment column
    if "Customer_Segment" in df.columns:
        encoder = OrdinalEncoder(categories=[["New", "Regular", "Premium"]])
        try:
            df["Customer_Segment_encoded"] = encoder.fit_transform(
                df[["Customer_Segment"]]
            )
        except ValueError:
            st.warning(
                "Customer_Segment column contains values other than 'New', 'Regular', 'Premium'. Skipping encoding."
            )

    # Encode sentiment if column exists
    if "sentiment" in df.columns:

        def categorise_sentiment(sentiment):
            if sentiment <= -0.5:
                return "Very Negative"
            elif -0.5 < sentiment < 0:
                return "Negative"
            elif 0 <= sentiment <= 0.5:
                return "Neutral"
            elif 0.5 < sentiment <= 1.0:
                return "Positive"
            else:
                return "Very Positive"

        df["sentiment_comment"] = df["sentiment"].apply(categorise_sentiment)
        encoder = OrdinalEncoder(
            categories=[
                ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
            ]
        )
        df["sentiment_comment_encoded"] = encoder.fit_transform(
            df[["sentiment_comment"]]
        )

    return df


# Function to perform RFM analysis
def perform_rfm_analysis(df):
    if (
        "Purchase_Date" in df.columns
        and "Transaction_ID" in df.columns
        and "Total_Amount" in df.columns
        and "Customer_ID" in df.columns
    ):
        reference_date = df["Purchase_Date"].max()
        rfm = df.groupby("Customer_ID").agg(
            {
                "Purchase_Date": lambda x: (reference_date - x.max()).days,  # Recency
                "Transaction_ID": "nunique",  # Frequency
                "Total_Amount": "sum",  # Monetary
            }
        )

        rfm.columns = ["Recency", "Frequency", "Monetary"]

        # Merge with static customer data
        customer_static = df[
            [
                "Customer_ID",
                "Age",
                "Gender_Female",
                "Gender_Male",
                "Income_encoded",
                "Customer_Segment_encoded",
                "Feedback_encoded",
                "sentiment_comment_encoded",
            ]
        ]

        rfm = rfm.merge(customer_static, on="Customer_ID", how="left")
        rfm = rfm.set_index("Customer_ID")

        st.write("RFM Analysis Results Snippet:")
        st.write(rfm.head())

        st.write("RFM Analysis Summary Statistics:")
        st.write(rfm.describe())

        # Scale RFM metrics
        cols_to_scale = ["Recency", "Frequency", "Monetary"]

        # Initialize the scaler
        global scaler 
        scaler = StandardScaler()

        # Fit and transform only the selected columns
        rfm_scaled_part = scaler.fit_transform(rfm[cols_to_scale])

        # Convert back to DataFrame
        rfm_scaled_part = pd.DataFrame(
            rfm_scaled_part, columns=cols_to_scale, index=rfm.index
        )

        # Now, combine the scaled columns with the rest of the dataset
        rfm = rfm.copy()
        rfm[cols_to_scale] = rfm_scaled_part

        # Convert booleans to integers if needed
        for col in ["Gender_Female", "Gender_Male"]:
            if col in rfm.columns:
                rfm[col] = rfm[col].astype(int)

        # Display
        st.write("Scaled output")
        st.write(rfm.head())

        return rfm
    else:
        st.warning("Required columns for RFM analysis are missing.")
        return pd.DataFrame()


# Function to perform clustering
def perform_clustering(rfm_df):
    features_for_clustering_options = [
        "Recency",
        "Frequency",
        "Monetary",
        "Age",
        "Income_encoded",
        "Gender_Female",
        "Gender_Male",
        "Feedback_encoded",
        "sentiment_comment_encoded",
    ]
    features_for_clustering = [
        col for col in features_for_clustering_options if col in rfm_df.columns
    ]

    if features_for_clustering:
        X = rfm_df[features_for_clustering].values
        # Using BIRCH for clustering
        optimal_k = 3  # Assuming 3 clusters
        birch = Birch(threshold=0.5, n_clusters=optimal_k)
        labels_birch = birch.fit_predict(X)
        rfm_df["Cluster"] = labels_birch

        cols_to_unscale = ["Recency", "Frequency", "Monetary"]

        # Fit and transform only the selected columns
        global scaler
        rfm_unscaled_part = scaler.inverse_transform(rfm_df[cols_to_unscale])

        # Convert back to DataFrame
        rfm_df[cols_to_unscale] = pd.DataFrame(
            rfm_unscaled_part, columns=cols_to_unscale, index=rfm_df.index
        )


        st.write("Clustering Results:")
        st.write(rfm_df.head())

        # Naming clusters based on RFM + Demographics + Sentiment
        # Compute mean RFM values per cluster
        rfm_avg = (
            rfm_df.groupby("Cluster")[
                [
                    "Recency",
                    "Frequency",
                    "Monetary",
                    "Age",
                    "Gender_Female",
                    "Gender_Male",
                    "Income_encoded",
                    "Feedback_encoded",
                    "sentiment_comment_encoded",
                ]
            ]
            .mean()
            .round(2)
        )

        # Display the average RFM + Demographics + Sentiment values per segment
        st.write("Average RFM + Demographic + Sentiment Values Per Cluster:")
        st.write(rfm_avg)

        return rfm_df
    else:
        st.warning("No suitable features found for clustering.")
        return rfm_df


# Function to generate and display visualisations
def generate_visualisations(df, df_cluster):
    st.subheader("Data Visualisations")

    st.subheader("Exploratory Data Analysis (EDA)")

    # Monthly Revenue Trend
    st.markdown("Monthly Revenue Trend")
    if "Purchase_Date" in df.columns and "Total_Amount" in df.columns:
        df["YearMonth"] = df["Purchase_Date"].dt.to_period("M")
        monthly_revenue = df.groupby("YearMonth")["Total_Amount"].sum()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            x=monthly_revenue.index.astype(str),
            y=monthly_revenue.values,
            marker="o",
            ax=ax,
        )
        ax.tick_params(axis="x", rotation=45)
        ax.set_title("Monthly Revenue Trend")
        ax.set_xlabel("Month")
        ax.set_ylabel("Total Revenue (£)")
        ax.grid(True)
        st.pyplot(fig)

    st.markdown('Top 10 Cities by Revenue')
    # Top 10 Cities by Revenue
    if "City" in df.columns and "Total_Amount" in df.columns:
        city_revenue = (
            df.groupby("City")["Total_Amount"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            x=city_revenue.index, y=city_revenue.values, palette="coolwarm", ax=ax
        )
        ax.tick_params(axis="x", rotation=45)
        ax.set_title("Top 10 Cities by Revenue")
        ax.set_xlabel("City")
        ax.set_ylabel("Total Revenue")
        st.pyplot(fig)

    # Customer Spending Distribution
    st.markdown('Customer Spending Distribution')
    if "Customer_ID" in df.columns and "Total_Amount" in df.columns:
        customer_spend = df.groupby("Customer_ID")["Total_Amount"].sum()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(customer_spend, bins=50, kde=True, ax=ax)
        ax.set_title("Customer Spending Distribution")
        ax.set_xlabel("Total Spend per Customer (£)")
        ax.set_ylabel("Customer Count")
        ax.set_xlim(0, customer_spend.quantile(0.99))
        st.pyplot(fig)

    # Visualisations by Cluster (if clustering was successful)
    st.subheader('Post-Hoc Visualisations')
    if "Cluster" in df_cluster.columns:
        # Age Distribution by Cluster
        st.markdown('Age Distribution by Cluster')
        if "Age" in df_cluster.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x="Cluster", y="Age", data=df_cluster, ax=ax)
            ax.set_title("Age Distribution by Cluster")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Age")
            st.pyplot(fig)

        # Gender Distribution by Cluster
        st.markdown('Gender Distribution by Cluster')
        if "Gender" in df_cluster.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x="Cluster", hue="Gender", data=df_cluster, ax=ax)
            ax.set_title("Gender Distribution by Cluster")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        # Income Distribution by Cluster
        st.markdown('Income Distribution by Cluster')
        if "Income" in df_cluster.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x="Cluster", hue="Income", data=df_cluster, ax=ax)
            ax.set_title("Income Distribution by Cluster")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        # Product Category Distribution by Cluster
        st.markdown('Product Category Distribution by Cluster')
        if "Product_Category" in df_cluster.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x="Cluster", hue="Product_Category", data=df_cluster, ax=ax)
            ax.set_title("Product Category Distribution by Cluster")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        # Ratings Distribution by Cluster
        st.markdown('Ratings Distribution by Cluster')
        if "Ratings" in df_cluster.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x="Cluster", hue="Ratings", data=df_cluster, ax=ax)
            ax.set_title("Ratings Distribution by Cluster")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        # Sentiment Distribution by Cluster
        st.markdown('Sentiment Distribution by Cluster')
        if "sentiment_comment" in df_cluster.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x="Cluster", hue="sentiment_comment", data=df_cluster, ax=ax)
            ax.set_title("Sentiment Distribution by Cluster")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        # Distribution of Spend by Cluster (Pie Chart)
        st.markdown('Distribution of Spend by Cluster (Pie Chart)')
        if "Total_Amount" in df_cluster.columns:
            cluster_spend = df_cluster.groupby("Cluster")["Total_Amount"].sum()
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.pie(cluster_spend.values, labels=cluster_spend.index, autopct="%1.1f%%")
            ax.set_title("Distribution of Spend by Cluster")
            ax.axis("equal")
            st.pyplot(fig)

        # Total Amount by Cluster (Box Plot)
        st.markdown('Total Amount by Cluster (Box Plot)')
        if "Total_Amount" in df_cluster.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x="Cluster", y="Total_Amount", data=df_cluster, ax=ax)
            ax.set_title("Total Amount by Cluster")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Total Amount (£)")
            st.pyplot(fig)

        # Total Purchases by Cluster (Box Plot)
        st.markdown('Total Purchases by Cluster (Box Plot)')
        if "Total_Purchases" in df_cluster.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x="Cluster", y="Total_Purchases", data=df_cluster, ax=ax)
            ax.set_title("Total Purchases by Cluster")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Total Purchases")
            st.pyplot(fig)

    # Ratings Distribution by Product Category
    st.markdown('Ratings Distribution by Product Category')
    if "Product_Category" in df.columns and "Ratings" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x="Product_Category", hue="Ratings", data=df, ax=ax)
        ax.set_title("Ratings Distribution by Product Category")
        ax.set_xlabel("Product Category")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # Top 10 Product Brands by Average Ratings
    st.markdown('Top 10 Product Brands by Average Ratings')
    if "Product_Brand" in df.columns and "Ratings" in df.columns:
        top_brands = (
            df.groupby("Product_Brand")["Ratings"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_brands.index, y=top_brands.values, ax=ax)
        ax.set_title("Top 10 Product Brands by Average Ratings")
        ax.set_xlabel("Product Brand")
        ax.set_ylabel("Average Ratings")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)


# Streamlit application
st.title("Retail Data Analysis Customer Segmentation")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")

        # Display raw data
        st.subheader("Raw Data")
        st.dataframe(df.head())

        # Display summary statistics for the entire dataset
        st.subheader("Dataset Summary Statistics")
        # Select only numeric columns for describe()
        numeric_cols = df.select_dtypes(include=np.number)
        if not numeric_cols.empty:
            st.write(numeric_cols.describe())
        else:
            st.write("No numeric columns to display summary statistics.")

        # Data Cleaning and Preprocessing
        with st.spinner("Performing data cleaning and preprocessing..."):
            df_processed = preprocess_data(df.copy())
        st.success("Data cleaning and preprocessing complete.")

        # Feature Engineering
        with st.spinner("Performing feature engineering..."):
            df_engineered = feature_engineer(df_processed.copy())
        st.success("Feature engineering complete.")

        # RFM Analysis
        with st.spinner("Performing RFM analysis..."):
            rfm_df = perform_rfm_analysis(df_engineered.copy())
        st.success("RFM analysis complete.")

        # Clustering
        if not rfm_df.empty:
            with st.spinner("Performing clustering..."):
                rfm_clustered = perform_clustering(rfm_df.copy())
            st.success("Clustering complete.")

            rfm_clustered = rfm_clustered.reset_index()

            # Merge cluster labels back to the main dataframe
            if (
                "Cluster" in rfm_clustered.columns
                and "Customer_ID" in df_engineered.columns
            ):
                df_cluster = df_engineered.merge(
                    rfm_clustered[
                        [
                            "Customer_ID",
                            "Cluster",
                        ]
                    ],
                    on="Customer_ID",
                    how="left",
                )
            else:
                df_cluster = df_engineered
                st.warning(
                    "Could not merge cluster labels. Visualisations by cluster will be skipped."
                )

            # Generate Visualisations
            generate_visualisations(df_engineered, df_cluster)
        else:
            st.warning(
                "Skipping clustering and cluster-based visualisations due to issues in RFM analysis."
            )

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

# Prompt to upload another file
if uploaded_file is not None:
    if st.button("Upload Another File"):
        st.experimental_rerun()
