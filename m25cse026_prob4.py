import pandas as pd
import matplotlib.pyplot as plt
import io
import requests
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import metrics

if "SSLKEYLOGFILE" in os.environ:
    del os.environ["SSLKEYLOGFILE"]

def main():
    # Using the BBC dataset link as it's reliable for news classification
    data_url = "https://storage.googleapis.com/learning-datasets/bbc-text.csv"
    
    print("Step 1: Fetching the dataset...")
    try:
        # verify=False bypasses local SSL issues; we ignore the warnings for a cleaner terminal
        requests.packages.urllib3.disable_warnings()
        resp = requests.get(data_url, verify=False, timeout=10)
        resp.raise_for_status()
        
        # Load csv into a pandas dataframe
        raw_df = pd.read_csv(io.StringIO(resp.text))
    except Exception as err:
        print(f"Oops, something went wrong with the download: {err}")
        return

    # We only care about sports and politics for this assignment
    # Filtering the dataframe to keep only relevant rows
    target_classes = ['sport', 'politics']
    filtered_df = raw_df[raw_df['category'].isin(target_classes)]
    
    # Mapping text labels to numbers: 0 for sports, 1 for politics
    text_data = filtered_df['text']
    labels = filtered_df['category'].map({'sport': 0, 'politics': 1})
    
    # Splitting into 80% training and 20% testing
    # Stratify ensures we have an even mix of classes in both sets
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        text_data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Data Loaded: {len(filtered_df)} total samples used.")

    # Step 2: Feature Extraction using TF-IDF
    # We remove 'english' stop words like 'the', 'is', etc. and limit to 3000 features
    tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
    X_train = tfidf.fit_transform(X_train_text)
    X_test = tfidf.transform(X_test_text)

    # Step 3: Comparing 3 different ML models
    my_models = {
        "NB": MultinomialNB(),
        "LogReg": LogisticRegression(),
        "LinearSVM": LinearSVC(dual='auto')
    }

    results_map = {}
    print("\n--- Training & Testing Results ---")
    for name, clf in my_models.items():
        # Train the model
        clf.fit(X_train, y_train)
        
        # Test the model
        predictions = clf.predict(X_test)
        
        # Record accuracy
        acc_score = metrics.accuracy_score(y_test, predictions)
        results_map[name] = acc_score
        print(f"Model: {name} | Accuracy: {acc_score:.4f}")

    # Step 4: Generating the Comparison Bar Chart
    plt.figure(figsize=(9, 6))
    colors = ['skyblue', 'lightgreen', 'orange']
    plt.bar(results_map.keys(), results_map.values(), color=colors)
    
    plt.ylabel('Accuracy Percentage')
    plt.title('Performance Comparison of 3 ML Models')
    plt.ylim(0, 1.1) # Leave some space at the top for labels
    
    # Adding accuracy text on top of each bar
    for i, val in enumerate(results_map.values()):
        plt.text(i, val + 0.02, f"{val:.2%}", ha='center', weight='bold')

    plt.savefig('comparison_chart.png')
    print("\nSuccess! 'comparison_chart.png' has been saved")

if __name__ == "__main__":
    main()