def evaluate_model_performance(new_data_path, tfidf_vectorizer, nb_model, lr_model):
    try:
        # Load new data
        new_dataset = pd.read_csv(new_data_path)
        print("New data loaded successfully!")
        print("Columns in new data:", new_dataset.columns)
        print("First few rows of new data:\n", new_dataset.head())

        # Check if required columns exist
        if 'query' not in new_dataset.columns or 'lan_code' not in new_dataset.columns:
            raise KeyError("The dataset must contain 'query' and 'lan_code' columns.")

        new_texts = new_dataset['query'].astype(str)
        new_labels = new_dataset['lan_code']

        # Transform new data using the saved TF-IDF vectorizer
        new_tfidf = tfidf_vectorizer.transform(new_texts)

        # Evaluate Naive Bayes model
        nb_predictions = nb_model.predict(new_tfidf)
        print("Naive Bayes Performance:\n", classification_report(new_labels, nb_predictions))

        # Evaluate Logistic Regression model
        lr_predictions = lr_model.predict(new_tfidf)
        print("Logistic Regression Performance:\n", classification_report(new_labels, lr_predictions))

        # Plot confusion matrix for Logistic Regression
        cm = confusion_matrix(new_labels, lr_predictions, labels=lr_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lr_model.classes_)
        disp.plot()
        plt.savefig("evaluations/confusion_matrix.png")

    except FileNotFoundError:
        print(f"Error: File not found at '{new_data_path}'. Please check the file path.")
    except KeyError as e:
        print(f"Error: Column '{e}' not found in the dataset. Please check the column names.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{new_data_path}' is empty or does not contain valid CSV data.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")