from typing import Literal, Union


def nlp_score(
        model,
        vectorizer, 
        wb_path, 
        input_sheet_name = "Sheet1" , 
        col_input='portuguese', 
        print_index = True,
        inplace=True,
        close_after_scored = False,
        plot_confusion = False,
        y_name = 'usefulness'
        ):
    """
    !!! signature function
    # only works for Excel file for now
    
    
    Scores text data using a specified NLP model and vectorizer, and writes the results to an Excel workbook.

    This function reads data from a specified sheet in an Excel workbook, applies an NLP model to a specified column, 
    and writes the predictions back to the workbook. It optionally plots a confusion matrix for the scoring.

    Parameters:
    model: Trained machine learning model used for prediction.
    vectorizer: Text vectorizer used to transform the text data.
    wb_path: Path to the Excel workbook or an xlwings Book object.
    input_sheet_name (str, optional): Name of the sheet in the workbook to read data from. Defaults to "Sheet1".
    col_input (str, optional): Name of the column in the sheet to apply the model to. Defaults to 'portuguese'.
    print_index (bool, optional): Whether to print the DataFrame index in the Excel sheet. Defaults to True.
    inplace (bool, optional): If True, writes results in the same sheet; otherwise, creates a new sheet. Defaults to True.
    close_after_scored (bool, optional): Whether to close the workbook after processing. Defaults to False.
    plot_confusion (bool, optional): If True, plots a confusion matrix of the results. Defaults to True.
    y_name (str, optional): The name of the target column for plotting the confusion matrix. Defaults to 'usefulness'.

    Notes:
    - This function currently only works with Excel files.
    - The workbook is saved after processing.
    - If `plot_confusion` is True, the function attempts to plot a confusion matrix and may fail silently with a ValueError 
      if the necessary data is not available.
    - The function requires the 'xlwings' and 'pandas' libraries.
    - Custom library 'lib02_dataframe' is used for reading Excel data.
    
    
    """
    import xlwings as xw
    import pandas as pd
    from pathlib import Path
    import sys
    sys.path.append(r"C:\Users\Heng2020\OneDrive\Python MyLib")
    import lib02_dataframe as ds
    # Open the workbook
    
    if isinstance(wb_path, (str,Path)):
        wb = xw.Book(wb_path)
    elif isinstance(wb_path, (str,xw.Book)):
        wb = wb_path
        
    ws_names = [sheet.name for sheet in wb.sheets]

    # Read data from the input sheet
    input_sheet = wb.sheets[input_sheet_name]
    df = ds.pd_read_excel(df_path,input_sheet_name,header = 1)

    # Apply the model to the specified column
    prediction = nlp_predict(df,model,vectorizer, col_input= col_input,inplace=False)
    
    labels = ['Not Useful','Already Knew','Normal','Useful']
    
    if plot_confusion:
        try:
            plot_confusion_matrix(prediction[y_name], prediction['prediction'], 'Scoring metrics',labels,max_percentile=90)
        except ValueError:
            pass

    # Decide whether to replace data in the same sheet or in a new sheet
    if inplace:
        scored_sheet = input_sheet
    else:
        ws_scored_name = input_sheet_name + "_scored"
        if ws_scored_name in ws_names:
            scored_sheet = wb.sheets[ws_scored_name]
        else:
            scored_sheet = wb.sheets.add(ws_scored_name)

    # Paste the scored result into the workbook
    scored_sheet.range('A1').options(index=print_index).value = prediction

    # Save the workbook
    wb.save()
    
    if close_after_scored:
        # Optionally, close the workbook
        wb.close()

def plot_confusion_matrix(y_true, y_pred, title, labels=None,max_percentile = 80, adjusted = False, y_accept = None):
    # Generate the confusion matrix
    # if adjusted = True use adjusted confusion matrix
    import numpy as np
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    
    if adjusted:
        cm = confusion_matrix_adj(y_true, y_pred,y_accept, labels=labels)
    else:
        cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Calculate the 80th percentile of the values in the confusion matrix
    vmax = np.percentile(cm, max_percentile)

    plt.figure(figsize=(8, 6))
    
    # Determine xticklabels and yticklabels based on whether labels are provided
    if labels is None:
        xticklabels = y_true.unique()
        yticklabels = y_true.unique()
    else:
        xticklabels = labels
        yticklabels = labels

    # Create the heatmap
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', vmax=vmax, xticklabels=xticklabels, yticklabels=yticklabels)

    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def confusion_matrix_adj(y_true, y_accept, y_pred, labels=None):
    from sklearn.metrics import confusion_matrix
    import numpy as np
    # FIX seems like the function is still not correct 
    # please do the recon before using this function
    # unusable for now
    
    """
    Compute a confusion matrix with adjustments.

    Parameters:
    y_true: Array-like of true class labels.
    y_accept: Array-like of acceptable class labels.
    y_pred: Array-like of predicted class labels.
    labels: List of label names corresponding to the classes (optional).

    Returns:
    Confusion matrix as a 2D array.
    """
    # Adjust predictions
    adjusted_pred = []
    for true, accept, pred in zip(y_true, y_accept, y_pred):
        if pred == true or pred == accept:
            adjusted_pred.append(pred)
        # elif (accept is None) or ((accept is np.nan)):
        #     adjusted_pred.append(true)
        else:
            adjusted_pred.append(true)  # Considered as predicted 'true', actual 'true'

    # Compute confusion matrix
    return confusion_matrix(y_true, adjusted_pred, labels=labels)


def nlp_predict(data,model,tfidf_vectorizer,col_input = 'portuguese_lemma', inplace = True):
    import pandas as pd
    # vocab03 = tfidf_vectorizer.vocabulary_

    if isinstance(data, pd.Series):
        data_in = data.copy()
    elif isinstance(data, pd.DataFrame):
        data_in = data[col_input]
        
    data_tfidf = tfidf_vectorizer.transform(data_in)
    prediction = model.predict(data_tfidf)
    
    # vocab04 = tfidf_vectorizer.vocabulary_
    if isinstance(data, pd.Series):
        out_df = pd.DataFrame({'sentence':data, 'prediction':prediction})
        return out_df
    
    elif isinstance(data, pd.DataFrame):
        if inplace:
            data['prediction'] = prediction
            return data
        else:
            out_data = data.copy()
            out_data['prediction'] = prediction
            return out_data
    
    
    return out_df

# ########################################### Add function in Apr 13,24 ###########################################
def detect_language(input_text, 
                    return_as: Literal["full_name","2_chr_code","3_chr_code","langcodes_obj"] = "full_name"):
    from langdetect import detect
    import langcodes
    # medium tested
    # wrote < 30 min(with testing)
    if isinstance(input_text, str):
    # assume only 1d list
        try:
            # Detect the language of the text
            # language_code is 2 character code
            lang_code_2chr = detect(input_text)
            language_obj = langcodes.get(lang_code_2chr)
            language_name = language_obj.display_name()
            lang_code_3chr = language_obj.to_alpha3()


            if return_as in ["full_name"]:
                ans = language_name
            elif return_as in ["2_chr_code"]:
                ans = lang_code_2chr
            elif return_as in ["3_chr_code"]:
                ans = lang_code_3chr
            elif return_as in ["langcodes_obj"]:
                ans = language_obj

            return ans
        except Exception as e:
            err_str = f"Language detection failed: {str(e)}"
            return False
        
    elif isinstance(input_text, list):
        out_list = []
        for text in input_text:
            detect_lang = detect_language(text, return_as = return_as)
            out_list.append(detect_lang)
        return out_list

def closest_language(misspelled_language):
    
    from fuzzywuzzy import process
    import pycountry
    # Get a list of all language names
    language_names = [lang.name for lang in pycountry.languages if hasattr(lang, 'name')]

    # Use fuzzy matching to find the closest match
    closest_match = process.extractOne(misspelled_language, language_names)
    return closest_match[0] if closest_match else None

def closest_language_obj(misspelled_language):
    
    """
    Find the closest matching language object for a potentially misspelled language code.
    
    Parameters:
    -----------
    misspelled_language : str
        The potentially misspelled language code.
    
    Returns:
    --------
    langcodes.Language
        A language object representing the closest matching language.
    
    Notes:
    ------
    - This function uses the 'langcodes' library to find the closest matching language object
      for a potentially misspelled language code.
    - It can be useful for language code correction or normalization.
    
    Example:
    --------
    >>> closest_language_obj("englsh")
    <Language('en', 'English')>
    >>> closest_language_obj("español")
    <Language('es', 'Spanish')>
    
    """
    
    
    from langcodes import Language
    correct_language = closest_language(misspelled_language)
    return Language.find(correct_language)

#  -------------------------------------------- Add function in Apr 13,24 -----------------------------------------------------


def concat_vocab_df(df1,df2, plot = True):
    # imported from "C:\Users\Heng2020\OneDrive\Python NLP\NLP 08_VocabList\VocatList_func01.py"

    """
    

    Parameters
    ----------
    df1 & df2 are assumed to be the output from: nlp_word_freq_all

    Returns
    -------
    out_df: pd.Dataframe

    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    def flatten(list_of_lists):
        """Flatten a 2D list to 1D"""
        return [item for sublist in list_of_lists for item in sublist]


    word_raw_freq1 = df1[['word','count_word']]
    word_stem_freq1 = df1[['infinitive','count_infinitive']].drop_duplicates(subset = ['infinitive'])

    word_raw_freq2 = df2[['word','count_word']]
    word_stem_freq2 = df2[['infinitive','count_infinitive']].drop_duplicates(subset = ['infinitive'])

    word_raw_freq = pd.concat([word_raw_freq1,word_raw_freq2])
    word_raw_freq = word_raw_freq.groupby('word')['count_word'].sum().reset_index()

    word_stem_freq = pd.concat([word_stem_freq1,word_stem_freq2])
    word_stem_freq = word_stem_freq.groupby('infinitive')['count_infinitive'].sum().reset_index()

    df_combine = pd.concat([df1,df2],ignore_index=True)
    
    grouped_sen = df_combine.groupby('word')['sentence'].agg(flatten).reset_index()

    agg_word = df_combine.groupby(['word','type'],sort=False)[['count_word','n_doc_word']].sum().reset_index()

    agg_infinitive = df_combine.groupby(['infinitive','type'],sort=False)[['count_infinitive','n_sentence']].sum().reset_index()

    unique_word = df_combine[['word','infinitive','type']].drop_duplicates(subset = ['word','infinitive','type'] )

    temp_df = unique_word.merge(agg_word,on = ['word','type'],how = 'left')
    temp_df = temp_df.merge(agg_infinitive, on = ['infinitive','type'], how = 'left')

    n_doc_infinitive = temp_df.groupby(['infinitive'],sort=False)['n_doc_word'].max().reset_index()
    n_doc_infinitive = n_doc_infinitive.rename(columns = {'n_doc_word':'n_doc_infinitive'})
    out_df = temp_df.merge(n_doc_infinitive, on = ['infinitive'])
    out_df = out_df.merge(grouped_sen, on = ['word'], how = 'left')
    
    out_df = out_df[['word','infinitive','type','count_word','count_infinitive','n_doc_word','n_doc_infinitive','n_sentence','sentence']]
    
    # Categorize the values in 'count_infinitive'
    df_freq = pd.DataFrame(out_df['count_infinitive'].apply(lambda row: '>= 5' if row >= 5 else row))

    if plot:
        category_order = [1,2,3,4,'>= 5']
        ax = sns.countplot(x = 'count_infinitive',data = df_freq, order = category_order);
        plt.title('Infinitive Frequency Occurrence ');
        plt.xlabel('Freq');
        plt.ylabel('n_words');
        
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width() / 2.,  # x position (center of the bar)
                    p.get_height(),  # y position (top of the bar)
                    f'{int(p.get_height())}',  # the text (count)
                    ha='center',  # horizontal alignment
                    va='bottom')  # vertical alignment
        
        # Show the plot
        plt.show()
    
    return out_df



def nlp_word_freq_all(data_path,model,plot = True):
    # imported from "C:\Users\Heng2020\OneDrive\Python NLP\NLP 08_VocabList\VocatList_func01.py"
    import spacy
    from collections import Counter
    import matplotlib.pyplot as plt
    import seaborn as sns
    # middium tested
    """
    signature_function
    
    assume that data_path is the string of my generated csv from srt
    must have the column called sentence
    all - means include every word not just unique infinitive 

    Returns
    -------
    None.

    """
    import warnings
    import pandas as pd
    # to slient to warning creating from sns ploting
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    excel_extensions = (".xlsx", ".xlsm", ".xlsb")
    
    doc: spacy.tokens.doc.Doc
    input_sentence: pd.DataFrame
    
    if any(str(data_path).endswith(ext) for ext in excel_extensions):
        input_sentence = pd.read_excel(data_path)[['sentence']]
        single_text = ' '.join(input_sentence['sentence'])
        
    elif str(data_path).endswith(".txt"):
        
        with open(str(data_path), 'r', encoding='utf-8') as file:
            text_list = file.readlines() 
            text_list2 = []
            
            for sentence in text_list:
                if not_empty_string(sentence):
                    if not sentence.endswith("."):
                        sentence_with_dot = sentence.replace("\n", "") + "."
                    else:
                        sentence_with_dot = sentence.replace("\n", "") 
                    text_list2.append(sentence_with_dot)
                    
            
            single_text = ' '.join(text_list2)
    
    
    doc = model(single_text)
    
    # Counter for word frequencies
    word_raw_freq = Counter()

    # Iterate over the tokens
    for token in doc:
        if not token.is_punct and not token.is_stop:
            word_raw_freq[token.text.lower()] += 1
        
        
    word_stem_freq = Counter()

    for token in doc:
        if not token.is_punct and not token.is_stop:
            word_stem_freq[token.lemma_.lower()] += 1
    
    data_vocab = [
        {
            "sentence": sent.text,
            "word": token.text.lower(),
            "infinitive": token.lemma_.lower(),
            "type": token.pos_,

        }
        for sent in doc.sents 
        for token in sent if (token.pos_ in ['VERB', 'NOUN','ADJ','ADV']) and (not token.is_punct) and (not token.is_stop)
    ]
    model
    df_ungroup = pd.DataFrame(data_vocab)
    df_no_duplicate = df_ungroup.drop_duplicates(subset = ['word'])
    

    grouped_sen = df_ungroup.groupby('word',sort=False)['sentence'].agg(list).reset_index()


    df_group = grouped_sen.merge(df_no_duplicate[['word','infinitive','type']],on = 'word', how = 'inner')
    df_group = df_group[['word','infinitive','type','sentence']]

    df_group['count_word'] = df_group['word'].map(word_raw_freq)
    df_group['count_infinitive'] = df_group['infinitive'].map(word_stem_freq)
    df_group['n_sentence'] = df_group['sentence'].apply(len)
    
    # Categorize the values in 'count_infinitive'
    df_freq = pd.DataFrame(df_group['count_infinitive'].apply(lambda row: '>= 5' if row >= 5 else row))
    
    if isinstance(data_path,str):
        if plot:
            category_order = [1,2,3,4,'>= 5']
            ax = sns.countplot(x = 'count_infinitive',data = df_freq, order = category_order,);
            plt.title('Infinitive Frequency Occurrence ');
            plt.xlabel('Freq');
            plt.ylabel('n_words');
            
            for p in ax.patches:
                ax.text(p.get_x() + p.get_width() / 2.,  # x position (center of the bar)
                        p.get_height(),  # y position (top of the bar)
                        f'{int(p.get_height())}',  # the text (count)
                        ha='center',  # horizontal alignment
                        va='bottom')  # vertical alignment
            
            # Show the plot
            plt.show()
    
    df_group['n_doc_word'] = 1
    df_group['n_doc_infinitive'] = 1
    
    return df_group
    

def nlp_make_tfidf_matrix(X,text_col, ngram_range =(1,1),stop_words = [], max_df = 0.7):
    # imported from "C:\Users\Heng2020\OneDrive\Python NLP\NLP 05_UsefulSenLabel\sen_useful_GPT01.py"
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    if isinstance(X,pd.Series):
        X_in = X.copy()
    elif isinstance(X,pd.DataFrame):
        X_in = X[text_col]
    else:
        raise Exception("X should only pd.Series or pd.DataFrame as of now")
    
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words,ngram_range=ngram_range,max_df=max_df)
    X_tfidf = tfidf_vectorizer.fit_transform(X_in)
    X_out_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=X.index)
    
    return [X_out_df,tfidf_vectorizer]

def lemmatize(text,model):
    # imported from "C:\Users\Heng2020\OneDrive\Python NLP\NLP 05_UsefulSenLabel\sen_useful_GPT01.py"
    doc = model(text)
    lemmatized = " ".join([token.lemma_ for token in doc])
    return lemmatized

def nlp_predict(data,model,tfidf_vectorizer,col_input = 'portuguese_lemma', inplace = True):
    # imported from "C:\Users\Heng2020\OneDrive\Python NLP\NLP 05_UsefulSenLabel\sen_useful_GPT01.py"
    import pandas as pd
    # vocab03 = tfidf_vectorizer.vocabulary_

    if isinstance(data, pd.Series):
        data_in = data.copy()
    elif isinstance(data, pd.DataFrame):
        data_in = data[col_input]
        
    data_tfidf = tfidf_vectorizer.transform(data_in)
    prediction = model.predict(data_tfidf)
    
    # vocab04 = tfidf_vectorizer.vocabulary_
    if isinstance(data, pd.Series):
        out_df = pd.DataFrame({'sentence':data, 'prediction':prediction})
        return out_df
    
    elif isinstance(data, pd.DataFrame):
        if inplace:
            data['prediction'] = prediction
            return data
        else:
            out_data = data.copy()
            out_data['prediction'] = prediction
            return out_data
    
    
    return out_df


def nlp_predict_prob(data,model,tfidf_vectorizer,col_input = 'portuguese_lemma', inplace = True):
    # imported from "C:\Users\Heng2020\OneDrive\Python NLP\NLP 05_UsefulSenLabel\sen_useful_GPT01.py"
    # !!! TOFIX when inplace = True, it still doesn't change the original df
    # doesn't seem to be useful if the prob predict is very low and I want to flag it as not sure
    
    import pandas as pd
    # vocab03 = tfidf_vectorizer.vocabulary_

    if isinstance(data, pd.Series):
        data_in = data.copy()
    elif isinstance(data, pd.DataFrame):
        data_in = data[col_input]
        
    data_tfidf = tfidf_vectorizer.transform(data_in)
    prediction = model.predict_proba(data_tfidf)
    
    labels = model.classes_.tolist()
    
    prob_df = pd.DataFrame(prediction, columns=[label + '_prob' for label in labels]).set_index(data.index)
    out_df = nlp_predict(data, model, tfidf_vectorizer,col_input,inplace)
    
    # vocab04 = tfidf_vectorizer.vocabulary_
    if isinstance(data, pd.Series):
        data = pd.concat([out_df,prob_df], axis = 1)
        return out_df
    
    elif isinstance(data, pd.DataFrame):
        if inplace:
            data = pd.concat([out_df,prob_df ], axis = 1)
            return data
        else:
            out_data = pd.concat([out_df,prob_df ], axis = 1)
            return out_data

