# from
# "C:\Users\Heng2020\OneDrive\D_Code\Python\Python NLP\NLP 01\NLP 08_VocabList\VocatList_func01.py"
import warnings
import os
from typing import Union
from pathlib import Path

warnings.filterwarnings("ignore", message="'has_mps' is deprecated") # Suppress INFO and WARNING from spacy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING from spacy
import spacy
def nlp_word_freq_all(
        data_path: Union[str,Path]
        ,model: spacy.language
        ,plot:bool = True):
    # middium tested
    
    # spacy 3.7.2 needs 
    # to have pydantic 1.10.12
    
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
    import py_string_tool as pst
    from collections import Counter
    import seaborn as sns
    import matplotlib.pyplot as plt

    # to slient to warning creating from sns ploting
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    excel_extensions = (".xlsx", ".xlsm", ".xlsb")
    
    doc: spacy.tokens.doc.Doc
    input_sentence: pd.DataFrame
    
    if any(str(data_path).endswith(ext) for ext in excel_extensions):
        input_sentence = pd.read_excel(data_path)[['sentence']]
        # there's an error when it's nan
        input_sentence['sentence'] = input_sentence['sentence'].fillna("")
        
        single_text = ' '.join(input_sentence['sentence'])
        
    elif str(data_path).endswith(".txt"):
        
        with open(str(data_path), 'r', encoding='utf-8') as file:
            text_list = file.readlines() 
            text_list2 = []
            
            for sentence in text_list:
                if pst.not_empty_string(sentence):
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
    

def nlp_word_freq_infinitive():
    pass

def concat_vocab_df(df1,df2, plot = True):
    import seaborn as sns
    import matplotlib.pyplot as plt

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
    import python_wizard.pw_list as pwl
    


    word_raw_freq1 = df1[['word','count_word']]
    word_stem_freq1 = df1[['infinitive','count_infinitive']].drop_duplicates(subset = ['infinitive'])

    word_raw_freq2 = df2[['word','count_word']]
    word_stem_freq2 = df2[['infinitive','count_infinitive']].drop_duplicates(subset = ['infinitive'])

    word_raw_freq = pd.concat([word_raw_freq1,word_raw_freq2])
    word_raw_freq = word_raw_freq.groupby('word')['count_word'].sum().reset_index()

    word_stem_freq = pd.concat([word_stem_freq1,word_stem_freq2])
    word_stem_freq = word_stem_freq.groupby('infinitive')['count_infinitive'].sum().reset_index()

    df_combine = pd.concat([df1,df2],ignore_index=True)
    
    grouped_sen = df_combine.groupby('word')['sentence'].agg(pwl.flatten).reset_index()

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
