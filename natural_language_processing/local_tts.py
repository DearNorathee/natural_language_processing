from typing import Literal,Union,List
import pandas as pd
from pathlib import Path

#%%
def lang_voice_name():
    """
    return the list of avaliable tts in your local machine

    Returns
    -------
    list.

    """
    import pyttsx3
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    out_list = []
    for voice in voices:
        out_list.append(voice.name)
    return out_list

def engine_by_lang(language):
    # must spell language correctly, the upper vs lower case doesn't matter
    voice_name = lang_voice_name()
    import pyttsx3
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    # still can't tell the gender of the engine
    for i, name in enumerate(voice_name):
        if language.lower() in name.lower():
            used_id = voices[i].id
            engine.setProperty('voice', used_id) 
            
            return engine
    # can't find the language in the system
    return False

def create_audio(text:str,language:str = "auto",filename = "", output_folder = "",playsound = False):
    # doesn't seem to work in VSCode but it works fine in Spyder
    
    if language in ["auto"]:
        language_in = detect_language(text)
    else:
        language_in = language
        
    engine = engine_by_lang(language_in)
    if engine is False:
        raise Exception(f"Doesn't have '{language_in}' in the keyboard system. Please download this language in your keyboard. ")
    
    if isinstance(text, list):
        
        if isinstance(filename, list):
            if len(text) != len(filename):
                raise Exception(f"The length of text is: {len(text)}. The length of filename is {len(filename)}. Make sure they have the same length.")
            
        n_digit = len(str(len(text)))
            
        if filename in [""]:
            filename_in = list(text)
        else:
            filename_in = list(filename)
        
        out_filenames = []
        for i, curr_filename in enumerate(filename_in):
            out_filename = f"{i+1:0{n_digit}}_{curr_filename}"
            out_filenames.append(out_filename)

            
        for i, curr_text in enumerate(text):
            create_audio(
                curr_text,
                language = language,
                filename = out_filenames[i],
                output_folder = output_folder,
                playsound = playsound,
                
                )
        return
    
    
    if playsound:
        engine.say(text)
        
        
    engine.runAndWait()
        # engine.stop()
    if filename + str(output_folder) != "":
        if str(output_folder) == "":
            
            # have no extension
            if filename[-4] not in [".mp3",".wav"]:
                outpath = filename + ".mp3"
            else:
                outpath = filename
        else:
            if filename[-4] not in [".mp3",".wav"]:
                outpath = str(output_folder) + "\\" + filename + ".mp3"
            else:
                outpath = str(output_folder) + "\\" + filename
        # engine.save_to_file(text, filename)
        engine.save_to_file(text, outpath)

    # when there's something to save
    


def detect_language(input_text: Union[str,list[str]], 
                    return_as: Literal["full_name","2_chr_code","3_chr_code","langcodes_obj"] = "full_name"):
    import pandas as pd
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
    elif isinstance(input_text, pd.Series):
        # not tested this part yet
        unique_text = pd.Series(input_text.unique()).dropna(how="all")
        unique_text = unique_text.loc[unique_text != False]
        unique_text = unique_text.astype(str)
        data_types_check = unique_text.apply(lambda x: type(x).__name__)
        full_text = unique_text.str.cat(sep=' ')
        detect_lang = detect_language(full_text,return_as)
        return detect_lang

def audio_from_df(df: pd.DataFrame,
                  audio_col:str,
                  output_folder:Union[str,Path] = "", 
                  filename_col:str = None,
                  language:str = "auto",
                  add_prefix_number: bool = True,
                  ) -> None:
    """
    This already include the prefix number
    
    # haven't test when add_prefix_number = False,
    
    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.
    audio_col : str
        DESCRIPTION.
    output_folder : Union[str,Path], optional
        DESCRIPTION. The default is "".
    filename_col : str, optional
        DESCRIPTION. The default is None.
    language : str, optional
        DESCRIPTION. The default is "auto".

    Returns
    -------
    None.

    """
    if filename_col is None:
        filename_col_in = audio_col
    else:
        filename_col_in = filename_col
    
    if language in ["auto"]:
        language_in = detect_language(df[audio_col])
    else:
        language_in = language
    # for testing avalibility of the tts(keyboard in your pc)
    n_rows = df.shape[0]
    
    digit_rows = len(str(n_rows))
    
    
    engine = engine_by_lang(language_in)
    if engine is False:
        raise Exception(f"Doesn't have '{language_in}' in the keyboard system. Please download this language in your keyboard. ")
    
    df_copy = df.copy()
    
    df_copy.reset_index(drop=True, inplace=True)
    df_copy.index += 1
    
    df_copy['chosen_filename'] = df_copy.index.map(lambda x: f"{str(x).zfill(digit_rows)}") + "_" + df_copy[filename_col_in]
    
    
    if add_prefix_number:
        chosen_col = 'chosen_filename'
    else:
        chosen_col = filename_col
    
    
    df_copy.apply(lambda row: create_audio(
        text = row[audio_col],
        filename = row[chosen_col],
        language = language_in,
        playsound = False,
        output_folder = output_folder
        ) ,
        axis = 1)


def create_audio_folder(excel_path,sheet_name):
    # NOT DONE
    import excel_tool as xt
    import excel_tool.worksheet as ws
    import dataframe_short as ds
    vocab_df = ds.pd_read_excel(excel_path,sheet_name = sheet_name)
    vocab_dict_df = ds.pd_split_into_dict_df(vocab_df,add_prefix_index = True)
    
    first_df = vocab_dict_df['01_Basics 1']