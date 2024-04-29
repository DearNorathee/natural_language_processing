
import pysrt
import pandas as pd
import os
from pydub import AudioSegment
from playsound import playsound
# Parse the SRT subtitle file

srt_folder_path = r"H:\D_Video\TheBigBangTheory\TheBigBangTheory6\EN Subtitle"
output_folder = r"C:\Users\Heng2020\OneDrive\Python NLP\NLP 01\out"

# for a single file
srt_path = r"H:\D_Video\Westworld Portugues 04\Eng Sub\Westworld.S04E01 EngSub 02.srt"
sub_output_name = 'Westworld_S04E01_EN02.xlsx'


alarm_path = r"H:\D_Music\Sound Effect positive-logo-opener.wav"


sub_output = os.path.join(srt_folder_path,sub_output_name)
# what if I extract str directly from video .mvk?
def srt_to_df(srt_path,
              remove_stopwords=True,
              stopwords = ["♪","\n","<i>","</i>","<b>","</b>"]):
# remove_newline will remove '\n' from the extracted text
    
    if ".srt" in srt_path:
        # 1 file case
        subs = pysrt.open(srt_path)
        # Initialize empty lists for storing data
        sentences = []
        start_times = []
        end_times = []
    
        # Extract data from each subtitle sentence
        for sub in subs:
            sentences.append(sub.text)
            start_times.append(sub.start.to_time())
            end_times.append(sub.end.to_time())
    
        # Create a DataFrame
        if remove_stopwords:
            #FIX it's still can't replace properly 
            sentences = [St_replace(sentence,stopwords,"") for sentence in sentences]
        df = pd.DataFrame({
            'sentence': sentences,
            'start': start_times,
            'end': end_times
        })
        return df
    else:
        # many srt's file using folder
        str_file_names = get_full_filename(srt_path,".srt")
        df_list = []
        for str_file_name in str_file_names:
            each_df = srt_to_df(str_file_name)
            df_list.append(each_df)
        return df_list


def srt_to_csv(srt_path,output_path,encoding='utf-8-sig',index=False):
    # output should be total_path
    df_sub = srt_to_df(srt_path)
    # encoding='utf-8-sig' for Portuguese
    df_sub.to_csv(output_path, encoding=encoding,index=index)

def srt_to_Excel(srt_path,output_path,encoding='utf-8-sig',index=True):
    """ 
    Wrote on Aug 27, 2023
    I already wrote it for 1 file but it took me about 3 additional hrs to 
    make it work with multiple files in folder
    """
    # output should be total_path
    df_sub = srt_to_df(srt_path)
    
    if isinstance(df_sub,pd.DataFrame):
    # encoding='utf-8-sig' for Portuguese
        df_sub.to_excel(output_path, encoding=encoding,index=index)
    elif isinstance(df_sub,list):
        short_names = get_filename(srt_path,".srt")
        out_full_name = [os.path.join(output_path,short_name).replace(".srt",".xlsx") for short_name in short_names]

        for i,df in enumerate(df_sub):
            df.to_excel(out_full_name[i], encoding=encoding,index=index)

def to_ms(time_obj):
    time_obj_ms = (time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second) * 1000 + time_obj.microsecond // 1000
    return time_obj_ms

def get_filename(folder_path,extension = "all"):
    # also include "folder"  case
# tested small
# new feature1: include subfolders
    if extension == "all":
        out_list = [ file for file in os.listdir(folder_path) ]

    elif isinstance(extension,str):
        extension_temp = [extension]

        out_list = []

        for file in os.listdir(folder_path):
            if "." in file:
                file_extension = file.split('.')[-1]
                for each_extention in extension_temp:
                    # support when it's ".csv" or only "csv"
                    if file_extension in each_extention:
                        out_list.append(file)
            elif extension == "folder":
                out_list.append(file)


    elif isinstance(extension,list):
        out_list = []
        for file in os.listdir(folder_path):

            if "." in file:
                file_extension = file.split('.')[-1]
                for each_extention in extension:
                    # support when it's ".csv" or only "csv"
                    if file_extension in each_extention:
                        out_list.append(file)

            elif "folder" in extension:
                out_list.append(file)

        return out_list

    else:
        print("Don't support this dataype for extension: please input only string or list")
        return False

    return out_list

def get_full_filename(folder_path,extension = "all"):
    # tested small
    short_names = get_filename(folder_path,extension)
    out_list = []
    for short_name in short_names:
        full_name = os.path.join(folder_path,short_name)
        out_list.append(full_name)
    return out_list

def St_replace(text,to_replace,replace_by):
    # unit_tested
    for word in to_replace:
        new_text = text.replace(word, replace_by)
        
    return new_text

# TODO: srt_to_Excel => similar to srt_to_csv but output as excel
# srt_to_Excel(srt_path,sub_output)

# n_file = len(srt_to_df(srt_folder_path))
# srt_to_Excel(srt_folder_path,output_folder)

# print(f"Done converting srt to Excel in Total {n_file} files ")
# playsound(alarm_path)



