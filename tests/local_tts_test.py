from natural_language_processing.natural_language_processing.local_tts import *

def test_create_audio():
    # still doesn't work in VSCode(didn't create audio as files),
    
    # seems like it's because of the path format
    # if I only use the output file(.mp3 only) it would create the audio normally
    language = "french"

    text_list = ["les pâtes", "la sauce", "le bonbon", "l'oignon ", "la carotte"]
    output_folder = r"C:\Users\Heng2020\OneDrive\D_Code\Python\Python NLP\NLP 02\01 OutputData\test_create_audio"
    
    text_list02 = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    meaning_list02 = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # test for 00 format
    text_list03 = ["le haricot", "la viande", "la tomate", "le sandwich", "la baguette", "la soupe", "tu bois", "l'eau ", "l'alcool ", "l'oeuf ", "la salade", "ils boivent", "le riz"]


    
    for text in text_list:
        create_audio(text,language,filename = text + ".mp3",playsound=True,output_folder = output_folder)
    
    # test for list
    create_audio(
        text = text_list02,
        language = language,
        filename = meaning_list02,
        output_folder = output_folder,
        playsound = True,
        )
    
    # test for 00 format
    create_audio(
        text = text_list03,
        language = language,
        
        output_folder = output_folder,
        playsound = True,
        )
    
    
    

def test_detect_language():
    
    texts = [
        "Hello, how are you?",
        "Hallo, wie geht es dir?",
        "Hola, ¿cómo estás?",
        "今日はどうですか？",
        "Привет, как дела?"
    ]
    
    detect_language(texts)

def test_audio_from_df():
    import dataframe_short as ds
    excel_path = r"C:\Users\Heng2020\OneDrive\D_Documents\_Learn Languages\_LearnLanguages 02 Main\Duolingo\Duolingo French 02.xlsm"
    sheet_name = "python_test"
    out_folder = r"C:\Users\Heng2020\OneDrive\D_Code\Python\Python NLP\NLP 02\01 OutputData\test_audio_from_df"
    vocab_df = ds.pd_read_excel(excel_path,sheet_name=sheet_name)
    
    audio_from_df(vocab_df,'French',out_folder)