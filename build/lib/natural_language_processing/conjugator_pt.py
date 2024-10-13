
class Conjugator_PT():
    
    # used mlconjug3

    # verbecc will have other structure
    
    def __init__(self,conjugator):
        self.conjugator = conjugator
        
        
    def present(self,verb_inf):
        import python_wizard as pw
        # TODO add subject as parameter
        verb = self.conjugator.conjugate(verb_inf)
        subject_order = ['eu','ele','nós','eles']
        conju_info = verb.conjug_info['Indicativo']['Indicativo presente']
        conju_info_ordered = pw.reorder_dict(conju_info,subject_order)
        
        return conju_info_ordered
    
    def conju_table_1verb(self,verb_inf):
        conju_dict = self.conjug_info(verb_inf)
        out_df = self._conju_table(conju_dict)
        return out_df
        
                
    def _conju_table(self,nested_dict):
        # only works for mlconjug3
        import pandas as pd
        import python_wizard as pw
        df = pd.DataFrame()

        exclude_mood = ["Gerúndio","Imperativo","Infinitivo","Particípio"]
        count = 0
        main_subject = ["eu","ele","nós","eles"]
        rename_map = {
            "Indicativo presente"                           :	"Present Simple",
            "Indicativo pretérito perfeito simples"         :	"Past Simple",  
            "Indicativo pretérito imperfeito"               :	"(1) Pretérito Imperfeito do Indicativo",
            "Indicativo Futuro do Presente Simples"         :	"(2) Future Simple",
            "Condicional Futuro do Pretérito Simples"       :	"(3) Futuro do Pretérito do Indicativo",
            "Conjuntivo  Subjuntivo Presente"               :	"(4) Subjuntivo",
            "Conjuntivo  Subjuntivo Pretérito Imperfeito"   :	"(5) Past Subjuntivo"
            }
        
        for mood, tenses in nested_dict.items():
            
            for tense, subjects in tenses.items():
                if tense in rename_map.keys():

                    s = pw.filter_dict(subjects, main_subject)
                    s = pd.DataFrame([s], columns=s.keys())
                    
                    new_name = rename_map[tense]
         
                    s['tense'] = new_name

                    col_order = ['tense'] + main_subject
                    s_reorder = s[col_order]
                    
                    count += 1
        
                    # Append the series to the dataframe
                    df = pd.concat([df,s_reorder])
                
        df = df.reset_index(drop=True)
        # swap row index 1&2
        df.iloc[[1, 2]] = df.iloc[[2, 1]].to_numpy()
        return df
                    
                
    
    def conjug_info(self,verb_inf):
        return self.conjugator.conjugate(verb_inf).conjug_info
    
    
    def conju_table(self,verb_list):
        import pandas as pd
        import python_wizard as pw
        # levar past simple is wrong: the result is: lev)   !!!!!
        # same problem with pular, odiar, atuar
        
        out_df = pd.DataFrame()
        
        verb_list_in = pw.to_list(verb_list)

        NEW_TENSE_NAMES = ["Present Simple",
                  "Past Simple",
                  "(1) Pretérito Imperfeito do Indicativo",
                  "(2) Future Simple",
                  "(3) Futuro do Pretérito do Indicativo",
                  "(4) Subjuntivo",
                  "(5) Past Subjuntivo"
                  ]

        MAIN_SUBJECTS = ["eu","ele","nós","eles"]

        for curr_verb in verb_list_in:
            conju_df = self.conju_table_1verb(curr_verb)
            conju_df = conju_df.set_index('tense')
            verb_conju_list = [conju_df.values.flatten().tolist()]
            
            index = pd.MultiIndex.from_arrays([[curr_verb]], names = ['verb'])
            # use .from_product
            columns = pd.MultiIndex.from_product([NEW_TENSE_NAMES,MAIN_SUBJECTS], names=['tense','subject'] )
            df = pd.DataFrame(verb_conju_list,index=index,columns=columns)
            out_df = pd.concat([out_df,df])
        return out_df