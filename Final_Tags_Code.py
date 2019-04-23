import re
import pandas as pd
def read_file(file_name):
    data_frame = pd.read_csv(file_name)
    final_tag_list = []
    for key, row in data_frame.iterrows():
            tag = row["tags"]
            try:
                programming_language_list = ["php", "c#", "javascript", "java", "ruby-on-rails", "c++", "python", "c"]
                if tag in programming_language_list:
                    data_frame.loc[key, "tags"] = tag   
                else:
                    data_frame = data_frame.drop(key)    
            except:
                data_frame = data_frame.drop(key)
    return data_frame

if __name__ == "__main__":
    data_frame = read_file("Tags.csv")
    data_frame.to_csv("Final_Tags.csv", sep='\t')
