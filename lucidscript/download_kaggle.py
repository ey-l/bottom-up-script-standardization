from kaggle.api.kaggle_api_extended import KaggleApi
import os
import sys
import csv
import pandas as pd

if __name__ == "__main__":
    api = KaggleApi()
    api.authenticate()

    # Set args
    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = ROOT_PATH + '/kaggle_data/'
    competition = 'titanic'
    PATH = os.path.join(DATA_DIR, competition)
    language ='python'
    fields = ['ref', 'title', 'author', 'lastRunTime', 'totalVotes'] # from the API documentation
    sort_by = 'voteCount'
    page_size = 200 # num of scripts to download
    pageno_from = 1
    pageno_to = 500
    
    # Get list of kernels
    for p in range(pageno_from, pageno_to, 1):
        print("=== Page ===: ", p)
        ks = api.kernels_list(competition=competition,
                            language=language,
                            sort_by=sort_by,
                            page_size=100, # max
                            page=p)
        # Download notebooks
        for i in ks:
            ref = api.string(getattr(i, fields[0]))
            print(ref)
            try:
                api.kernels_pull(kernel=ref, path=os.path.join(PATH, 'notebooks')) # kernel must be {username}/{kernel_slug}
            except:
                print("Error downloading: ", ref)
                continue
    data = {
        'ref': [], 
        'title': [], 
        'author': [], 
        'lastRunTime': [], 
        'totalVotes': []
    }

    # Get list of kernels
    for p in range(pageno_from, pageno_to, 1):
        ks = api.kernels_list(competition=competition,
                            language=language,
                            sort_by=sort_by,
                            page_size=100, # max=100
                            page=p)
        for i in ks:
            for f in fields:
                data[f].append(api.string(getattr(i, f)))

    df = pd.DataFrame(data=data)
    df.to_csv(os.path.join(PATH, 'notebooks.csv'), index=False)