# Functions to help import the youtube dataset
# dataset link: https://www.kaggle.com/datasnaek/youtube-new/version/115

# all possible columns (you can retrieve only the columns you want by using relevant_cols):
# ['video_id' 'trending_date' 'title' 'channel_title' 'category_id'
#  'publish_time' 'tags' 'views' 'likes' 'dislikes' 'comment_count'
#  'thumbnail_link' 'comments_disabled' 'ratings_disabled'
#  'video_error_or_removed' 'description']

import pandas as pd

def load_dataset(path, relevant_cols=[]):
    data = pd.read_csv(path)
    data = data.drop_duplicates(subset='video_id', ignore_index=True) # drop duplicate video ids
    if relevant_cols != []:
        data = pd.DataFrame(data, columns=relevant_cols)

    return data

def main():
    path_to_dataset = '../data/CAvideos.csv'
    dataset = load_dataset(path_to_dataset, ['video_id', 'title'])

if __name__ == "__main__":
    main()
