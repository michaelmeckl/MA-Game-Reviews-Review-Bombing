"""
Util to add the user info for each review in a csv file retrospectively. Necessary because a lot of reviews had already
been scraped before the code for scraping the user data was added (otherwise all of these would need to be scraped
again).
"""
from metacritic_scraper_new import scrape_user_profile
import pathlib
import pandas as pd


def cleanup_helpful_score(review_df: pd.DataFrame):
    # split the "helpful_score" column with the format "5 / 6" into two new columns "helpful_votes" & "unhelpful_votes"
    review_df[['helpful_votes', 'unhelpful_votes']] = review_df['helpful_score'].str.split(' / ', expand=True)
    # calculate the correct new values
    review_df['unhelpful_votes'] = review_df['unhelpful_votes'].astype('int')
    review_df['helpful_votes'] = review_df['helpful_votes'].astype('int')
    review_df['unhelpful_votes'] = review_df['unhelpful_votes'] - review_df['helpful_votes']
    # remove the now redundant old column
    review_df = review_df.drop(columns=['helpful_score'], axis=1)
    # reorder columns
    review_df = review_df.reindex(
        columns=["username", "review_date", "helpful_votes", "unhelpful_votes", "rating", "review"])
    return review_df


def get_user_information(username):
    user_dict = {'author_ratings_overall': [], 'author_reviews_overall': [],
                 'author_num_game_reviews': [], 'author_average_score': [],
                 'author_review_distribution': []}
    try:
        scrape_user_profile(username, user_dict)
    except Exception as e:
        print(f"ERROR: {e}")

    # convert to dataframe first to get rid of the lists in the dictionary, then squeeze df into a pd.Series
    return pd.DataFrame(user_dict).squeeze()


def add_user_data_retrospectively(review_df: pd.DataFrame):
    # fetch and add the new user information based on the existing "username" column
    user_data = review_df["username"].apply(lambda name: get_user_information(name))
    new_review_df = pd.concat([review_df, user_data], axis=1)
    return new_review_df


if __name__ == "__main__":
    pd.options.display.width = 0
    # load the csv file with the metacritic reviews that still lacks the user information
    Metacritic_Data_Folder = pathlib.Path(__file__).parent / "metacritic_data"
    filepath = Metacritic_Data_Folder / "user_reviews_pc_cyberpunk-2077_without_user_info.csv"
    existing_data_df = pd.read_csv(filepath)

    new_dataframe = cleanup_helpful_score(existing_data_df)
    # print(new_dataframe.head())

    print("\nAdding user information to review dataframe retrospectively ...")
    df_with_user_infos = add_user_data_retrospectively(new_dataframe)
    print(df_with_user_infos.head())
    df_with_user_infos.to_csv(Metacritic_Data_Folder / "user_reviews_pc_cyberpunk-2077.csv", index=False)
