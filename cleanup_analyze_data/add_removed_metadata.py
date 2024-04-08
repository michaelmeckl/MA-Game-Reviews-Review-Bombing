"""
For uploading and annotating the reviews in Label Studio some columns were removed that were not needed for
the annotation but contained possibly helpful metadata. These are re-added to the now annotated reviews.
"""

import pathlib
import pandas as pd
from cleanup_analyze_data.review_bombing_incidents_info import review_bombing_incidents
from utils.utils import compare_pandas_dataframes


def add_removed_columns_to_df(df: pd.DataFrame):
    # load and combine the full_combined_review - dataframes for all incidents and then merge it with the annotated df
    combined_df_all_incidents = pd.DataFrame()
    combined_reviews_folder = pathlib.Path(__file__).parent.parent / "data_for_analysis_cleaned" / "reviews"
    for content_path in combined_reviews_folder.iterdir():
        if content_path.is_file():
            incident_dataframe = pd.read_csv(content_path)
            combined_df_all_incidents = pd.concat([combined_df_all_incidents, incident_dataframe], ignore_index=True)

    # check unicode to see if there were any special characters added to the review text that prevent merging them
    # combined_df_all_incidents.review = combined_df_all_incidents.review.str.encode('utf-8')
    # print(combined_df_all_incidents.iloc[0]["review"])

    # replace all newline and carriage return characters with a single newline because this caused a lot of problems
    # while trying to merge the "review" columns
    df.review = df.review.str.replace(r'[\r\n]+', '\n', regex=True)
    combined_df_all_incidents.review = combined_df_all_incidents.review.str.replace(r'[\r\n]+', '\n', regex=True)
    # combined_df_all_incidents.review = combined_df_all_incidents.review.str.replace(r'\r+|\n+|\t+', r'\n', regex=True)

    # merge the columns from both dataframes for the annotated reviews
    merged_df = df.merge(combined_df_all_incidents, how="inner")

    # sanity check: (should be the same!)
    print(f"Annotated dataframe before merge: {len(df)}")
    df.groupby("project").apply(lambda x: print(len(x)))
    print(f"\nMerged dataframe: {len(merged_df)}")
    merged_df.groupby("project").apply(lambda x: print(len(x)))
    merged_df = merged_df.drop_duplicates(subset=["id"])
    print(f"\nMerged dataframe after drop duplicates: {len(merged_df)}")
    merged_df.groupby("project").apply(lambda x: print(len(x)))

    compare_pandas_dataframes(df, merged_df, merge_column="review", df_1_name="df_original", df_2_name="df_merged",
                              write_to_csv=False)

    # cols_to_use = combined_df_all_incidents.columns.difference(df.columns)
    # dfNew = df.merge(combined_df_all_incidents[cols_to_use], left_index=True, right_index=True, how='outer')

    columns_to_drop = ["game", "affected_games", "review_bomb_time", "combined_rating_display", "review_id",
                       "steam_rating_positive", "created_at", "last_updated", "author_country_code",
                       "author_last_online", "author_real_name", "game_id", "title", "release_date", "price_euro",
                       "detailed_description", "platform", "game_title"]
    merged_df = merged_df.drop(columns=columns_to_drop, axis=1)

    # move column position
    merged_df.insert(0, "review_bombing_incident", merged_df.pop("review_bombing_incident"))

    # merged_df = merged_df.rename(columns={"id": "label_studio_id"})
    # merged_df.sort_values(by=["project", "review_date"], ascending=False, inplace=True)
    return merged_df


def add_additional_metadata(df: pd.DataFrame):
    # add columns and fill with default values before filling in the correct values
    df.insert(17, "marked_off_topic", [False] * len(df))
    df.insert(6, "rb_end_date", ["01.01.1970"] * len(df))
    df.insert(6, "rb_start_date", ["01.01.1970"] * len(df))

    for incident in df["review_bombing_incident"].unique():
        rb_start_date = review_bombing_incidents[incident]["rb_start_date"]
        rb_end_date = review_bombing_incidents[incident]["rb_end_date"]
        marked_off_topic = review_bombing_incidents[incident]["marked_steam_off_topic"]

        df.loc[df["review_bombing_incident"] == incident, ["marked_off_topic", "rb_start_date",
                                                           "rb_end_date"]] = marked_off_topic, rb_start_date, rb_end_date


if __name__ == "__main__":
    pd.options.display.width = 0
    input_data_path = pathlib.Path(__file__).parent.parent / "label_studio_study" / "parsed_data" / "combined_final_annotation_all_projects.csv"
    output_data_path = pathlib.Path(__file__).parent.parent / "combined_final_annotation_all_projects_updated.csv"

    should_add_removed_columns = False
    if should_add_removed_columns:
        input_data = pd.read_csv(input_data_path)
        updated_input_data = add_removed_columns_to_df(input_data)
        updated_input_data.to_csv(output_data_path, index=False)

    # add some additional metadata columns to the dataframe
    updated_input_data = pd.read_csv(output_data_path)
    add_additional_metadata(updated_input_data)
    updated_input_data.to_csv(output_data_path, index=False)
