import json
import pathlib
import pprint
import pandas as pd
import re
from datetime import datetime
from cleanup_analyze_data.review_bombing_incidents_info import review_bombing_incidents
from utils import utils


def analyze_steam_review_graph(graph_df: pd.DataFrame):
    """
    Search for temporal bursts and other suspicious changes in the number of reviews per month (except for the first
    two month as most reviews are usually there; besides there are obviously 0 reviews before the first one,
    so this is most likely the one with the highest entries)
    """
    def analyze_game_graph(game_df):
        suspicious_dates_set = set()
        game_name = game_df.name
        print(f"\n################## Analyzing game: {game_name} ##################\n")
        game_df = game_df.reset_index(drop=True)  # reset index so iloc[...] works on all groups

        # calculate mean and median for columns
        mean_votes_up = game_df["recommendations_up"].mean()
        mean_votes_down = game_df["recommendations_down"].mean()
        mean_votes_overall = (mean_votes_up + mean_votes_down) / 2
        print(f"Mean for votes up is {mean_votes_up:.2f} and for votes down {mean_votes_down:.2f}; mean overall: "
              f"{mean_votes_overall:.2f}")
        median_votes_up = game_df["recommendations_up"].median()
        median_votes_down = game_df["recommendations_down"].median()
        print(f"Median for votes up is {median_votes_up} and for votes down {median_votes_down}")

        # define thresholds, i.e. the amount of new reviews from one month to the next seen as suspicious
        threshold_votes_up = median_votes_up * 2
        threshold_votes_down = median_votes_down * 2

        # also find max value in column and check if it is outside the first two months after release!
        max_value_up = game_df["recommendations_up"].max()
        max_value_down = game_df["recommendations_down"].max()
        idx_of_max_value_up = game_df["recommendations_up"].idxmax()
        idx_of_max_value_down = game_df["recommendations_down"].idxmax()
        month_of_max_value_up = game_df.iloc[idx_of_max_value_up]["month"]
        month_of_max_value_down = game_df.iloc[idx_of_max_value_down]["month"]
        max_value_up_suspicious = True if idx_of_max_value_up > 1 else False
        max_value_down_suspicious = True if idx_of_max_value_down > 1 else False
        print(f"Timeseries max value up {max_value_up} at month {month_of_max_value_up} (suspicious:"
              f" {max_value_up_suspicious})\n max value down {max_value_down} at month {month_of_max_value_down} ("
              f"suspicious: {max_value_down_suspicious})\n")

        sub_df = game_df.set_index("month")[["recommendations_up", "recommendations_down", "recommendations_diff"]]
        votes_diff = sub_df.diff()
        votes_diff.iloc[0] = sub_df.iloc[0]  # replace the first row with the old value
        # df_pct_change = sub_df.pct_change()
        # df_pct_change *= 100

        # check for which month / row the calculated difference values exceed the thresholds defined above
        votes_diff_new = votes_diff.iloc[2:]  # remove the first two rows / month
        votes_diff_suspicious = votes_diff_new[(votes_diff_new["recommendations_up"] >= threshold_votes_up) | (
                    votes_diff_new["recommendations_down"] >= threshold_votes_down)]
        # move index to a column
        votes_diff_suspicious["date"] = votes_diff_suspicious.index

        print(f"Suspicious entries for defined thresholds up ({threshold_votes_up}) and down ({threshold_votes_down}):")
        pprint.pprint(votes_diff_suspicious)

        # check overlap month with max value for suspicious month:
        for month in votes_diff_suspicious["date"]:
            if month == month_of_max_value_up:
                print(f"Overlap for suspicious date: {month} (suspicious amount of new reviews as well as highest "
                      f"amount of positive reviews overall at this point)")
                suspicious_dates_set.add(month_of_max_value_up)
            elif month == month_of_max_value_down:
                print(f"Overlap for suspicious date: {month} (suspicious amount of new reviews as well as highest "
                      f"amount of negative reviews overall at this point)")
                suspicious_dates_set.add(month_of_max_value_down)
        """
        # alternative:
        if max_value_up_suspicious:
            suspicious_dates_set.add(month_of_max_value_up)
        if max_value_down_suspicious:
            suspicious_dates_set.add(month_of_max_value_down)
        """

        # also compare recommendations up and down column per month (e.g. are there different trends at the same
        # time, i.e. high diff between recommendations up and down?)
        df_suspicious_difference = votes_diff[abs(votes_diff["recommendations_diff"] >= mean_votes_overall)]

        # merge both dataframes to narrow down the suspicious dates
        merged_suspicious_dates = votes_diff_suspicious.merge(df_suspicious_difference, how="inner")
        merged_suspicious_date_list = merged_suspicious_dates["date"].tolist()

        # suspicious_dates_set.update(merged_suspicious_date_list)

        # filter these, e.g. check also one month before and after if the number of reviews differs greatly from them
        shift_vals = [-1, 1]  # -2, 2
        for date in merged_suspicious_date_list:
            consecutive_diff_vals = []
            for shift_val in shift_vals:
                diff_results = sub_df['recommendations_diff'] - sub_df['recommendations_diff'].shift(shift_val)
                consecutive_diff_vals.append(diff_results.loc[date])

            # TODO mean_votes_overall is a wild guess, test others
            if any(consecutive_diff_vals >= mean_votes_overall):
                # is a valid suspicious date if either the difference with the value before or after is great enough
                # (i.e. if the number of reviews at this date is significantly greater than before or after)
                suspicious_dates_set.add(date)

        # for all final suspicious dates, try to find the end of the possible review bombing (e.g. find the next
        # month where the diff is <= (around) mean_votes_overall (or mean_up/down depending on the suspicious part?))
        suspicious_time_periods = {}
        for suspicious_date in suspicious_dates_set:
            test = sub_df.loc[suspicious_date:, :]
            for row in test.iterrows():
                recommendations_diff = row[1]["recommendations_diff"]
                if recommendations_diff <= mean_votes_overall:   # TODO mean_votes_overall is a wild guess, test others
                    row_date = row[0]
                    suspicious_time_periods[suspicious_date] = row_date
                    break
        print(f"\nFound the following suspicious time periods: {suspicious_time_periods}")

        print(f"=> Found {len(suspicious_dates_set)} ({suspicious_dates_set}) suspicious dates for game {game_name}")
        suspicious_dates.update({game_name: suspicious_dates_set})

    # keep track of all suspicious months found during the analysis of the games' review graphs
    suspicious_dates = {
        # "game_name": [list_of_suspicious_dates]
    }
    graph_df.groupby(by=["game"]).apply(lambda game_group: analyze_game_graph(game_group))


def parse_review_graph(review_graph_data: dict, game_name: str, rb_incident: str):
    # convert unix timestamps to readable dates
    start_date_unix = review_graph_data["start_date"]
    end_date_unix = review_graph_data["end_date"]
    start_date = datetime.fromtimestamp(start_date_unix).strftime('%d.%m.%Y')
    end_date = datetime.fromtimestamp(end_date_unix).strftime('%d.%m.%Y')
    print(f"\nParsing steam review graph for game {game_name} from {start_date = } to {end_date = }")

    marked_as_off_topic = review_bombing_incidents[rb_incident]["marked_steam_off_topic"]
    steam_review_graph_dict = {"review_bombing_incident": rb_incident, "game": game_name,
                               "marked_off_topic": marked_as_off_topic, "month": [], "recommendations_up": [],
                               "recommendations_down": []}

    months_data = review_graph_data["review_data"]
    for month in months_data:
        date_utc = month["date"]
        date = datetime.fromtimestamp(date_utc).strftime('%m.%Y')
        upvotes = month["recommendations_up"]
        downvotes = month["recommendations_down"]
        # print(f"{date}: {upvotes} Upvotes / {downvotes} Downvotes")
        steam_review_graph_dict["month"].append(date)
        steam_review_graph_dict["recommendations_up"].append(upvotes)
        steam_review_graph_dict["recommendations_down"].append(downvotes)

    # convert data to csv
    review_graph_df = pd.DataFrame(steam_review_graph_dict)
    # review_graph_df['month'] = pd.to_datetime(review_graph_df['month'], format='%m.%Y')
    # review_graph_df['month'] = review_graph_df['month'].dt.strftime('%m.%Y')
    return review_graph_df


def cleanup_steam_review_graphs():
    if not output_folder_path.is_dir():
        output_folder_path.mkdir()

    rb_names = ["Skyrim-Paid-Mods", "Assassins-Creed-Unity", "Firewatch", "Mortal-Kombat-11",
                "Borderlands-Epic-Exclusivity", "Ukraine-Russia-Conflict"]
    for rb_name in rb_names:
        Sub_Folder = output_folder_path / rb_name
        if not Sub_Folder.is_dir():
            Sub_Folder.mkdir()
        else:
            print("WARNING: Subfolder already exists!")
            continue

        games_title_terms = review_bombing_incidents[rb_name]["games_title_terms"]
        relevant_files = set()
        for term in games_title_terms:
            pattern = f"{term}.json"
            relevant_files.update([f for f in input_folder_path.glob(pattern)])

        print(f"{len(relevant_files)} files found for incident {rb_name}:")
        print(relevant_files)

        combined_review_graphs = []
        for file_path in relevant_files:
            file_path_name_parts = re.findall(r'([A-Z].+)', file_path.stem)
            game = file_path_name_parts[0]
            with open(file_path, "r") as f:
                steam_graph_data = json.load(f)
            graph_data_df = parse_review_graph(steam_graph_data, game, rb_name)
            graph_data_df.to_csv(Sub_Folder / f"review_graph_{game}.csv", index=False)
            combined_review_graphs.append(graph_data_df)

        # if there is more than one game / review graph for this rb incident, save a combined csv file too
        if len(relevant_files) > 1:
            combined_review_graph_df = pd.concat(combined_review_graphs, ignore_index=True)
            # combined_review_graph_df.sort_values(by=["game"], inplace=True)
            combined_review_graph_df.to_csv(output_folder_path / f"combined_review_graph_{rb_name}.csv", index=False)


def combine_all_review_graphs():
    dataframe_list = []
    for folder_content in output_folder_path.iterdir():
        if folder_content.is_dir():
            for file in folder_content.iterdir():
                # converter important to keep trailing zero for year
                graph_dataframe = pd.read_csv(file, converters={'month': str})
                dataframe_list.append(graph_dataframe)

    all_review_graphs_df = pd.concat(dataframe_list, ignore_index=True)
    # add a new column with the difference between votes up and down for each month
    all_review_graphs_df["recommendations_diff"] = all_review_graphs_df["recommendations_up"] - all_review_graphs_df["recommendations_down"]
    all_review_graphs_df.to_csv(output_folder_path / f"all_review_graphs_combined.csv", index=False)


if __name__ == "__main__":
    utils.enable_max_pandas_display_size()
    input_folder_path = pathlib.Path(__file__).parent.parent / "data_for_analysis" / "steam"
    output_folder_path = pathlib.Path(__file__).parent.parent / "data_for_analysis_cleaned" / "review_graphs"

    cleanup_combine_graph_data = False
    if cleanup_combine_graph_data:
        cleanup_steam_review_graphs()
        combine_all_review_graphs()

    analyze_graph_data = True
    if analyze_graph_data:
        all_review_graphs = pd.read_csv(output_folder_path / f"all_review_graphs_combined.csv", converters={'month': str})
        analyze_steam_review_graph(all_review_graphs)
