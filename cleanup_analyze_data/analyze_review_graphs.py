import json
import pathlib
import pandas as pd
import re
from datetime import datetime
from cleanup_analyze_data.review_bombing_incidents_info import review_bombing_incidents


def analyze_steam_review_graph(graph_df: pd.DataFrame):
    pass
    # TODO search for temporal bursts and other suspicious changes in the number of reviews per month
    #  define threshold (i.e. how many more new reviews) and check from month to month
    # don't search in the first (two?) month as most reviews are usually there (besides there are obviously 0 reviews
    # before that)


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
                graph_dataframe = pd.read_csv(file)
                dataframe_list.append(graph_dataframe)

    all_review_graphs_df = pd.concat(dataframe_list, ignore_index=True)
    all_review_graphs_df.to_csv(output_folder_path / f"all_review_graphs_combined.csv", index=False)


if __name__ == "__main__":
    input_folder_path = pathlib.Path(__file__).parent.parent / "data_for_analysis" / "steam"
    output_folder_path = pathlib.Path(__file__).parent.parent / "data_for_analysis_cleaned" / "review_graphs"

    # cleanup_steam_review_graphs()
    # combine_all_review_graphs()
