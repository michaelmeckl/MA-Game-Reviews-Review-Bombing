#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The code to scrape the steam game data is based on this Github repository:
https://github.com/FronkonGames/Steam-Games-Scraper (Original Author: Martin Bustos <fronkongames@gmail.com>")
"""
import os
import pathlib
import pandas as pd


OUTPUT_FILE = "games.json"
PROGRAM_PATH = pathlib.Path(__file__).parent / "Steam-Games-Scraper-main" / "SteamGamesScraper.py"


def extract_relevant_fields(json_file):
    if os.path.exists(json_file):
        game_df = pd.read_json(json_file, orient="index")
        print(game_df.head(5))
        # column_names = list(game_df.columns)

        # specify only the fields we need
        filtered_game_df = game_df[["name", "release_date", "detailed_description", "price", "estimated_owners",
                                    "reviews", "metacritic_score", "metacritic_url", "user_score", "positive",
                                    "negative", "score_rank", "recommendations", "average_playtime_forever",
                                    "developers", "publishers", "genres"]]
        filtered_game_df.index.name = 'app_id'
        filtered_game_df.to_csv("steam_games.csv")
    else:
        print(f"ERROR: {json_file} does not exist!")


def scrape_steam_games():
    exit_code = os.system(f"python {PROGRAM_PATH} -o {OUTPUT_FILE} -c \"EUR\"")

    if exit_code != 0:
        print(f"Executing SteamGamesScraper.py did not work! Exit code: {exit_code}")


if __name__ == "__main__":
    scrape_steam_games()
    # extract_relevant_fields(json_file="games.json")
