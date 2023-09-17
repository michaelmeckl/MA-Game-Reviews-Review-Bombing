#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pathlib
import pprint
import shutil
import re
import time
import pandas as pd
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from enum import Enum


##################### Global Variables #####################
class PLATFORM(Enum):
    PC = "pc"
    SWITCH = "switch"
    PS5 = "playstation-5"
    PS4 = "playstation-4"
    PS3 = "playstation-3"
    XBOX_ONE = "xbox-one"
    XBOX_SERIES_X = "xbox-series-x"
    XBOX_360 = "xbox-360"


# list of games where the user reviews should be scraped and corresponding platforms
# the names are directly used for the url, so they should be written exactly as expected by Metacritic
games_to_scrape = {
    # "hogwarts-legacy": [PLATFORM.PC.value],
    # "cyberpunk-2077": [PLATFORM.PC.value],  # PLATFORM.PS5, PLATFORM.PS4, PLATFORM.XBOX_SERIES_X, PLATFORM.XBOX_ONE],
    # "the-last-of-us-part-ii": [PLATFORM.PS4.value],
    # "firewatch": [PLATFORM.PC.value],
    # "borderlands": [PLATFORM.PC.value],
    # "borderlands-the-pre-sequel": [PLATFORM.PC.value],
    # "borderlands-2": [PLATFORM.PC.value],
    # "borderlands-3": [PLATFORM.PC.value],
    # "metro-exodus": [PLATFORM.PC.value],
    # "metro-last-light-redux": [PLATFORM.PC.value],
    # "metro-last-light": [PLATFORM.PC.value],
    # "metro-2033-redux": [PLATFORM.PC.value],
    # "metro-2033": [PLATFORM.PC.value],
    # "overwatch-2": [PLATFORM.PC.value],
    "gwent-the-witcher-card-game": [PLATFORM.PC.value],
    "thronebreaker-the-witcher-tales": [PLATFORM.PC.value],
    "the-witcher": [PLATFORM.PC.value],
    "the-witcher-enhanced-edition": [PLATFORM.PC.value],
    "the-witcher-2-assassins-of-kings": [PLATFORM.PC.value],
    "the-witcher-3-wild-hunt": [PLATFORM.PC.value],
    "s-t-a-l-k-e-r-shadow-of-chernobyl": [PLATFORM.PC.value],
    "s-t-a-l-k-e-r-call-of-pripyat": [PLATFORM.PC.value],
    "s-t-a-l-k-e-r-clear-sky": [PLATFORM.PC.value],
    "frostpunk": [PLATFORM.PC.value],
}

since = "24-02-2022"  # "start date", i.e. the oldest date
until = "31-03-2022"  # "end date", i.e. the most recent date (inclusive!)

headers = {'User-agent': 'Mozilla/5.0'}

############################################################


def scrape_general_game_information(requests_session, game: str, out_data: dict):
    """
    Scrape general game information
    """
    metacritic_url = f"https://www.metacritic.com/game/{game}/"
    response = requests_session.get(metacritic_url, headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')

    score_info = soup.find("div", class_="c-productHero_scoreInfo")
    try:
        user_score_regex = re.compile("User score")
        critic_score_regex = re.compile("Metascore")
        critic_score = score_info.find("div", title=critic_score_regex).find("span").text
        user_score = score_info.find("div", title=user_score_regex).find("span").text
    except Exception as e:
        print(f"ERROR: {e}")
        critic_score = ""
        user_score = ""

    print(f"Critic Score: {critic_score}")
    print(f"User Score: {user_score}")
    out_data["critic_score"].append(critic_score)
    out_data["user_score"].append(user_score)

    num_ratings = ""
    try:
        reviews_total_elements = soup.findAll("span", class_="c-ScoreCard_reviewsTotal")
        for el in reviews_total_elements:
            user_reviews_el = el.find("a", href=re.compile(r"user-reviews"))
            if user_reviews_el is not None:
                num_user_reviews_text = user_reviews_el.find("span").text
                num_user_ratings_regex = re.compile(r"([0-9]+[.,]*[0-9]+)")
                # extract the number from the string
                num_ratings = re.search(num_user_ratings_regex, num_user_reviews_text).group(1)
                num_ratings = num_ratings.replace(",", "").replace(".", "")   # remove the delimiter
                break
    except Exception as e:
        print(f"ERROR: {e}")

    print(f"Number of User Ratings: {num_ratings}")
    out_data["num_user_ratings"].append(num_ratings)

    try:
        game_release_date_container = soup.find("div", class_="c-gameDetails_ReleaseDate")
        game_release_date = game_release_date_container.findAll("span")[1].text
        formatted_release_date = datetime.strptime(game_release_date, '%b %d, %Y').strftime('%d.%m.%Y')
    except Exception as e:
        print(f"ERROR: {e}")
        formatted_release_date = ""

    print(f"Release Date: {formatted_release_date}")
    out_data["release_date"].append(formatted_release_date)

    try:
        user_rating_distribution = {}

        user_ratings_container_regex = re.compile("c-reviewsSection_header-user")
        user_ratings_container = soup.find("div", class_=user_ratings_container_regex)
        user_rating_distribution_container = user_ratings_container.find("div", class_="c-reviewsStats")
        user_rating_distribution_entries = user_rating_distribution_container.findAll("div")

        # the first one is always "Positive", then "Mixed" and "Negative"
        score_distribution_count_pos = user_rating_distribution_entries[0].findChildren()[2]  # take the 3. child
        score_distribution_count_mix = user_rating_distribution_entries[1].findChildren()[2]
        score_distribution_count_neg = user_rating_distribution_entries[2].findChildren()[2]
        # extract the digits from the strings
        user_rating_distribution["Positive"] = re.findall(r'\d+', score_distribution_count_pos.text)[0]
        user_rating_distribution["Mixed"] = re.findall(r'\d+', score_distribution_count_mix.text)[0]
        user_rating_distribution["Negative"] = re.findall(r'\d+', score_distribution_count_neg.text)[0]
    except Exception as e:
        print(f"ERROR: {e}")
        user_rating_distribution = {}

    out_data["user_score_distribution"].append(user_rating_distribution)


def scrape_game_description(requests_session, game: str):
    game_details_url = f"https://www.metacritic.com/game/{game}/details/"
    response = requests_session.get(game_details_url, headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')
    try:
        game_description_el = soup.find("div", class_="c-pageProductDetails_description")
        game_description = game_description_el.contents[1].text
        game_description = game_description.replace('\n', '').strip()
    except Exception as e:
        print(f"ERROR: {e}")
        game_description = ""

    return game_description


def scrape_user_profile(username: str, user_dict: dict):
    user_profile_url = f"https://www.metacritic.com/user/{username}"
    response = requests.get(user_profile_url, params={"filter": "games"}, headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')

    main_section = soup.find("section", class_="u-grid-column-span2")
    try:
        average_review_score = main_section.find("span", class_="c-scoreOverview_avgScoreText").text
        user_dict["author_average_score"].append(average_review_score)
    except Exception as e:
        print(f"ERROR: {e}")
        user_dict["author_average_score"].append("")

    # apparently dynamically generated, so this doesn't work
    """
    try:
        num_game_reviews_container = soup.find("ul", class_="c-globalHeader_container")
        num_game_reviews = num_game_reviews_container.find("span", class_="c-globalHeader_menu_subText").text
        user_dict["author_num_game_reviews"].append(num_game_reviews)
    except Exception as e:
        print(f"ERROR: {e}")
        user_dict["author_num_game_reviews"].append("")
    """

    try:
        review_score_distribution = {}
        score_distribution_el = main_section.find("div", class_="c-scoreCount_container u-grid")
        distribution_entries = score_distribution_el.findAll("div", class_="c-scoreCount_count")
        # the first one is "Positive", then "Mixed" and "Negative"
        score_distribution_count_pos = distribution_entries[0].findChild()
        score_distribution_count_mix = distribution_entries[1].findChild()
        score_distribution_count_neg = distribution_entries[2].findChild()
        review_score_distribution["Positive"] = score_distribution_count_pos.text
        review_score_distribution["Mixed"] = score_distribution_count_mix.text
        review_score_distribution["Negative"] = score_distribution_count_neg.text
        user_dict["author_review_distribution"].append(review_score_distribution)

        # calculate the number of game reviews for this user by adding them up
        num_game_reviews = int(score_distribution_count_pos.text) + int(score_distribution_count_mix.text) + int(
            score_distribution_count_neg.text)
    except Exception as e:
        print(f"ERROR: {e}")
        user_dict["author_review_distribution"].append({})
        num_game_reviews = 0

    user_dict["author_num_game_reviews"].append(num_game_reviews)

    # for compatibility reasons:
    user_dict['author_ratings_overall'].append(None)
    user_dict['author_reviews_overall'].append(None)


def scrape_reviews(requests_session, out_data: dict, game: str, platform: str):
    print(f"Scraping all reviews for game {game} ...")

    # we fetch all reviews (not only negative) and order them by date
    reviews_per_request = 100
    filter_type = "all"  # "negative"
    parameters = {
        "apiKey": "1MOZgmNFxvmljaQR1X9KAij9Mo4xAY3u", "filterBySentiment": filter_type, "sort": "date",
        "offset": 0, "limit": reviews_per_request, "componentName": "user-reviews", "componentTyp": "ReviewList",
    }
    user_reviews_url = f"https://fandom-prod.apigee.net/v1/xapi/reviews/metacritic/user/games/{game}/platform" \
                       f"/{platform}/web"

    # convert since and until into timestamps for easier comparison; both are given in the format "DD-MM-YYYY"
    start_date = datetime.strptime(since, '%d-%m-%Y').timestamp() if since is not None else None
    end_date = datetime.strptime(until, '%d-%m-%Y').timestamp() if until is not None else None

    current_reviews_fetched = 0
    more_reviews = True
    while more_reviews:
        response = requests_session.get(user_reviews_url, params=parameters, headers=headers)
        response_data = response.json()
        # pprint.pprint(response_data)
        # total_reviews = data["totalResults"]
        data = response_data["data"]
        review_list = data["items"]

        # update the http parameters for the next request
        parameters["offset"] += reviews_per_request
        current_reviews_fetched += reviews_per_request

        # break condition:
        if response_data["links"]["next"]["href"] is None:
            more_reviews = False

        # reviews are sorted by date; if the last review in this batch is newer than the given until date,
        # we can skip this batch completely
        last_review_in_batch = review_list[-1]
        last_review_date = last_review_in_batch["date"]
        last_review_timestamp = datetime.strptime(last_review_date, '%Y-%m-%d').timestamp()
        if (end_date is not None) and (last_review_timestamp > end_date):
            print(f"skipping review batch because the last review date ({last_review_date}) is newer than the given "
                  "end date")
            continue

        for review in review_list:
            # check first if the review is relevant for us if we are searching in a specific time period
            review_date = review["date"]
            review_timestamp = datetime.strptime(review_date, '%Y-%m-%d').timestamp()
            if (start_date is not None) and (review_timestamp < start_date):
                print(f"stop extracting because review date ({review_date}) is older than the given start date")
                # stop here completely because the reviews should be sorted by date so only older ones will follow
                more_reviews = False
                break
            if (end_date is not None) and (review_timestamp > end_date):
                print(f"skipping because review date ({review_date}) is newer than the given end date")
                continue

            review_date_formatted = datetime.strptime(review_date, '%Y-%m-%d').strftime('%d.%m.%Y')
            review_text = review["quote"]
            review_score = review["score"]
            review_thumbs_up = review["thumbsUp"] if review["thumbsUp"] != "null" else None
            review_thumbs_down = review["thumbsDown"] if review["thumbsDown"] != "null" else None
            review_author = review["author"]

            out_data['review'].append(review_text)
            out_data['rating'].append(review_score)
            out_data['review_date'].append(review_date_formatted)
            out_data['helpful_votes'].append(review_thumbs_up)
            out_data['unhelpful_votes'].append(review_thumbs_down)
            out_data['username'].append(review_author)

            try:
                # also try to scrape the user information for the review author (as every user can only review once per
                # game there won't be any unnecessary duplicates)
                scrape_user_profile(review_author, out_data)
            except Exception as e:
                print(f"ERROR while trying to scrape user information: {e}")
                out_data["author_average_score"].append("")
                out_data["author_review_distribution"].append({})
                out_data["author_num_game_reviews"].append(0)

        yield current_reviews_fetched
        time.sleep(3)


def scrape_metacritic():
    Output_Folder = pathlib.Path(__file__).parent / "metacritic_data"

    # create new folder for the scraped data
    if not Output_Folder.is_dir():
        Output_Folder.mkdir()

    # create a session so request don't have to open a new connection on every request which causes a huge overhead,
    # see https://thehftguy.com/2020/07/28/making-beautifulsoup-parsing-10-times-faster/
    requests_session = requests.Session()

    for game, platforms in games_to_scrape.items():
        print("\n####################")
        print("Game: " + game)
        Game_Folder = Output_Folder / game

        if not Game_Folder.is_dir():
            # create a subfolder for this game
            Game_Folder.mkdir()
        else:
            print("WARNING: Game folder already exists!")
            # if there is already one for this game
            # ask user if the old folder should be overwritten
            answer = input(f"Do you want to overwrite the existing folder for \"{game}\"? [y/n]\n")
            if str.lower(answer) == "y" or str.lower(answer) == "yes":
                shutil.rmtree(Game_Folder)
                Game_Folder.mkdir()
            else:
                print(f"Skipping game \"{game}\"")
                continue

        for platform in platforms:
            print("\nPlatform: " + platform)
            game_info_dict = {'game_title': game, 'game_description': [], 'platform': platform, 'release_date': [],
                              'critic_score': [], 'user_score': [], 'num_user_ratings': [],
                              'user_score_distribution': []}

            # The number of (un)helpful votes as well as the overall ratings and reviews of a user are not available
            # anymore on the new Metacritic Website but are kept for compatibility with the already extracted reviews
            # from the old Metacritic website.
            game_review_dict = {'username': [], 'review_date': [], 'helpful_votes': [], 'unhelpful_votes': [],
                                'rating': [], 'review': [], 'author_ratings_overall': [],
                                'author_reviews_overall': [], 'author_num_game_reviews': [],
                                'author_average_score': [], 'author_review_distribution': []}

            # scrape the non-review related information first
            scrape_general_game_information(requests_session, game, game_info_dict)
            game_description = scrape_game_description(requests_session, game)
            game_info_dict["game_description"].append(game_description)

            game_info_df = pd.DataFrame.from_dict(game_info_dict)   # , orient='index').transpose()
            game_info_df.to_csv(Game_Folder / f"game_info_{platform}_{game}.csv", index=False)

            output_path = Game_Folder / f"user_reviews_{platform}_{game}.csv"
            for currently_fetched_reviews in scrape_reviews(requests_session, game_review_dict, game, platform):
                print(f"\nCurrently fetched {currently_fetched_reviews} reviews! Saving ...")
                review_batch_df = pd.DataFrame(game_review_dict)
                review_batch_df.to_csv(output_path, mode="a", index=False, header=not os.path.exists(output_path))

                # clear dictionary to prevent duplicates in the csv file
                for value in game_review_dict.values():
                    value.clear()

        print("####################\n")


if __name__ == "__main__":
    print("Start scraping data from metacritic ...\n")
    scrape_metacritic()
