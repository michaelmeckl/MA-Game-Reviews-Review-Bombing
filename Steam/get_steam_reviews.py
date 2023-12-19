#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import os
import pathlib
import re
import shutil
from datetime import datetime
import time
import pprint
import requests
import pandas as pd
from smart_open import open
from ssl import SSLError
from dotenv import dotenv_values

OUTPUT_FOLDER = pathlib.Path(__file__).parent / "steam_review_data"
APP_LIST_FILE = "steam_app_list.csv"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 "
                  "Safari/537.36",
}

# load the steam web api key from a .env file (only needed for fetching user information)
config = dotenv_values()
steam_api_key = config["STEAM_API_KEY"]


def get_request(url, parameters=None):
    """ Method taken from https://nik-davis.github.io/posts/2019/steam-data-collection/
    Return json-formatted response of a get request using optional parameters.

    Parameters
    ----------
    url : string
    parameters : {'parameter': 'value'}
        parameters to pass as part of get request

    Returns
    -------
    json_data
        json-formatted response (dict-like)
    """
    try:
        response = requests.get(url=url, params=parameters)
    except SSLError as s:
        print('SSL Error:', s)

        for i in range(5, 0, -1):
            print('\rWaiting... ({})'.format(i), end='')
            time.sleep(1)
        print('\rRetrying.' + ' ' * 10)

        # recursively try again
        return get_request(url, parameters)
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError,
            requests.exceptions.Timeout, requests.exceptions.RequestException) as ex:
        print(f'ERROR: An exception of type {type(ex).__name__} ocurred.')
        response = None
        return response

    if response:
        return response.json()
    else:
        # response is none usually means too many requests. Wait and try again
        print('No response, waiting 10 seconds...')
        time.sleep(10)
        print('Retrying.')
        return get_request(url, parameters)


def get_steam_app_list():
    """
    Load all apps with the corresponding name from Steam and save the result.
    """
    print('Loading list of app ids from Steam ...')
    url = 'http://api.steampowered.com/ISteamApps/GetAppList/v2/'
    data = get_request(url)

    if data:
        apps = data['applist']['apps']
        app_list_df = pd.DataFrame(apps)
        app_list_df = app_list_df.sort_values(by='appid')
        app_list_df.to_csv(APP_LIST_FILE, index=False)

    print('Finished loading app ids from Steam!')


def get_steam_review_graph_data(app_id):
    """
    This loads the review timeseries data for the graphs shown in the Steam Store Pages at the bottom for each game
    """
    print(f"Requesting review timeseries data for app id {app_id} ...")
    resp = requests.get(f'https://store.steampowered.com/appreviewhistogram/{app_id}?language=english,german')
    data = resp.json()
    if int(data['success']) != 1:
        raise ConnectionRefusedError()

    results = data['results']
    histogramm_data = {
        "start_date": results['start_date'], "end_date": results['end_date'], "weeks": results['weeks'],
        "review_data": results['rollups'], "timeseries_type": results['rollup_type']
    }
    return histogramm_data


def get_id_for_game(game_dataframe: pd.DataFrame, game_name: str):
    """
    Returns the corresponding id for the game with the given name.
    """
    try:
        matching_entries = game_dataframe.loc[game_dataframe['name'] == game_name, 'appid']
        print(f"Found the following entries on steam for the name \"{game_name}\":\n{matching_entries}")
        game_id = matching_entries.iloc[0]    # take the first match if there were more than 1
    except (KeyError, pd.errors.IndexingError) as e:
        print(f"Error when trying to get for game {game_name}: {e}")
        game_id = None

    return game_id


def cleanup_text(text):
    """
    Removes HTML codes, escape codes and URLs.
    Method taken from https://github.com/FronkonGames/Steam-Games-Scraper
    """
    text = text.replace('\n\r', ' ')
    text = text.replace('\r\n', ' ')
    text = text.replace('\r \n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('&quot;', "'")
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
    text = re.sub('<[^<]+?>', ' ', text)
    text = re.sub(' +', ' ', text)
    text = text.lstrip(' ')
    return text


def price_to_float(price, decimals=2):
    """
    Convert the price in text to a float.
    Method taken from https://github.com/FronkonGames/Steam-Games-Scraper
    """
    price = price.replace(',', '.')
    return round(float(re.findall('([0-9]+[,.]+[0-9]+)', price)[0]), decimals)


def get_general_information_for_game(game_id: str):
    """
    Returns general information for the game, i.e. description, developer, price, etc.
    """
    game_info_dict = {}
    print(f'Loading general information for game with id {game_id} ...')

    url = "http://store.steampowered.com/api/appdetails"
    data = get_request(url, parameters={"appids": game_id, "cc": "EUR"})
    if data:
        app = data[game_id]
        if not app["success"]:
            return game_info_dict
        game_infos = app["data"]

        game_info_dict["game_id"] = game_id
        game_info_dict["title"] = game_infos["name"].strip()
        game_info_dict["release_date"] = game_infos["release_date"]["date"] if "release_date" in game_infos and not \
            game_infos["release_date"]["coming_soon"] else ""
        game_info_dict["release_date"] = datetime.strptime(game_info_dict["release_date"], "%d %b, %Y").strftime("%d.%m.%Y")

        if game_infos["is_free"] or "price_overview" not in game_infos:
            price = 0.0
        else:
            price = price_to_float(game_infos["price_overview"]["final_formatted"])
        game_info_dict["price_euro"] = price

        game_info_dict["detailed_description"] = game_infos[
            "detailed_description"].strip() if "detailed_description" in game_infos else ""
        game_info_dict["short_description"] = game_infos[
            "short_description"].strip() if "short_description" in game_infos else ""
        # cleanup the text by removing html etc.
        game_info_dict["detailed_description"] = cleanup_text(game_info_dict["detailed_description"])
        game_info_dict["short_description"] = cleanup_text(game_info_dict["short_description"])

        game_info_dict["developers"] = []
        if "developers" in game_infos:
            for developer in game_infos["developers"]:
                game_info_dict["developers"].append(developer.strip())

        game_info_dict["publishers"] = []
        if "publishers" in game_infos:
            for publisher in game_infos["publishers"]:
                game_info_dict["publishers"].append(publisher.strip())

    return game_info_dict


def get_user_information(user_id, user_info_dict: dict):
    """
    Fetches information for this steam user. This part needs a Steam Web API Key.
    For documentation, see https://developer.valvesoftware.com/wiki/Steam_Web_API#Game_interfaces_and_methods
    """
    print(f'Loading user information for user with id {user_id} ...')

    url = f"http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2/?key={steam_api_key}&steamids=" \
          f"{user_id}&format=json"
    user_data = get_request(url)
    if user_data:
        # pprint.pprint(user_data)
        data = user_data["response"]["players"][0]  # take the first since we search for only 1 user anyway

        # get public information
        username = data["personaname"] if "personaname" in data else ""
        profile_url = data["profileurl"] if "profileurl" in data else ""
        user_country_code = data["loccountrycode"] if "loccountrycode" in data else ""
        last_online = data["lastlogoff"] if "lastlogoff" in data else ""  # last time the user was online, in unix time
        if last_online != "":
            last_online = datetime.fromtimestamp(last_online).strftime('%d.%m.%Y %H:%M:%S')
        user_info_dict["author_username"] = username
        user_info_dict["profile_url"] = profile_url
        user_info_dict["author_country_code"] = user_country_code
        user_info_dict["author_last_online"] = last_online

        # check profile visibility, if user profile is private only public information is available
        profile_visibility = data["communityvisibilitystate"]  # either 1 for private or 3 for public profile
        user_info_dict["profile_visibility"] = "private" if profile_visibility == 1 else "public"

        if profile_visibility == 1:
            print(f"Profile for user {user_id} is private")
        else:
            # get private information
            user_real_name = data["realname"] if "realname" in data else ""
            profile_created = data["timecreated"] if "timecreated" in data else ""
            if profile_created != "":
                profile_created = datetime.fromtimestamp(profile_created).strftime('%d.%m.%Y %H:%M:%S')
            user_info_dict["author_real_name"] = user_real_name
            user_info_dict["profile_created"] = profile_created

            # try to get information about friends (works only if profile is public)
            try:
                friends_url = f"http://api.steampowered.com/ISteamUser/GetFriendList/v1/?key=" \
                              f"{steam_api_key}&steamid={user_id}&relationship=friend&format=json"
                response = requests.get(url=friends_url, params=None)
                friends_data = response.json()
                if friends_data:
                    all_friends = friends_data["friendslist"]["friends"]
                    user_info_dict["author_num_friends"] = len(all_friends)
            except Exception as e:
                print(f"ERROR while trying to fetch friends information for user: {e}")

            # try to get information about the user's owned games (works only if profile is public)
            try:
                owned_games_url = f"http://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key=" \
                                  f"{steam_api_key}&steamid={user_id}&include_played_free_games=1&format=json"
                response = requests.get(url=owned_games_url, params=None)
                data = response.json()
                owned_games_data = data["response"]
                if owned_games_data:
                    game_count = owned_games_data["game_count"]
                    user_info_dict["author_num_owned_games"] = game_count
            except Exception as e:
                print(f"ERROR while trying to fetch owned games information for user: {e}")

            # also try to get the user's steam level
            try:
                steam_level_url = f"http://api.steampowered.com/IPlayerService/GetSteamLevel/v1/?key=" \
                                  f"{steam_api_key}&steamid={user_id}&format=json"
                response = requests.get(url=steam_level_url, params=None)
                data = response.json()
                level_data = data["response"]
                if level_data:
                    steam_level = level_data["player_level"]
                    user_info_dict["author_steam_level"] = steam_level
            except Exception as e:
                print(f"ERROR while trying to get the user's steam level: {e}")
    else:
        print("ERROR: Failed to load json data for the steam user information!")


def parse_steam_reviews(result, user_reviews, already_extracted_reviews):
    for review in result["reviews"]:
        try:
            if review["written_during_early_access"]:
                # filter out early-access reviews
                continue

            review_id = review['recommendationid']  # eindeutige ID des Reviews
            if review_id in already_extracted_reviews:
                # make sure we don't extract the same review again
                continue

            already_extracted_reviews.add(review_id)

            review_created_date = review['timestamp_created']  # Erstellungsdatum des Reviews (Unix-Zeitstempel)
            review_last_update_date = review['timestamp_updated']  # Datum der letzten Aktualisierung (Unix-Zeitstempel)
            review_created_date_formatted = datetime.fromtimestamp(review_created_date).strftime('%d.%m.%Y %H:%M:%S')
            review_last_update_date_formatted = datetime.fromtimestamp(review_last_update_date).strftime('%d.%m.%Y '
                                                                                                         '%H:%M:%S')

            review_text = review['review']  # Text des geschriebenen Reviews
            review_author_id = review['author']['steamid']  # die Steam-ID des Autors
            # Anzahl der Spiele, die der Nutzer besitzt; leider relativ fehlerhaft (z.B. 0 bei privaten Profilen, ...)
            review_author_games_owned = review['author']['num_games_owned']
            review_author_num_reviews = review['author']['num_reviews']  # die Anzahl der vom Nutzer verfassten Reviews
            # in der Anwendung erfasste Spielzeit in Minuten (insgesamt)
            review_author_playtime_overall = review['author']['playtime_forever']
            # Spielzeit in Minuten zum Zeitpunkt der Reviewverfassung
            review_author_playtime_at_review = review['author']['playtime_at_review']
            # voted_up true bedeutet, dass es sich um eine positive Empfehlung handelt
            review_positive = review['voted_up']
            review_score_useful = review['votes_up']  # die Anzahl der Nutzer, die dieses Review hilfreich fanden
            review_score_funny = review['votes_funny']  # die Anzahl der Nutzer, die dieses Review lustig fanden
            review_weighted_score = review['weighted_vote_score']  # Wie hilfreich ist Review (Formel nicht öffentlich)
            review_comment_count = review['comment_count']  # Anzahl der zu diesem Review verfassten Kommentare
            review_steam_purchase = review['steam_purchase']  # true, wenn der Nutzer das Spiel auf Steam erworben hat
            # true, wenn vom Nutzer angegeben wurde, dass dieser das Spiel kostenlos erhalten hat
            review_received_for_free = review['received_for_free']

            user_info_data = {
                "author_username": None, "author_real_name": None, "profile_created": None, "author_last_online": None,
                "profile_visibility": None, "author_country_code": None, "author_num_friends": None,
                "author_num_owned_games": None, "author_steam_level": None, "profile_url": None,
            }
            try:
                get_user_information(review_author_id, user_info_dict=user_info_data)
            except Exception as e:
                print(f"Error while trying to fetch user information: {e}")

            review_data = {
                "review_id": review_id, "content": review_text, "rating_positive": review_positive,
                "created_at": review_created_date, "created_at_formatted": review_created_date_formatted,
                "last_updated": review_last_update_date, "last_updated_formatted": review_last_update_date_formatted,
                "weighted_score": review_weighted_score, "useful_score": review_score_useful,
                "funny_score": review_score_funny, "comment_count": review_comment_count,
                "was_steam_purchase": review_steam_purchase, "game_received_for_free": review_received_for_free,
                "author_id": review_author_id, "author_playtime_overall_min": review_author_playtime_overall,
                "author_playtime_at_review_min": review_author_playtime_at_review,
                "author_num_reviews": review_author_num_reviews,
            }
            review_data = review_data | user_info_data   # merge the two dictionaries (operator requires Python 3.9+)
            user_reviews.append(review_data)

        except Exception as e:
            print(f"ERROR: {e}")


def get_steam_user_reviews(app_id, requests_session, next_cursor, current_num_reviews, since=None, until=None,
                           steam_off_topic_reviews_included=True):
    """
    See https://andrew-muller.medium.com/scraping-steam-user-reviews-9a43f9e38c92 and
    https://partner.steamgames.com/doc/store/getreviews

    @:param app_id: the id of the game from which the user reviews should be fetched
    @:param steam_off_topic_reviews_enabled: whether off-topic reviews as marked by steam should be fetched too
    """
    already_extracted_reviews = set()
    more_reviews = True
    num_reviews_fetched = 0 if current_num_reviews is None else current_num_reviews

    off_topic_code = 0 if steam_off_topic_reviews_included else 1
    cursor = "*" if next_cursor is None else next_cursor  # start with cursor '*' and update after each request
    review_type = "negative"  # positive, negative, all
    purchase_type = "all"  # steam, non_steam_purchase, all
    num_per_page = 100  # default is 20, max is 100
    sort = "recent"  # recent, updated, all
    # TODO only search for reviews written in under 2 hours of playtime ? (i.e. refund duration)
    playtime_min = "0"  # playtime in hours; 0 can either mean 0 hours or no limit (when used as max)
    playtime_max = "0"
    # default date is "-1", which means no specific start / end date
    start_date = datetime.strptime(since, '%d-%m-%Y').timestamp() if since is not None else -1
    end_date = datetime.strptime(until, '%d-%m-%Y').timestamp() if until is not None else -1
    # day_range = "30"  # max is 365, search between now and number of days to go back (only works with filter "all")

    game_reviews_url = f"https://store.steampowered.com/appreviews/{app_id}"
    parameters = {"json": "1",
                  "cursor": f"{cursor}",
                  "num_per_page": f"{num_per_page}",
                  # "day_range": f"{day_range}",
                  "start_date": f"{start_date}", "end_date": f"{end_date}", "date_range_type": "include",
                  "filter": f"{sort}",
                  # TODO filter language?
                  "language": "all",   # "german,english"
                  "review_type": f"{review_type}",
                  "purchase_type": f"{purchase_type}",
                  "playtime_filter_min": f"{playtime_min}", "playtime_filter_max": f"{playtime_max}",
                  "filter_offtopic_activity": f"{off_topic_code}",
                  }

    while more_reviews:
        print(f"Fetching new set of user reviews ... (already fetched {num_reviews_fetched} reviews)")
        try:
            response = requests_session.get(game_reviews_url, headers=headers, params=parameters)
            result = response.json()
        except Exception as e:
            print(f"ERROR: Exception occurred during request user reviews from steam: {e}")
            break

        # query summary shows general information for all reviews for this game (most of the fields in
        # there are only available if the review_type filter above is set to "all")
        # print(f"Query Summary: {result['query_summary']}")

        # update cursor to fetch the next set of user reviews
        if result["cursor"] is None:
            break
        new_cursor = result["cursor"]
        parameters["cursor"] = new_cursor.encode()
        print("new cursor: ", new_cursor)
        num_new_reviews = len(result["reviews"])
        num_reviews_fetched += num_new_reviews

        # break condition: stop requesting data when no new reviews were fetched
        if num_new_reviews == 0:
            print("\nNo new reviews fetched; stop requesting data ...")
            more_reviews = False

        user_reviews = list()
        parse_steam_reviews(result, user_reviews, already_extracted_reviews)

        # yield results at the end of each loop and update local file (so we don't have to start from the beginning
        # again in case something interrupts the data fetching); also save current cursor + num_reviews_fetched
        yield user_reviews, new_cursor, num_reviews_fetched
        # wait for a few seconds before the next request
        time.sleep(2)

    print(f"---------------------------\nFinished extracting steam user reviews for app id {app_id}!")


def load_reviews_for_games():
    # load steam user reviews for certain games
    # specify the timeperiod to extract reviews from or use 'None' to specify no time period
    # the end date is NOT included! (i.e. for end date 21-01 the actual end date is 20-01)
    game_dict = {
        # "Hogwarts Legacy": ("06-02-2023", None),
        # "Cyberpunk 2077": ("09-12-2020", "17-12-2020"),
        # "Cyberpunk 2077": ("01-03-2022", "14-03-2022"),
        # "Cyberpunk 2077": ("01-01-2023", "01-02-2023"),
        # "Borderlands GOTY": ("01-04-2019", "30-04-2019"),
        # "Borderlands GOTY Enhanced": ("01-04-2019", "30-04-2019"),
        # "Borderlands: The Pre-Sequel": ("01-04-2019", "30-04-2019"),
        # "Borderlands 2": ("01-04-2019", "30-04-2019"),
        # "Borderlands 3": ("13-03-2020", "31-03-2020"),
        # "Metro 2033 Redux": ("28-01-2019", "14-02-2019"),
        # "Metro: Last Light Redux": ("28-01-2019", "14-02-2019"),
        # "Metro Exodus": ("15-02-2020", "28-02-2020"),
        # "Overwatch® 2": ("10-08-2023", "30-08-2023"),
        # "Firewatch": ("12-09-2017", "01-10-2017"),
        # "GWENT: The Witcher Card Game": ("01-03-2022", "15-03-2022"),
        # "Thronebreaker: The Witcher Tales": ("01-03-2022", "15-03-2022"),
        # "The Witcher: Enhanced Edition": ("01-03-2022", "15-03-2022"),  # ID 20900
        # "The Witcher 2: Assassins of Kings Enhanced Edition": ("01-03-2022", "15-03-2022"),
        # "The Witcher 3: Wild Hunt": ("01-03-2022", "17-03-2022"),
        # "Frostpunk": ("24-02-2022", "15-03-2022"),
        # "S.T.A.L.K.E.R.: Shadow of Chernobyl": ("01-03-2022", "01-04-2022"),  # ID 4500
        # "S.T.A.L.K.E.R.: Call of Pripyat": ("01-03-2022", "01-04-2022"),
        # "S.T.A.L.K.E.R.: Clear Sky": ("01-03-2022", "01-04-2022"),  # ID 20510
        # "The Elder Scrolls V: Skyrim": ("23-04-2015", "01-05-2015"),  # ID 72850
        # "The Elder Scrolls V: Skyrim Special Edition": ("20-09-2017", "01-12-2017"),  # ID 489830
        # "Fallout 4": ("29-08-2017", "01-12-2017"),  # ID 377160
        # "Grand Theft Auto V": ("14-06-2017", "01-07-2017"),  # ID 271590 (there is another empty entry for this title)
        # "Total War: ROME II - Emperor Edition": ("22-09-2018", "31-10-2018"),
        # "Mortal Kombat 11": ("22-04-2019", "01-05-2019"),
        # "Assassin's Creed Unity": ("19-04-2019", "01-05-2019"),  # here only positive reviews (was a positive RB!)
        # "Crusader Kings II": ("19-10-2019", "01-11-2019"),
        # "The Long Dark": ("02-03-2020", "01-05-2020"),
        # "SUPERHOT VR": ("21-07-2021", "18-09-2021"),
    }

    if os.path.exists(APP_LIST_FILE):
        app_list_df = pd.read_csv(APP_LIST_FILE)
        # ensure pandas tables show all columns
        pd.options.display.max_columns = 50

        # create a new folder for the steam review data
        if not OUTPUT_FOLDER.is_dir():
            OUTPUT_FOLDER.mkdir()

        # create a session so requests doesn't have to open a new connection on every request
        requests_session = requests.Session()

        for game, (since, until) in game_dict.items():
            app_id = get_id_for_game(app_list_df, game)
            if app_id is None:
                print(f"App ID for game {game} is None!")
                continue

            # characters like ':' need to be removed from the game title in order to use it as a filename
            game_name_cleaned = re.sub('[^A-Za-z0-9]+', "_", game)

            # set up a folder for this game or overwrite existing one optionally
            Game_Folder = OUTPUT_FOLDER / game_name_cleaned
            if Game_Folder.is_dir():
                print(f"WARNING: Folder \"{game_name_cleaned}\" already exists!")
                # ask if downloading reviews should resume or if existing data should be overriden ?
                answer = input("Do you want to overwrite the existing data? [y/n]\n")
                if str.lower(answer) == "y" or str.lower(answer) == "yes":
                    print(f"\nDeleting old folder \"{game_name_cleaned}\" ...")
                    shutil.rmtree(Game_Folder)
                    Game_Folder.mkdir()
                else:
                    print(f"\nResuming data collection for folder \"{game_name_cleaned}\" ...")
            else:
                Game_Folder.mkdir()

            # extract general information for this game (i.e. description and developers)
            general_game_info_path = Game_Folder / f"steam_general_info_{game_name_cleaned}.csv"
            if not general_game_info_path.exists():
                general_information = get_general_information_for_game(game_id=str(app_id))
                general_info_df = pd.DataFrame.from_dict(general_information, orient='index').transpose()
                general_info_df.to_csv(general_game_info_path, index=False)

            # get the timeseries data from the app review chart
            review_timeseries_path = Game_Folder / f"steam_reviews_timeseries_{game_name_cleaned}.json"
            if not review_timeseries_path.exists():
                review_timeseries = get_steam_review_graph_data(app_id)
                with open(review_timeseries_path, "w") as f:
                    json.dump(review_timeseries, f, indent=4)

            # create / open a text file in the game folder to store current progress while fetching reviews
            progress_file_path = Game_Folder / "current_progress.txt"
            progress_file_path.touch(exist_ok=True)  # create file if it does not exist yet

            progress_file = open(progress_file_path, "r+")  # r+ does not truncate file in contrast to w+
            progress_file.seek(0)
            if progress_file.read(1):
                # file isn't empty, get current progress
                progress_file.seek(0)
                file_content = progress_file.readlines()
                next_cursor = file_content[0].strip()
                current_num_reviews = int(file_content[1].strip())
            else:
                next_cursor = None
                current_num_reviews = None

            print(f"Requesting user reviews for game \"{game}\" ...\n")
            csv_file_path = Game_Folder / f"steam_user_reviews_{game_name_cleaned}.csv"
            # review_df = pd.DataFrame()

            for user_review_batch, next_cursor, current_num_reviews in get_steam_user_reviews(app_id,
                                                                                              requests_session,
                                                                                              next_cursor,
                                                                                              current_num_reviews,
                                                                                              since, until):
                # print(f"Fetched {len(user_review_batch)} user reviews\n")
                user_review_batch_df = pd.DataFrame(user_review_batch)
                # review_df = pd.concat([review_df, user_review_batch_df], ignore_index=True, verify_integrity=True)

                # only append to csv file so the existing reviews won't be overriden
                user_review_batch_df.to_csv(csv_file_path, mode="a", index=False,
                                            header=not os.path.exists(csv_file_path))

                # clear current progress file content and update it
                progress_file.seek(0)
                progress_file.writelines([f"{next_cursor}\n", f"{current_num_reviews}\n"])
                progress_file.truncate()

            # only append to csv file so the existing reviews won't be overriden
            # review_df.to_csv(csv_file_path, mode="a", index=False)

            progress_file.close()  # close the text file when done
    else:
        print(f"ERROR: {APP_LIST_FILE} does not exist!")


if __name__ == "__main__":
    load_steam_app_list = False
    load_reviews = True

    if load_steam_app_list:
        get_steam_app_list()
    if load_reviews:
        load_reviews_for_games()
