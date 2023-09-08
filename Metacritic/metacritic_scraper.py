#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pathlib
import shutil
import time
import re
import pandas as pd
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from enum import Enum
from useful_code_from_other_projects.FPSMeasurer import timeit


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
    "overwatch-2": [PLATFORM.PC.value],
}

SCRAPE_ALL = True  # if False use time period instead of scraping all user reviews
since = "01-03-2022"  # "start date", i.e. the oldest date
until = "01-07-2023"  # "end date", i.e. the most recent date

headers = {'User-agent': 'Mozilla/5.0'}

############################################################


def get_number_of_review_pages(soup: BeautifulSoup):
    """
    determine how many user review pages exist for a particular game + platform
    """
    num_pages = 1
    try:
        last_page_btn = soup.find("li", class_="page last_page")
        num_pages = int(last_page_btn.contents[-1].text)
    except Exception as e:
        print(e)

    return num_pages


@timeit
def scrape_general_game_information(soup: BeautifulSoup, out_data: dict):
    """
    Scrape general game information
    """
    side_content = soup.find("div", id="side")
    try:
        critic_score = side_content.find("div", class_="metascore_w").find("span").text
    except Exception as e:
        print(f"ERROR: {e}")
        critic_score = ""
    print(f"Critic Score: {critic_score}")
    out_data["critic_score"].append(critic_score)

    main_content = soup.find("div", id="main")
    try:
        game_release_date = main_content.find("li", class_="release_data").find("span", class_="data").text
        formatted_release_date = datetime.strptime(game_release_date, '%b %d, %Y').strftime('%d.%m.%Y')
    except Exception as e:
        print(f"ERROR: {e}")
        formatted_release_date = ""
    print(f"Release Date: {formatted_release_date}")
    out_data["release_date"].append(formatted_release_date)

    user_score_summary = main_content.find("div", class_="score_details userscore_details")
    try:
        user_score = user_score_summary.find("div", class_="metascore_w").text
        user_score = user_score if user_score != "tbd" else ""
    except Exception as e:
        print(f"ERROR: {e}")
        user_score = ""
    print(f"User Score: {user_score}")
    out_data["user_score"].append(user_score)

    try:
        num_user_ratings = user_score_summary.find("div", class_="summary").find("span", class_="count").find(
            "strong").text
        num_ratings = re.findall(r'\d+', num_user_ratings)[0]  # extract the digits from the string
    except Exception as e:
        print(f"ERROR: {e}")
        num_ratings = ""
    print(f"Number of User Ratings: {num_ratings}")
    out_data["num_user_ratings"].append(num_ratings)

    try:
        user_score_distribution = {}
        user_score_distribution_entries = user_score_summary.find("ol", class_="score_counts").findAll("li", class_="score_count")

        for entry in user_score_distribution_entries:
            score_distribution_label = entry.find("span", class_="label")
            score_distribution_label = score_distribution_label.text[:-1]  # remove the last character from the string
            score_distribution_count = entry.find("span", class_="count")

            # print("User Score Distribution:")
            # print(f"    * {score_distribution_label} {score_distribution_count.text}")
            user_score_distribution[score_distribution_label] = score_distribution_count.text
    except Exception as e:
        print(f"ERROR: {e}")
        user_score_distribution = ""

    out_data["user_score_distribution"].append(user_score_distribution)


def scrape_game_description(requests_session, game: str, platform: str = "pc"):
    game_details_url = f"https://www.metacritic.com/game/{platform}/{game}/details"
    response = requests_session.get(game_details_url, headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')
    try:
        game_description = soup.find("div", class_="summary_detail product_summary").find("span", class_="data").text
    except Exception as e:
        print(f"ERROR: {e}")
        game_description = ""

    return game_description


def scrape_user_profile(username: str, user_dict: dict):
    user_profile_url = f"https://www.metacritic.com/user/{username}?myscore-filter=Game"
    response = requests.get(user_profile_url, headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')

    main_content = soup.find("div", id="main")
    try:
        ratings_overall = main_content.find("span", class_="total_summary_ratings mr20").find("span",
                                                                                              class_="data").text
        user_dict["author_ratings_overall"].append(ratings_overall)
    except Exception as e:
        print(f"ERROR: {e}")
        user_dict["author_ratings_overall"].append("")

    try:
        reviews_overall = main_content.find("span", class_="total_summary_reviews").find("span", class_="data").text
        user_dict["author_reviews_overall"].append(reviews_overall)
    except Exception as e:
        print(f"ERROR: {e}")
        user_dict["author_reviews_overall"].append("")

    try:
        # the 'selected' class is only available because we set 'myscore-filter=Game' in the request above !
        num_game_reviews = main_content.find("span", class_="tab_title selected").find("span").text
        user_dict["author_num_game_reviews"].append(num_game_reviews)
    except Exception as e:
        print(f"ERROR: {e}")
        user_dict["author_num_game_reviews"].append("")

    try:
        average_review_score = main_content.find("div", class_="review_average").find("span", class_="summary_data").find("span").text
        user_dict["author_average_score"].append(average_review_score)
    except Exception as e:
        print(f"ERROR: {e}")
        user_dict["author_average_score"].append("")

    try:
        review_score_distribution = {}
        score_distribution_el = main_content.find("div", class_="score_distribution")
        distribution_entries = score_distribution_el.find("ol", class_="score_counts").findAll("li",
                                                                                               class_="score_count")
        for entry in distribution_entries:
            score_distribution_label = entry.find("span", class_="label")
            score_distribution_label = score_distribution_label.text[:-1]  # remove the ':' at the end of the string
            score_distribution_count = entry.find("span", class_="count")

            # print(f"    * {score_distribution_label} {score_distribution_count.text}")
            review_score_distribution[score_distribution_label] = score_distribution_count.text

        user_dict["author_review_distribution"].append(review_score_distribution)
    except Exception as e:
        print(f"ERROR: {e}")
        user_dict["author_review_distribution"].append("")


# Util-Method to extract the creation date from a user review and return it as a datetime object
def get_datetime_from_review(review):
    review_date = review.find('div', class_='date').text
    return datetime.strptime(review_date, '%b %d, %Y')   # the date format is "Jun 8, 2023"


@timeit
def scrape_user_review(review, out_data: dict):
    """
    Code in this method is based on the article
    https://towardsdatascience.com/web-scraping-metacritic-reviews-using-beautifulsoup-63801bbe200e
    """
    username_el = review.find('div', class_='name')
    if username_el is None:
        return
    # username can either be in an <a> or <span> element
    username_tag = username_el.find('a')
    if username_tag is None:
        username_tag = username_el.find('span')
    username = username_tag.text
    out_data['username'].append(username)

    try:
        # also try to scrape the user information for the extracted username (as every user can only review once per
        # game there won't be any unnecessary duplicates)
        scrape_user_profile(username, out_data)
    except Exception as e:
        print(f"ERROR while trying to scrape user information: {e}")

    out_data['rating'].append(review.find('div', class_='review_grade').find_all('div')[0].text)
    formatted_created_date = get_datetime_from_review(review).strftime('%d.%m.%Y')
    out_data['review_date'].append(formatted_created_date)

    # extract how many found this review useful
    helpful_summary = review.find('div', class_='helpful_summary')
    total_ups = helpful_summary.find('span', class_='total_ups').text
    total_thumbs = helpful_summary.find('span', class_='total_thumbs').text
    out_data['helpful_votes'].append(total_ups)
    out_data['unhelpful_votes'].append(f'{int(total_thumbs) - int(total_ups)}')

    # Longer reviews (that require users to click on ‘Expand’ to see the full text) are within a blurb blurb_expanded
    # span class. Shorter reviews do not have this class.
    try:
        if review.find('span', class_='blurb blurb_expanded'):
            out_data['review'].append(review.find('span', class_='blurb blurb_expanded').text)
        else:
            out_data['review'].append(review.find('div', class_='review_body').find('span').text)
    except Exception as e:
        print(f"ERROR while trying to scrape review text: {e}")
        out_data['review'].append("")


"""
def scrape_until_start_date(start_date, start_page_idx: int, num_pages: int, metacritic_url, out_data):
    for page in range(start_page_idx, num_pages):
        querystring = {"page": page, "sort-by": "date"}
        response = requests.get(metacritic_url, headers=headers, params=querystring)
        time.sleep(5)

        soup = BeautifulSoup(response.text, 'html.parser')

        for review in soup.find_all('div', class_='review_content'):
            review_date = get_datetime_from_review(review)
            # if the date of the current review is older than the oldest date we are looking for, stop scraping
            if review_date < start_date:
                return

            scrape_user_review(review, out_data)
"""


def find_newest_relevant_review_on_page(all_reviews, end_date):
    """
    This function tries to find the most recent relevant review for a given end date by recursively splitting all
    reviews on the page in two sections and comparing the review date with the end date.
    """
    if len(all_reviews) == 0:
        return None
    if len(all_reviews) == 1:
        return all_reviews[0]

    # get middle element of list
    middle_index = len(all_reviews) // 2
    middle_review = all_reviews[middle_index]
    review_date = get_datetime_from_review(middle_review)

    # find out which side of the list is relevant for us, and repeat the search recursively
    if end_date > review_date:
        # first half is relevant, the end date is newer than the middle review date
        relevant_reviews = all_reviews[:middle_index]
    elif end_date == review_date:
        # this could probably be written a lot cleaner, but it does the job
        # Alternative: without this elif case (-> combining with the one above) we would find the first item that
        # DOES NOT match, so we could simply take idx + 1 afterwards instead
        if len(all_reviews) > 2:
            relevant_reviews = all_reviews[:middle_index + 1]
        else:
            relevant_reviews = all_reviews[middle_index:]
    else:
        # second half is relevant, the end date is older than the middle review date
        relevant_reviews = all_reviews[middle_index:]

    return find_newest_relevant_review_on_page(relevant_reviews, end_date)


# there are cases where the sorting by Metacritic itself is wrong, e.g. random wrong date between correctly sorted ones
def scrape_time_period(requests_session, out_data: dict, metacritic_url: str, num_pages: int):
    # since and until must be in the form "DD-MM-YYYY"
    start_date = datetime.strptime(since, '%d-%m-%Y')
    end_date = datetime.strptime(until, '%d-%m-%Y')

    # start_page = -1
    for page in range(0, num_pages):
        print(f"Scraping page {page + 1} from {num_pages} ...")
        querystring = {"page": page, "sort-by": "date"}
        response = requests_session.get(metacritic_url, headers=headers, params=querystring)

        time.sleep(4)
        soup = BeautifulSoup(response.text, 'lxml')

        # The idea of the algorithm is to first find the end date of the given time period (i.e. the newest relevant
        # review) and the simply iterate over all reviews until the start date (i.e. the oldest relevant review) is
        # reached.
        # TODO alternative (less efficient?): iterate through all in a loop until end and start date encountered and
        #  only scrape the reviews between them

        # the first and last review on each page have an additional class "first_review" / "last_review" respectively
        all_reviews_on_page = soup.find('ol', class_='reviews user_reviews').findAll("li", class_=["review",
                                                                                                   "user_review"])
        if len(all_reviews_on_page) == 0:
            return
        # reviews were sorted by date so the first review is also the most recent
        first_review = all_reviews_on_page[0]
        last_review = all_reviews_on_page[-1]  # alternatively: find("li", class_="last_review")

        # for debugging:
        """
        author = first_review.find('div', class_='name').find('a').text
        print(f"First review on page {page} is from author {author}")
        author = last_review.find('div', class_='name').find('a').text
        print(f"Last review on page {page} is from author {author}")
        """

        if first_review.find('div', class_='name') is None:
            return
        newest_date = get_datetime_from_review(first_review)
        oldest_date = get_datetime_from_review(last_review)

        if oldest_date > end_date:
            # if the oldest date on this page is greater (i.e. more recent) than the end date of the wanted time
            # period, this page is irrelevant so switch to the next
            print(f"Oldest Date ({oldest_date}) on page {page+1} is more recent than end date ({end_date}); continuing "
                  f"to next page ...")
            continue
        else:
            # if the oldest date on this page is lower than or equal to the end date, the newest relevant review is
            # on this page

            # if the newest date is older than or equal to the end_date, the newest relevant review is the first review
            # on the page; if not we have to manually search for the first relevant review
            if newest_date <= end_date:
                print(f"Page {page+1}: newest_date <= end_date; first_index = 0")
                first_index = 0
            else:
                first_relevant_review = find_newest_relevant_review_on_page(all_reviews_on_page, end_date)
                first_index = all_reviews_on_page.index(first_relevant_review)
                print(f"Page {page+1}: newest_date > end_date; first_index = {first_index}")

            # scrape the relevant reviews on this page
            relevant_reviews_on_page = all_reviews_on_page[first_index:]

            for review in relevant_reviews_on_page:
                review_date = get_datetime_from_review(review)
                # if the date of the current review is older than the oldest date we are looking for, stop scraping
                if review_date < start_date:
                    print(f"Review ({review_date}) on page {page+1} is older than start date ({start_date}). "
                          f"Stopping...")
                    return

                scrape_user_review(review, out_data)
    """
            start_page = page
            break

    if start_page != -1:
        scrape_until_start_date(start_date, start_page, num_pages, metacritic_url, out_data)
    """


@timeit
def scrape_all_reviews(requests_session, out_data: dict, metacritic_url: str, num_pages: int):
    # per default 100 reviews per page are returned
    for page in range(0, num_pages):
        print(f"Scraping page {page + 1} from {num_pages} ...")
        querystring = {"page": page, "sort-by": "date"}  # alternatively sort by "score"
        response = requests_session.get(metacritic_url, headers=headers, params=querystring)

        # sleep for a few seconds after each request to mitigate the risk of getting our IP address blocked
        time.sleep(5)

        soup = BeautifulSoup(response.text, 'lxml')  # "html.parser")
        # search for review_content tags since every user review has this tag
        for review in soup.find_all('div', class_='review_content'):
            scrape_user_review(review, out_data)
            # break  # for testing only use the first review per page

        yield page  # notify for each page we finished scraping


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

            game_review_dict = {'username': [], 'review_date': [], 'helpful_votes': [], 'unhelpful_votes': [],
                                'rating': [], 'review': [], 'author_ratings_overall': [], 'author_reviews_overall': [],
                                'author_num_game_reviews': [], 'author_average_score': [],
                                'author_review_distribution': []}

            metacritic_url = f"https://www.metacritic.com/game/{platform}/{game}/user-reviews"
            # make a first request to the metacritic user reviews page for this game & platform to fetch general
            # information about the game as well as the overall number of user review pages
            response = requests_session.get(metacritic_url, headers=headers)
            soup = BeautifulSoup(response.text, 'lxml')

            # scrape the non-review related information here, once for each platform
            scrape_general_game_information(soup, game_info_dict)
            game_description = scrape_game_description(requests_session, game, platform)
            game_info_dict["game_description"].append(game_description)

            # game_info_df = pd.DataFrame.from_dict(game_info_dict, orient='index').transpose()
            game_info_df = pd.DataFrame.from_dict(game_info_dict)
            game_info_df.to_csv(Game_Folder / f"game_info_{platform}_{game}.csv", index=False)

            # extract the number of user review pages
            num_pages = get_number_of_review_pages(soup)
            print(f"Overall Number of Review Pages: {num_pages}")

            output_path = Game_Folder / f"user_reviews_{platform}_{game}.csv"
            if SCRAPE_ALL:
                for page in scrape_all_reviews(requests_session, game_review_dict, metacritic_url, num_pages):
                    print(f"\nFinished scraping page {page + 1}! Saving reviews ...")
                    review_batch_df = pd.DataFrame(game_review_dict)
                    review_batch_df.to_csv(output_path, mode="a", index=False,
                                           header=not os.path.exists(output_path))

                    # clear dictionary to prevent duplicates in the csv file
                    for value in game_review_dict.values():
                        value.clear()
            else:
                scrape_time_period(requests_session, game_review_dict, metacritic_url, num_pages)
                user_reviews_df = pd.DataFrame(game_review_dict)
                user_reviews_df.to_csv(output_path, index=False)

        print("####################\n")


if __name__ == "__main__":
    print("Start scraping data from metacritic ...\n")
    scrape_metacritic()
    print("\nFinished scraping!")
