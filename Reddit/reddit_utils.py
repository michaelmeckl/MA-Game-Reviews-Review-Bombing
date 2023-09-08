import re
from datetime import datetime, timezone, timedelta
import pandas as pd
import praw
import prawcore
from praw.models import MoreComments
import pathlib
from smart_open import open

OUT_DATA = pathlib.Path(__file__).parent / "reddit_data"
REDDIT_BASE_URL = "https://www.reddit.com"


"""
def get_access_token():
    client_auth = requests.auth.HTTPBasicAuth('p-jcoLKBynTLew', 'gko_LXELoV07ZBNUXrvWZfzE3aI')
    post_data = {"grant_type": "password", "username": "reddit_bot", "password": "password"}
    headers = {"User-Agent": "ChangeMeClient/0.1 by YourUsername"}
    response = requests.post("https://www.reddit.com/api/v1/access_token", auth=client_auth, data=post_data,
                             headers=headers)
    pprint.pprint(response.json())

    headers = {"Authorization": "bearer fhTdafZI-0ClEzzYORfBSCR7x3M",
               "User-Agent": "ChangeMeClient/0.1 by YourUsername"}
    response = requests.get("https://oauth.reddit.com/api/v1/me", headers=headers)
    print(response.json())
"""


def get_user_information(redditor):
    user_info_dict = {
        "id": None, "username": None, "created_at": None, "created_at_formatted": None, "is_suspended": None,
        "is_mod": None, "is_employee": None, "link_karma": None, "comment_karma": None
    }

    # Suspended/banned accounts will only return the name and is_suspended attributes !
    user_info_available = True
    try:
        # the is_suspended attribute seems to only exist if the user is actually suspended and even then not always ...
        if hasattr(redditor, 'is_suspended') and redditor.is_suspended:
            # print("User is suspended!")
            user_info_available = False
    except prawcore.exceptions.NotFound:
        print("Warning: User not found!")
        user_info_available = False

    user_info_dict["username"] = redditor.name
    user_info_dict["is_suspended"] = False if user_info_available else True

    if user_info_available:
        user_info_dict["id"] = redditor.id
        user_info_dict["created_at"] = redditor.created_utc
        creation_date_formatted = datetime.fromtimestamp(redditor.created_utc).strftime('%Y-%m-%d')
        user_info_dict["created_at_formatted"] = creation_date_formatted
        user_info_dict["is_mod"] = redditor.is_mod
        user_info_dict["is_employee"] = redditor.is_employee
        user_info_dict["link_karma"] = redditor.link_karma
        user_info_dict["comment_karma"] = redditor.comment_karma
        # print(len(list(redditor.submissions.top(time_filter="all"))))
        # print(len(list(redditor.comments.top(time_filter="all"))))
    else:
        print("User is suspended, banned or doesn't exist anymore, so no information available :(\n")

    return user_info_dict


def get_subreddit_information(praw_instance):
    subreddit = praw_instance.subreddit("redditdev")
    print(subreddit.display_name)
    print(subreddit.title)
    print(subreddit.description)


def search_for_subreddits(query: str):
    """
    Searches for subreddits beginning with the query string.
    """
    relevant_subreddits = list()
    for subreddit in praw.models.Subreddits.search_by_name(query=query, exact=False):
        relevant_subreddits.append((subreddit.title, subreddit.description))
    return relevant_subreddits


def save_reddit_data(data: pd.DataFrame, out_path=OUT_DATA / "reddit_data.csv"):
    # sort dataframe by date
    data.sort_values(by="created_at", ascending=False, inplace=True)
    # save reddit data as a csv file
    data.to_csv(out_path, index=False, encoding="utf-8")  # sep=";"


def convert_time_expression_to_date(date_text: str):
    """
    Convert an expression in the form of "n days ago" to an actual datetime object by subtracting it from the current
    date.
    Expression times in reddit comments can be of the format: Min., Std., Tag/Tagen, Monat/Monaten, Jahr/Jahren (in
    German reddit that is ...)
    """
    # TODO also extract Min., Std., Tag/Tagen, Monat/Monaten, Jahr/Jahren first and convert accordingly
    num_days_ago = int(re.findall(r'\d+', date_text)[0])  # extract number
    current_date = datetime.now(timezone.utc)
    comment_date = current_date - timedelta(days=num_days_ago)
    return comment_date


def save_website_locally(html, local_path):
    """
    Save the html response locally, so we don't have to make a new request to the server everytime and therefore
    scrape more responsibly.
    """
    with open(local_path, 'wb') as f:
        f.write(html)


def load_local_website_cache(local_path):
    with open(local_path, 'rb') as f:
        return f.read()
