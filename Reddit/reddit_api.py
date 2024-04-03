#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import os
import pathlib
import re
import time
from datetime import datetime
import pandas as pd
import praw
import prawcore
from Reddit.comments_utils import search_comments_with_redditwarp, extract_comments_for_submission
from Reddit.reddit_utils import save_reddit_data, OUT_DATA, get_user_information
# from pmaw import PushshiftAPI


# If you are only analyzing public comments, entering a username and password is optional.
# init PRAW instance with .ini file
reddit = praw.Reddit("Search_Reddit", config_interpolation="basic")
print(reddit.read_only)


def enable_praw_logging():
    # enable http logging, see https://praw.readthedocs.io/en/stable/getting_started/logging.html
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    for logger_name in ("praw", "prawcore"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)


def get_specific_submission(identifier: str, choice: str = "u"):
    """
    :param choice: either "u" to pass an url or "i" to pass an id
    :param identifier: the actual url or id for the submission
    :return: a reddit submission object
    """
    if choice == "u":
        return reddit.submission(url=identifier)
    elif choice == "i":
        return reddit.submission(id=identifier)
    else:
        print("[ERROR] Either 'u' for an URL or 'i' for an ID must be entered for the [choice] parameter!")
        return


def extract_submissions_from(subreddit_names: list):
    # combine subreddits with "+"
    subreddit_name = "+".join(subreddit_names)
    print("Subreddit Name: ", subreddit_name)

    all_submissions = list()
    existing_submissions = set()  # use a set to filter out duplicates

    # If you want to retrieve as many as possible pass in limit=None
    for submission in reddit.subreddit(subreddit_name).hot(limit=5):
        # to see all available fields, use pprint.pprint(vars(submission))
        """ Useful:
            author, title, name, (author_is_blocked, banned_by, num_reports, removal_reason), created, created_utc, 
            downs, ups, upvote_ratio, id, score, selftext, subreddit, subreddit_id, subreddit_type, (likes),
            num_comments, permalink, url
        """
        # only extract submissions we haven't had yet
        if submission.name in existing_submissions:
            continue
        # exclude bot posts/advertised posts
        if submission.stickied:
            continue

        print(f"################## Extracting submission {submission.name} (id: {submission.id}):")
        print(f"Title {submission.title} from author {submission.author}")
        print("Url: ", submission.url)
        extract_relevant_submission_content(submission, existing_submissions, all_submissions)
        # save comments for this submission in an own file with the id of the submission added for reference ?
        # extract_comments_for_submission(submission)

        print(f"Finished with submission {submission.name} (id: {submission.id})\n##################\n")

    return all_submissions


def extract_relevant_submission_content(submission, existing_submissions: set, out_data: list):
    existing_submissions.add(submission.name)

    submission_content = re.sub(r"\s+", " ", submission.selftext)
    post_time = datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
    post_time_with_timezone = datetime.strptime(post_time + "+0000", '%Y-%m-%d %H:%M:%S' + '%z')

    # get the user information of the submission author
    user_info = get_user_information(submission.author)

    out_data.append({
        "id": submission.id, "title": submission.title, "content": submission_content, "score": submission.score,
        "subreddit": submission.subreddit, "created_at": submission.created_utc,
        "created_at_formatted": post_time_with_timezone, "num_comments": submission.num_comments,
        "url": submission.permalink,
        "author_id": user_info["id"], "author_name": user_info["username"],
        "author_created_at": user_info["created_at"], "author_created_at_formatted": user_info["created_at_formatted"],
        "author_is_suspended": user_info["is_suspended"], "author_is_mod": user_info["is_mod"],
        "author_is_employee": user_info["is_employee"], "author_link_karma": user_info["link_karma"],
        "author_comment_karma": user_info["comment_karma"],
    })


def search_submissions(query: str, subreddits: list[str] = None, limit=None, since=None, until=None):
    """
    Since and until must be in the form "DD-MM-YYYY"; as they are converted to timestamps they are always converted
    to 00:00:00 time, which basically means the start date is inclusive and the until date is exclusive.

    Search tips (see https://www.reddit.com/wiki/search/)
        * search in specific fields like so: selftext:cats or title:cats
        * self:True filters posts made by an individual account, rather than posts linking to another site
        * multi-word search with "", e.g. selftext:"cat owner"
        * typical boolean search operators like AND OR NOT as well as (...) work
        * AND query is the default, e.g. "title:cats selftext:cats" == "title:cats AND selftext:cats"
        * searching multiple subreddits with "+" works here as well
        * wildcards (*) seem to work as well if used in "..." (?)
        * query seems to be case-insensitive by default
    """
    if subreddits is None:
        subreddit_name = "all"
    else:
        subreddit_name = "+".join(subreddits)

    # searching by time period is not possible with reddit api or praw and needs to be implemented manually
    """
    current_time = datetime.now(timezone.utc).timestamp()
    two_weeks_ago = 14 * 24 * 3600
    since = current_time - two_weeks_ago
    """
    start_date = datetime.strptime(since, '%d-%m-%Y').timestamp() if since is not None else None
    end_date = datetime.strptime(until, '%d-%m-%Y').timestamp() if until is not None else None

    all_submissions = list()
    existing_submissions = set()

    print("\nSearching reddit ...")
    for submission in reddit.subreddit(subreddit_name).search(query, limit=limit, sort="relevance", time_filter="all"):
        try:
            # exclude duplicates
            if submission.name in existing_submissions:
                continue
            # exclude bot posts/advertised posts
            if submission.stickied:
                continue
            # check time period constraints if any
            if (start_date is not None) and (submission.created_utc < start_date):
                continue
            if (end_date is not None) and (submission.created_utc > end_date):
                continue

            extract_relevant_submission_content(submission, existing_submissions, all_submissions)
            # extract_comments_for_submission(submission)  # TODO also extract the comments for each submission ?

        except prawcore.exceptions.Forbidden as exc:
            print("Forbidden error!")
            print(exc.request.url)
            # raise

    return all_submissions


def get_submissions_for_game(game, subreddits, query_subreddit, query_all, is_specific_query, start_date=None,
                             end_date=None):
    print(f"\nSearching submissions from reddit for game \"{game}\" ...")

    extracted_submissions = search_submissions(query=query_subreddit, since=start_date, until=end_date,
                                               subreddits=subreddits)
    df = pd.DataFrame(extracted_submissions)
    num_extracted = len(extracted_submissions)
    print(f"Extracted {num_extracted} submissions from subreddits")
    if num_extracted > 0:
        save_reddit_data(df, out_path=OUT_DATA / f"submissions_subreddit_{game}.csv")

    # search for the same query in /all to get other results too
    extracted_submissions_all = search_submissions(query=query_all, since=start_date, until=end_date,
                                                   subreddits=None)
    df_all = pd.DataFrame(extracted_submissions_all)
    num_extracted_all = len(extracted_submissions_all)
    print(f"Extracted {num_extracted_all} submissions from r/all")
    if num_extracted_all > 0:
        save_reddit_data(df_all, out_path=OUT_DATA / f"submissions_all_{game}.csv")

    if num_extracted > 0 and num_extracted_all > 0:
        # combine both dataframes and remove duplicate rows
        merged_df = pd.concat([df, df_all], join="outer").drop_duplicates(subset=['id'], keep='first').reset_index(
            drop=True)
        # merged_df = df.merge(df_all, how="outer", on="id")
        save_reddit_data(merged_df,
                         OUT_DATA / f"merged_reddit_submissions_{'' if is_specific_query else 'general_'}{game}.csv")


def get_comments_for_game(game, subreddits, query, is_specific_query, start_date=None, end_date=None):
    print(f"\nSearching comments from reddit for game \"{game}\" ...")

    dataframes = []
    for subreddit in subreddits:
        extracted_comments = search_comments_with_redditwarp(query=query, subreddit=subreddit, since=start_date,
                                                             until=end_date, reddit_instance=reddit)
        print(f"\nExtracted {len(extracted_comments)} comments from subreddit \"{subreddit}\"")
        comments_df = pd.DataFrame(extracted_comments)
        print(comments_df.head())
        print("#####################\n")
        # comments_df.to_csv(OUT_DATA / f"reddit_comments_{game}_{subreddit}.csv", index=False)
        dataframes.append(comments_df)

    df_all = pd.concat(dataframes, join="outer").drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)
    df_all.to_csv(OUT_DATA / f"merged_reddit_comments_{'' if is_specific_query else 'general_'}{game}.csv", index=False)
    # TODO search r/all for comments as well ?


def get_reddit_data_for_games():
    # enable_praw_logging()
    if not OUT_DATA.is_dir():
        OUT_DATA.mkdir()

    """
    # the query to use for searching submissions in the subreddits
    query_subreddit = "\"ReviewBomb*\" OR \"review-bomb*\" OR \"review bomb*\""
    # the query to use for searching submissions in the r/all subreddit
    query_all = '("ReviewBomb*" OR "review-bomb*" OR "review bomb*") AND "Hogwarts Legacy"'
    """
    #################################################################
    default_query = 'ReviewBomb OR boycott OR controversy OR fake OR sabotage OR manipulate OR spam OR hate'
    default_query_comments = 'ReviewBomb OR "review bombing"'
    # default_query_comments = 'review AND (good OR bad OR negative OR positive OR hate)'   # alternative

    use_game_specific_query = True  # whether to use the default queries above or the game specific ones
    #################################################################

    def get_all_query(game_name: str):
        return f'{game_name} AND ({default_query})'

    games = {
        "ukraine_russia_review_bombing": {
            "name": '(ukraine OR russia) AND (ReviewBomb OR "review bombing" OR game review)',
            "subreddits": ["cyberpunkgame", "CyberpunkTheGame", "LowSodiumCyberpunk", "witcher", "thewitcher3",
                           "Witcher3", "CDProjektRed", "gwent", "thronebreaker", "stalker", "Frostpunk", "ThisWarofMine"
                           ],
            "start_date": "24-02-2022", "end_date": "01-05-2022",
            "query_subreddit": '(review OR support OR sales) AND (ukraine OR russia)',
            "query_comments": '(ReviewBomb OR "review bombing" OR review OR support OR sales) AND (ukraine OR russia)',
            # query_all is usually: name AND query_subreddit
            "query_all": '(ukraine OR russia) AND (ReviewBomb OR "review bombing" OR game review) AND ((review OR '
                         'support OR sales) AND (ukraine OR russia))',
        },
        "Cyberpunk 2077": {
            "name": '"Cyberpunk 2077"',
            "subreddits": ["cyberpunkgame", "CyberpunkTheGame", "LowSodiumCyberpunk"],
            "start_date": "09-12-2020", "end_date": "01-02-2021",
            "query_subreddit": "lie OR fraud OR scam OR broken OR disappoint OR disappointment OR disappointing",
            "query_comments": "lie OR fraud OR scam",
            "query_all": '"Cyberpunk 2077" AND (lie OR fraud OR scam OR broken OR disappoint OR disappointment OR '
                         'disappointing)',
            # "start_date": "01-01-2023", "end_date": "01-02-2023",
            # "query_subreddit": 'award OR "Steam Awards" OR "Labor of Love"',
            # "query_comments": 'steam OR award OR "Labor of Love"',
            # "query_all": '"Cyberpunk 2077" AND (award OR "Steam Awards" OR "Labor of Love")',
        },
        "Borderlands 3": {
            "name": "borderlands 3",
            "subreddits": ["borderlands3", "Borderlands2", "Borderlands"],
            "start_date": "01-04-2019", "end_date": "01-05-2020",
            "query_subreddit": '"epic store" OR "epic games" OR exclusive',
            "query_comments": "epic games store exclusive",
            "query_all": 'borderlands 3 AND ("epic store" OR "epic games" OR exclusive)',
        },
        "Metro Exodus": {
            "name": '(metro OR "metro exodus")',  # search for all Metro parts as others were review-bombed as well
            "subreddits": ["metro", "metroexodus", "metro_exodus"],
            "start_date": "20-01-2019", "end_date": "01-03-2020",
            "query_subreddit": '"epic store" OR "epic games" OR exclusive',
            "query_comments": "epic games store",   # better without 'exclusive' here
            "query_all": '(metro OR "metro exodus") AND ("epic store" OR "epic games" OR exclusive)',
        },
        "Firewatch": {
            "name": "Firewatch",
            "subreddits": ["Firewatch"],
            "start_date": "01-09-2017", "end_date": None,   # "01-11-2017"  # None otherwise no results in subreddit
            "query_subreddit": "DCMA OR takedown OR pewdiepie",
            "query_comments": "DCMA OR takedown OR pewdiepie",
            "query_all": 'Firewatch AND (DCMA OR takedown OR pewdiepie)',
        },
        "Overwatch 2": {
            "name": '"Overwatch 2"',
            "subreddits": ["Overwatch", "overwatch2"],
            "start_date": "01-10-2022", "end_date": None,
            "query_subreddit": "promise OR shutdown OR greed OR monetization OR microtranscation",
            "query_comments": "(negative OR hate) AND (promise OR shutdown OR greed OR monetization OR "
                              "microtranscation)",
            "query_all": '"Overwatch 2" AND (promise OR shutdown OR greed OR monetization OR microtranscation)',
        },
        "The Elder Scrolls V Skyrim": {
            "name": "Skyrim",
            "subreddits": ["ElderScrolls", "skyrim"],
            "start_date": "22-04-2015", "end_date": None,  # "23-05-2015",
            "query_subreddit": '"paid mod"',
            "query_comments": '"paid mod"',
            "query_all": 'Skyrim AND ("paid mod")',
        },
        "bethesda_creation_club": {
            "name": '(Skyrim OR "Fallout 4" OR "creation club")',
            "subreddits": ["BethesdaGameStudios", "BethesdaSoftworks", "ElderScrolls", "skyrim", "Fallout", "fo4"],
            "start_date": "27-08-2017", "end_date": None,  # "01-12-2017",
            "query_subreddit": '"paid mod" OR "creation club" OR bethesda',
            "query_comments": '"paid mod" OR "creation club" OR bethesda',
            "query_all": '(Skyrim OR "Fallout 4" OR "creation club") AND ("paid mod" OR "creation club" OR bethesda)',
        },
        "Grand Theft Auto V": {
            "name": '("Grand Theft Auto V" OR "Grand Theft Auto 5" OR GTA5 OR GTAV)',
            "subreddits": ["GTA", "GTAV", "GrandTheftAutoV"],
            "start_date": "13-06-2017", "end_date": None,  # "14-07-2017",
            "query_subreddit": 'openiv OR ban OR "cease-and-desist" OR mod OR "take-two" OR "take 2"',
            "query_comments": "openiv OR mod",
            "query_all": '("Grand Theft Auto V" OR "Grand Theft Auto 5" OR GTA5 OR GTAV) AND (openiv OR ban OR '
                         '"cease-and-desist" OR mod OR "take-two" OR "take 2")',
        },
        "Total War Rome II": {
            "name": '("Total War Rome" OR "Rome II" OR TW:RII)',
            "subreddits": ["totalwar", "RomeTotalWar"],
            "start_date": "21-09-2018", "end_date": None,  # "01-11-2018",
            "query_subreddit": 'accurate OR accuracy OR "female general" OR feminist OR agenda',
            "query_comments": "female (accuracy OR agenda)",
            "query_all": '("Total War Rome" OR "Rome II" OR TW:RII) AND (accurate OR accuracy OR "female general" OR '
                         'feminist OR agenda)',
        },
        "Mortal Kombat 11": {
            "name": '(Mortal Kombat 11 OR MK11)',
            "subreddits": ["MortalKombat", "mk11", "MortalKombat11"],
            "start_date": "21-04-2019", "end_date": None,  # "22-05-2019",
            "query_subreddit": 'microtransaction OR sjw OR propaganda OR woke OR ending',
            "query_comments": "sjw woke",
            "query_all": '(Mortal Kombat 11 OR MK11) AND (microtransaction OR sjw OR propaganda OR woke OR ending)',
        },
        "Assassins Creed Unity": {
            "name": '("assassins creed unity" OR "assassin\'s creed unity" OR "AC Unity" OR "AC:Unity")',
            "subreddits": ["assassinscreed", "acunity"],
            "start_date": "17-04-2019", "end_date": None,  # "20-05-2019",
            "query_subreddit": 'unity AND ("notre-dame" OR fire OR positive)',
            "query_comments": 'unity AND ("notre-dame" OR fire OR positive)',
            "query_all": '("assassins creed unity" OR "assassin\'s creed unity" OR "AC Unity" OR "AC:Unity") AND ('
                         '"notre-dame" OR fire OR positive)',
        },
    }

    for game in games:
        game_infos = games[game]
        name = game_infos["name"]
        subreddits = game_infos["subreddits"]

        if use_game_specific_query:
            query_subreddit = game_infos["query_subreddit"]
            query_comments = game_infos["query_comments"]
            query_all = game_infos["query_all"]
            start_date = game_infos["start_date"]
            end_date = game_infos["end_date"]
        else:
            query_subreddit = default_query
            query_comments = default_query_comments
            query_all = get_all_query(name)
            start_date = end_date = None   # for general query use no time period because results are only a few anyway

        print(f"Getting submissions for game \"{game}\" with all_query \"{query_all}\"...")
        get_submissions_for_game(game, subreddits, query_subreddit, query_all, use_game_specific_query, start_date,
                                 end_date)
        print(f"{'-' * 50}")
        get_comments_for_game(game, subreddits, query_comments, use_game_specific_query, start_date, end_date)
        print(f"\nFinished with game {game}\n######################################\n")
        time.sleep(2)


def get_comments_for_extracted_submissions():
    def load_comments(submission_row):
        submission_id = submission_row["id"]
        try:
            submission = get_specific_submission(submission_id, choice="i")
            comment_list = extract_comments_for_submission(submission, only_top_level=False)

            # Fill in the missing values from the associated submission; this could be done when extracting the
            # comment, but it is far more efficient (in terms of requests) to only do it once here for all comments
            submission_date = submission_row["created_at_formatted"]
            submission_subreddit = submission_row["subreddit"]
            submission_url = submission_row["url"]
            submission_author = submission_row["author_name"]
            submission_author_id = submission_row["author_id"]
            for comment_infos in comment_list:
                comment_infos.update({
                    "subreddit": submission_subreddit, "original_post_date": submission_date,
                    "original_post_url": submission_url, "original_post_author": submission_author,
                    "original_post_author_id": submission_author_id,
                })

            all_comments.extend(comment_list)
        except Exception as e:
            print(f"ERROR when trying to load comments for submission ({submission_id}): {e}")

    submissions_folder = pathlib.Path(__file__).parent.parent / "data_for_analysis" / "reddit"
    for submissions_file in os.listdir(submissions_folder):
        if "submissions" not in submissions_file:
            continue
        print("Loading comments for submission file: ", submissions_file)
        all_comments = list()

        submissions_df = pd.read_csv(submissions_folder / submissions_file)
        # submissions_df = submissions_df.head(2)    # take first n rows only for faster testing
        # submissions_df["id"].apply(lambda submission: load_comments(submission))
        submissions_df.apply(lambda row: load_comments(row), axis=1)
        comments_df = pd.DataFrame(all_comments)

        submissions_file_name = (submissions_folder / submissions_file).stem   # remove the file extension
        comments_df.to_csv(OUT_DATA / f"{submissions_file_name}_comments.csv", index=False)
        print("\n#########################\n"
              f"Finished with submissions file {submissions_file}"
              "\n#########################\n")


# ! nach neuem Pricing (seit 1. Juli 2023) nur noch 10 queries pro Minute ohne OAuth - Authentifikation ?
if __name__ == "__main__":
    # extracted_submissions = extract_submissions_from(["cyberpunkgame"])
    get_reddit_data_for_games()
    # get_comments_for_extracted_submissions()  # TODO
