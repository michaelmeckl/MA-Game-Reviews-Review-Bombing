import time
import re
from datetime import datetime
from typing import TYPE_CHECKING
from Reddit.reddit_utils import get_user_information
if TYPE_CHECKING:
    from redditwarp.types import JSON
from redditwarp.dark.SYNC import Client as DarkClient
from redditwarp.dark.core.const import GRAPHQL_BASE_URL
from redditwarp.http.util.json_loading import load_json_from_response


def extract_comments_for_submission(submission, only_top_level=False):
    print(f"Number of Comments: {submission.num_comments}\n")
    extracted_comments = list()
    already_done = set()

    # submission.comment_limit = 15  # set comment limit
    submission.comment_sort = "new"  # set comment sort order  # apparently not working correctly on reddit's side :(

    if only_top_level:
        # #####################################
        #       Only top-level comments
        # #####################################

        # ignore the "more comments" objects by setting limit to 0; if they should be returned as well, use
        # limit=None, see https://praw.readthedocs.io/en/stable/tutorials/comments.html#the-replace-more-method
        submission.comments.replace_more(limit=0)
        comment_list = submission.comments
    else:
        # #####################################
        #   All comments (including replies)
        # #####################################
        submission.comments.replace_more(limit=None)
        comment_list = submission.comments.list()

    for i, comment in enumerate(comment_list):
        if comment.id in already_done:
            continue

        try:
            # print("Comment {0} by {1}\n".format(i, comment.author))
            extract_relevant_comment_content(comment, already_done, extracted_comments)
        except Exception as e:
            print(f"ERROR when trying to extract comment: {e}")
            print("Waiting for a few seconds ...")
            time.sleep(10)

        already_done.add(comment.id)

    return extracted_comments


def extract_relevant_comment_content(comment, existing_comments: set, out_data: list):
    comment_content = re.sub(r"\s+", " ", comment.body)
    post_time = datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S')
    post_time_with_timezone = datetime.strptime(post_time + "+0000", '%Y-%m-%d %H:%M:%S' + '%z')

    # parent = comment.parent()   # either the submission or another comment
    # print(parent.id)
    # submission_instance = comment.submission
    # subreddit = comment.subreddit
    # subreddit_name = subreddit.name

    # get the user information of the comment author
    user_info = get_user_information(comment.author)

    existing_comments.add(comment.name)
    out_data.append({
        "id": f"t1_{comment.id}", "content": comment_content, "subreddit": None,
        "upvote_score": comment.score, "created_at": post_time_with_timezone,
        "original_post_date": None, "comment_url": comment.permalink, "original_post_url": None,
        "original_post_author": None, "original_post_author_id": None,
        # "parent_id": comment.parent_id, "num_replies": len(comment.replies),
        "author_name": user_info["username"], "author_id": user_info["id"],
        "author_created_at": user_info["created_at"], "author_created_at_formatted": user_info["created_at_formatted"],
        "author_is_suspended": user_info["is_suspended"], "author_is_mod": user_info["is_mod"],
        "author_is_employee": user_info["is_employee"], "author_link_karma": user_info["link_karma"],
        "author_comment_karma": user_info["comment_karma"],
    })


def search_comments_with_redditwarp(query: str, subreddit: str = None, since=None, until=None, reddit_instance=None):
    relevant_comments = list()
    extracted_comments = set()

    start_date = datetime.strptime(since, '%d-%m-%Y').timestamp() if since is not None else None
    end_date = datetime.strptime(until, '%d-%m-%Y').timestamp() if until is not None else None
    sort = 'NEW'

    json_request_data: JSON = {
        "id": "8e8ea0cefd5f",
        "variables": {
            "query": query,
            "sort": sort,
            "filters": [
                {"key": "nsfw", "value": "1"},
                # {"key": "subreddit_names", "value": subreddit}
            ],
            "productSurface": "web2x",
            "includePosts": False,
            "includeCommunities": False,
            "includeAuthors": False,
            "includeComments": True,
            "postsAfter": None,
            "communitiesAfter": None,
            "authorsAfter": None,
            "commentsAfter": None,
            # "searchInput": {"queryId": "81761512-38b4-4885-8a70-17421a65cadf", "structureType": "search"},
            "communitySearch": True,
            # "subredditNames": [subreddit] ,
            "customFeedSearch": False
        },
    }

    if subreddit is not None:
        # if a subreddit was specified update the json request
        json_request_data["variables"]["filters"].append({"key": "subreddit_names", "value": subreddit})
        json_request_data["variables"]["subredditNames"] = [subreddit]

    dark_client = DarkClient()
    more_comments = True

    while more_comments:
        resp = dark_client.http.request('POST', GRAPHQL_BASE_URL, json=json_request_data)
        resp.ensure_successful_status()
        # wait for a few seconds after each request
        time.sleep(5)

        out_json_data = load_json_from_response(resp)

        try:
            page_info = out_json_data['data']['search']['general']['comments']['pageInfo']
            end_comment = page_info['endCursor']
            has_next_page = page_info['hasNextPage']
            print(f"HasNextPage: {has_next_page}")

            # update break condition
            more_comments = has_next_page
            # update the commentsAfter field in the json to request the next set of comments
            json_request_data["variables"]["commentsAfter"] = end_comment

            for edge in out_json_data['data']['search']['general']['comments']['edges']:
                try:
                    node = edge['node']

                    comment_id = node['id']
                    if comment_id in extracted_comments:
                        continue

                    comment_creation_date = node['createdAt']
                    # check time period constraints if any
                    comment_creation_timestamp = datetime.strptime(comment_creation_date,
                                                                   "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
                    if (start_date is not None) and (comment_creation_timestamp < start_date):
                        print(f"breaking because comment date ({comment_creation_timestamp}) is older than start date")
                        # continue
                        break  # break here because the comments should be sorted by date so only older ones will follow
                    if (end_date is not None) and (comment_creation_timestamp > end_date):
                        print(f"skipping because comment date ({comment_creation_timestamp}) is newer than end date")
                        continue

                    comment_text = node['content']['markdown']
                    comment_author = node['authorInfo']['name'] if "name" in node['authorInfo'] else "deleted"
                    comment_author_id = node['authorInfo']['id']
                    comment_upvote_score = node['score']
                    comment_url = node['permalink']
                    original_post_author = node['postInfo']['authorInfo']['name'] if "name" in node['postInfo'][
                        'authorInfo'] else "deleted"
                    original_post_author_id = node['postInfo']['authorInfo']['id']
                    original_post_date = node['postInfo']['createdAt']
                    original_post_subreddit = node['postInfo']['subreddit']['name']
                    original_post_url = node['postInfo']['permalink']

                    """print(f'''\
                        {comment_id}
                        {comment_creation_date}
                        {comment_url}
                        {comment_upvote_score}
                        {comment_text}
                        \nAuthor:
                        {comment_author_id}, {comment_author}
                        \nAssociated Post:
                        {original_post_author_id}, {original_post_author}
                        {original_post_date}
                        {original_post_url}
                        {original_post_subreddit}\
                        ''')"""

                    extracted_comments.add(comment_id)
                    comment_info = {
                        "id": comment_id, "content": comment_text, "subreddit": original_post_subreddit,
                        "upvote_score": comment_upvote_score, "created_at": comment_creation_date,
                        "original_post_date": original_post_date, "comment_url": comment_url,
                        "original_post_url": original_post_url, "original_post_author": original_post_author,
                        "original_post_author_id": original_post_author_id, "author_name": comment_author,
                        "author_id": comment_author_id, "author_created_at": None, "author_created_at_formatted": None,
                        "author_is_suspended": None, "author_is_mod": None, "author_is_employee": None,
                        "author_link_karma": None, "author_comment_karma": None
                    }

                    # try to get additional information about the author of the comment
                    if comment_author != "deleted" and reddit_instance is not None:
                        redditor = reddit_instance.redditor(comment_author)
                        user_info = get_user_information(redditor)

                        comment_info.update({
                            "author_created_at": user_info["created_at"],
                            "author_created_at_formatted": user_info["created_at_formatted"],
                            "author_is_suspended": user_info["is_suspended"], "author_is_mod": user_info["is_mod"],
                            "author_is_employee": user_info["is_employee"],
                            "author_link_karma": user_info["link_karma"],
                            "author_comment_karma": user_info["comment_karma"],
                        })

                    relevant_comments.append(comment_info)

                except Exception as e:
                    print(f"ERROR when parsing comment \"{edge}\": {e}")

        except Exception as e:
            print(f"ERROR when trying to extract comments from page: {e}")

    print("---------------------------\nFinished scraping reddit comments!")
    return relevant_comments
