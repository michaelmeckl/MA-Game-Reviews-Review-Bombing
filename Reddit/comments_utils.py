import os.path
import pathlib
import pprint
import time
from requests_html import HTMLSession
import re
from datetime import datetime
from bs4 import BeautifulSoup
from tqdm import tqdm
from typing import TYPE_CHECKING
from Reddit.reddit_utils import load_local_website_cache, save_website_locally, convert_time_expression_to_date, \
    REDDIT_BASE_URL, get_user_information

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
    submission.comment_sort = "new"  # set comment sort order  # apparently not working correctly on reddit side :(

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


# This method does not really work, mainly because a lot of subreddits don't have comments under the .comments
#   endpoint (so this method is only useful for r/all or to watch comments continuously)
# Unfortunately, searching for comments with the official api does not work either as the type="comment" query parameter
#   is not yet supported by the Reddit API.
"""
def search_comments(praw_instance, query: str, subreddits: list[str] = None, limit=None, since=None, until=None):
    if subreddits is None:
        subreddits = ["all"]
    subreddit_names = "+".join(subreddits)

    # TODO use this instead of query ?
    review_bomb_keywords = ["ReviewBomb", "review bombing", "review-bomb", "review bomb"]
    game_keywords = ["cyberpunk 2077"]

    start_date = datetime.strptime(since, '%d-%m-%Y').timestamp() if since is not None else None
    end_date = datetime.strptime(until, '%d-%m-%Y').timestamp() if until is not None else None

    relevant_comments = list()
    extracted_comments = set()
    # TODO keep track of already existing comments to make sure they aren't processed twice (i.e. a simple text file
    #  with the ids would probably be enough)

    # To retrieve all new comments made to this subreddit: (per default the 100 latest are returned)
    # To only retrieve new submissions starting when the stream is created, pass (skip_existing=True)
    try:
        count = 0
        # for comment in reddit.subreddit(subreddit_names).comments(limit=limit):
        for comment in praw_instance.subreddit(subreddit_names).stream.comments(skip_existing=False):
            if limit is not None and count == limit:
                break
            count += 1
            # print(f"Count {count}, Comment: {comment.permalink}")

            if comment.id in extracted_comments:
                continue
            if (start_date is not None) and (comment.created_utc < start_date):
                continue
            if (end_date is not None) and (comment.created_utc > end_date):
                continue

            # TODO check if this comment is relevant for query / if any keywords match
            if any(keyword in comment.body for keyword in review_bomb_keywords) and any(keyword in comment.body for
                                                                                        keyword in game_keywords):
                print(f"Found matching comment: {comment.permalink}")
                extract_relevant_comment_content(comment, extracted_comments, relevant_comments)

    except (praw.exceptions.PRAWException, prawcore.exceptions.PrawcoreException) as e:
        print(f'praw/stream related exception: {e}')
    except Exception as e:
        print(f'other exception: {e}')

    return relevant_comments
"""


# not used atm since scraping reddit comments manually is quite challenging (especially as a lot of the content is
# dynamically generated by javascript and therefore requires more complex scraping solutions)
def scrape_reddit_comments(query: str, subreddits: list[str] = None, limit=None, since=None, until=None):
    """
    Scrapes the comments search reddit page as this feature is not implemented in their API yet
    """
    if subreddits is None:
        subreddits = ["all"]
    subreddit_names = "+".join(subreddits)

    start_date = datetime.strptime(since, '%d-%m-%Y').timestamp() if since is not None else None
    end_date = datetime.strptime(until, '%d-%m-%Y').timestamp() if until is not None else None

    relevant_comments = list()
    extracted_comments = set()

    Local_Website_Cache = pathlib.Path(__file__).parent / "reddit_data" / f"reddit_comments_{subreddit_names}"
    # if we have the website save locally use that one instead of requesting it again
    if os.path.exists(Local_Website_Cache):
        print("Local website cache found, loading ...")
        rendered_html = load_local_website_cache(Local_Website_Cache)
        session = None
    else:
        print("No local website cache found, requesting website ...")

        url = f"https://reddit.com/r/{subreddit_names}/search"
        querystring = {"q": f"{query}", "type": "comment", "sort": "new", "restrict_sr": 1, "t": "all"}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/113.0.0.0 Safari/537.36",
        }
        # we use the request-html package to render and extract the javascript generated content on the page;
        # more sophisticated options would be tools like Scrapy or Playwright
        session = HTMLSession()
        r = session.get(url, headers=headers, params=querystring)

        # todo comments are only loaded in batches, how to load more dynamically ??
        #  -> per default only 16 elements
        #  -> scrolldown 20 => 52 comments for the current query
        # => use Playwright or Scrapy instead ?
        r.html.render(scrolldown=5, timeout=10, sleep=5)  # wait for a few seconds to execute the javascript
        rendered_html = r.html.html

        save_website_locally(r.content, Local_Website_Cache)  # cache locally

    # use the rendered html as input for beautifulsoup
    soup = BeautifulSoup(rendered_html, 'lxml')

    """
    rendered_response = r.html
    comment_list = [comment for comment in rendered_response.find('div[data-testid=post-container]')]

    # get attribute with .attrs.get('id')
    for comment in comment_list:
        # get information about the original post this comment was made in as well as the comment author
        # as this information is dynamically rendered, the requests_html package is required for these
        comment_author_el = comment.find('a[data-testid=comment_author_link]', first=True)
        comment_author = comment_author_el.text if comment_author_el is not None else "n.a."
        print(f"\ncomment_author: {comment_author}")

        original_submission_author_el = comment.find('a[data-testid="post_author_link"]', first=True)
        original_submission_author = original_submission_author_el.text if original_submission_author_el is not None 
            else "n.a."
        print(f"\noriginal_submission_author: {original_submission_author}")

        original_submission_date_el = comment.find('span[data-testid=post_timestamp]', first=True)
        original_submission_date = original_submission_date_el.text if original_submission_date_el is not None else 
            "n.a."
        print(f"\noriginal_submission_date: {original_submission_date}")

        original_submission_subreddit_el = comment.find('a[data-testid=subreddit-name]', first=True)
        original_submission_subreddit = original_submission_subreddit_el.text if original_submission_subreddit_el is 
            not None else "n.a."
        print(f"\noriginal_submission_subreddit: {original_submission_subreddit}")

    num_comments = len(comment_list)
    print("Num Comments requests-html: ", num_comments)
    """

    # for the comment list, search for a div with the tag: data-testid="comments-list"
    comment_list_container = soup.find("div", attrs={'data-testid': "comments-list"})
    first_child = next(comment_list_container.children, None)

    if first_child is not None:
        comment_list = first_child.contents
        num_comments = len(comment_list)
        print("Num Comments: ", num_comments)

        for comment_container in tqdm(comment_list):
            try:
                # get information about the original post this comment was made in
                original_submission_subreddit_el = comment_container.find("a", attrs={'data-testid': "subreddit-name"})
                original_submission_subreddit = original_submission_subreddit_el.text.strip() \
                    if original_submission_subreddit_el is not None else "n.a."
                print(f"\noriginal_submission_subreddit: {original_submission_subreddit}")

                original_submission_author_el = comment_container.find("a", attrs={'data-testid': "post_author_link"})
                original_submission_author = original_submission_author_el.text.strip() \
                    if original_submission_author_el is not None else "n.a."
                print(f"original_submission_author: {original_submission_author}")

                original_submission_date_el = comment_container.find("span", attrs={'data-testid': "post_timestamp"})
                original_submission_date = original_submission_date_el.text.strip() \
                    if original_submission_date_el is not None else "n.a."
                print(f"original_submission_date: {original_submission_date}")

                original_submission_url = comment_container.find("a", attrs={'data-testid': "go_to_thread_link"})[
                    "href"]
                print(f"original_submission_url: {REDDIT_BASE_URL + original_submission_url}")

                # extract the comment id from the comment container's class tag
                comment_id_tag_list = [el["class"] for el in comment_container.select('div[class*="Comment t1_"]')]
                comment_id = comment_id_tag_list[0][1]  # the comment id is the second element in the class tag
                print(f"Comment_Id: {comment_id}")

                # get author, date and number of upvotes of this comment
                comment_header = comment_container.find("div", attrs={'data-testid': "post-comment-header"})

                comment_author_el = comment_header.find("a", attrs={'data-testid': "comment_author_link"})
                # the comment author could be [deleted] so we check if it is actually there
                comment_author = comment_author_el.text.strip() if comment_author_el is not None else "n.a."
                print(f"comment_author: {comment_author}")

                # TODO the text content of the comment creation day is somewhat ruined, most likely through the html
                #  rendering of the requests-html package (=> now there are some errors such as month written as min...)
                comment_creation_date_el = comment_header.find("a", attrs={'data-testid': "comment_timestamp"})
                comment_creation_date = comment_creation_date_el.text if comment_creation_date_el is not None else "n.a"
                print(f"comment_creation_date: {comment_creation_date}")
                comment_date = convert_time_expression_to_date(comment_creation_date)
                print(f"Correct comment date: {comment_date}")

                comment_upvote_text = comment_container.find("span", class_="_vaFo96phV6L5Hltvwcox").text.strip()
                comment_upvote_score = re.findall(r'-?\d+', comment_upvote_text)[0]
                print(f"Comment upvote score: {comment_upvote_score}")

                # extract the actual comment text
                comment_body = comment_container.find("div", attrs={'data-testid': "comment"})
                comment_text_list = [text_section.text.strip() for text_section in comment_body.findAll("p")]
                comment_text = "\n".join(comment_text_list)
                print(f"Comment Text:\n{comment_text}")

                extracted_comments.add(comment_id)
                relevant_comments.append({
                    "id": comment_id, "author": comment_author, "content": comment_text, "created_at":
                        comment_creation_date, "created_at_formatted": comment_date, "upvote_score":
                        comment_upvote_score, "parent": original_submission_url, "subreddit":
                        original_submission_subreddit, "original_submission_author": original_submission_author,
                    "original_submission_date": original_submission_date,
                })

            except Exception as e:
                print(f"ERROR: {e}")
    else:
        raise Exception("Could not find first child of comment list !")

    if session is not None:
        session.close()  # close the html-request session

    return relevant_comments


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
