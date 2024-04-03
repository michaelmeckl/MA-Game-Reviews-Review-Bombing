import os.path
import pathlib
from requests_html import HTMLSession
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
from Reddit.reddit_utils import REDDIT_BASE_URL


# old method without praw wrapper:
"""def make_reddit_api_call(query: str, request_limit: int = 2):
    subreddit = "cyberpunkgame"
    url = f"https://reddit.com/r/{subreddit}/search.json"

    querystring = {"q": f"{query}", "limit": f"{request_limit}"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/113.0.0.0 Safari/537.36",
    }

    response = requests.request("GET", url, headers=headers, params=querystring)
    result = response.json()
    print(f"Result type: {result['kind']}")   # t1_	Comment, t3_ Link
    result_list = result["data"]["children"]
    # pprint.pprint(result_list)

    for result_item in result_list:
        print(f"\nResult item type: {result_item['kind']}")
        item = result_item["data"]
        print("NewItem:")
        pprint.pprint(f"Title: {item['title']}, Text: {item['selftext']},\n Author: {item['author']}, Subreddit:"
                      f" {item['subreddit']}, (Net)Score: {item['score']}, Upvote Ratio: {item['upvote_ratio']}, "
                      f"Upvotes: {item['ups']}, Downvotes: {item['downs']}, num_comments: {item['num_comments']}, "
                      f"created at: {item['created']}, permalink: {REDDIT_BASE_URL + item['permalink']}")

        article_id = item["id"]
        # print(article_id)

        comments_url = f"https://reddit.com/comments/{article_id}.json"
        querystring = {"limit": "3"}  # only get the first 3 comments for testing
        comments_response = requests.request("GET", comments_url, headers=headers, params=querystring)
        print(f"Comments:\n {comments_response.text}")

        comments_result = comments_response.json()
        print(len(comments_result))

        # ? is the first element always the original post ?
        comments_list = comments_result[1]["data"]["children"]
        # pprint.pprint(comments_list)

        # the last comment always seems to be a miscellaneous listing with ids for further comments ?
        # see https://github.com/reddit-archive/reddit/wiki/JSON#more
        for i, comment_item in enumerate(comments_list):
            comment = comment_item["data"]
            print(f"\nComment {i}:")
            print(f"Author: {comment['author'] if 'author' in comment else 'no author'}, Body:"
                  f" {comment['body'] if 'body' in comment else 'no body'}")
            # print(f"Parent ID: {comment['parent_id'] if 'parent_id' in comment else 'no parent'}")

            replies_data = comment['replies'] if 'replies' in comment else []
            if len(replies_data) > 0:
                replies = replies_data['data']['children']
                print(f"Replies List: {len(replies)}")
                for reply in replies:
                    print(reply)
            else:
                print("No replies for this comment.")
"""


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


def convert_time_expression_to_date(date_text: str):
    """
    Convert an expression in the form of "n days ago" to an actual datetime object by subtracting it from the current
    date.
    Expression times in reddit comments can be of the format: Min., Std., Tag/Tagen, Monat/Monaten, Jahr/Jahren (in
    German reddit that is ...)
    """
    # also extract Min., Std., Tag/Tagen, Monat/Monaten, Jahr/Jahren first and convert accordingly
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
