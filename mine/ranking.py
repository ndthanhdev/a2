from babi_rnn_vi import tokenize
import datetime


def score_base(query, story):
    '''
        Score a story
        # Arguments
        query: list of token
        story: str story
    '''
    total = 0
    tokenized_story = tokenize(story)
    for token in query:
        total += tokenized_story.count(token)
    return total


def score_time(query, story):
    now = datetime.datetime.now()
    tokenized_story = tokenize(story)
    return tokenized_story.count(str(now.hour))

def score(query, story, answer_type=None):
    '''
        Score a story

        # Arguments
        query: list of token
        story: str story
        answer_type: int 
    '''
    if answer_type == 1:
        return score_time(query, story)
    else:
        return score_base(query, story)


def ranking_stories(query, stories, answer_type=None):
    '''
    Return ranked stories
    # Arguments
    query: list of token
    answer_type: int
    stories: stories
    '''

    return sorted(stories, key=lambda story: score(query, story, answer_type), reverse=True)
