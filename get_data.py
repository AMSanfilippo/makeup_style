#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import praw
import string
import os
import re

os.chdir('your/code/dir/')

# function to return subreddit posts associated with a given keyword
def get_keyword_data(reddit,sub,kw,attrib_list,word_set):
    
    keyword_df = pd.DataFrame()
    
    # get posts associated with the keyword 
    key_posts = sub.search(kw)
    
    # get post-level data for each post
    ct = 0
    for p in key_posts:
        post_data = get_post_data(reddit,p,attrib_list,word_set)
        post_df = pd.DataFrame(post_data,index=[ct])
        keyword_df = keyword_df.append(post_df)
        ct += 1
    
    # return df with each row being a single post
    return keyword_df
    
    
# function to get post-level data 
def get_post_data(reddit,p,attrib_list,word_set):
    
    # get post-level data, then send it off to get comment data
    post_vars = vars(p)
    post_dict = {}
        
    print('now parsing post ', post_vars['id'])
    
    # get the post-level attributes required
    for attrib in attrib_list:
        post_dict[attrib] = post_vars[attrib]
            
    # send the post off for comment-level processing
    post_id = post_vars['id']
    post_author = post_vars['author'] 
    comment_dict = get_comment_data(reddit,post_id,post_author,word_set)
        
    # return a dict with post id, author, upvotes, word dummies
    post_dict.update(comment_dict)
    
    return post_dict
 
# function to get comment-level data       
def get_comment_data(reddit,post_id,post_author,word_set):
    
    comment_dict = dict.fromkeys([i.replace(' ','_') for i in word_set])
    
    punctuation_table = str.maketrans(dict.fromkeys(string.punctuation))
    
    # get the post submission based on the post id
    submission = reddit.submission(id=post_id)
    
    # parse through the top-level comments and get those from the author
    author_comments = ''
    
    for tlc in submission.comments:
        try:
            if tlc.author == post_author:
                v = vars(tlc)
                body = v['body']
                cleaned_body = body.translate(punctuation_table).lower().replace('\n',' ')
                author_comments += cleaned_body
        except:
            break
    
    # get the set intersection of author words and words of interest 
    intersection = set(re.findall(r'(?:' + '|'.join(word_set) + ')',author_comments))
    comment_dict.update(zip([i.replace(' ','_') for i in list(intersection)],['Y']*len(intersection)))
    comment_dict.update(zip([i.replace(' ','_') for i in list(set(word_set) - intersection)],['N']*len(set(word_set) - intersection)))

    return comment_dict

# user data for auth
client_id = 'your_client_id'
client_secret = 'your_client_secret'
username = 'your_username'
password = 'your_password'
user_agent = 'your_agent'

# search keyword
keyword = 'beginner'

# post-level attributes to obtain
attrib_list = ['id','author','score']

# words of interest to parse from comments
word_set = ['eye liner','eyeliner','eye shadow','eyeshadow','mascara',
                'primer','brow gel','brow pencil','brow powder','foundation',
                'concealer','highlighter','powder','bronzer','blush','lipstick',
                'lip gloss','lip stain','lip balm','lip tint','lip liner']

# initialize praw utility to help get data
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent,
                     username=username,
                     password=password)

# specify subreddit of interest
subreddit = reddit.subreddit('MakeupAddiction')

keyword_df = get_keyword_data(reddit,subreddit,keyword,attrib_list,word_set)

# modify output df st it has one column for vars like "eye liner"/"eyeliner"
word_pairs = [['eyeliner','eye_liner'],['eyeshadow','eye_shadow'],
              ['brow_pencil','brow_powder'],['lip_stain','lip_tint']]

for pair in word_pairs:
    idx = np.sum(keyword_df[pair]=='Y',axis=1)
    keyword_df.loc[idx>0,pair[0]] = 'Y'
    keyword_df = keyword_df.drop(labels=pair[1:],axis=1)

keyword_df['keyword'] = keyword

keyword_df.to_csv('data/' + keyword + '.csv',index=False)


    


