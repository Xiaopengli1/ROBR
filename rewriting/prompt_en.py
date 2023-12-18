

import openai
import os
import re
import pandas as pd
from tqdm import tqdm
import time
import requests
import backoff
def req_gen(agent,prompt1, model_type):
    url = "url"
    headers={
        # "Content-Type": "application/json",
        "Authorization": "sk"
    }
    params1={
        "model": model_type,
        "messages": [],
        "temperature":0.7,
        "max_tokens": 1024,
        # "stream": True, 
        "top_p": 1
    }
    message=[]

    message.append({"role": "user", "content":prompt1})
    params1["messages"] = message
    response1 = requests.post(url=url, headers=headers,json=params1)
    message1 = response1.json()
    content1 = message1['choices'][0]['message']['content']
    print(content1)
    t0 = re.findall('What this query intend to ask is:.*', content1)[0][33:]
    t1 = re.findall("The rewrite query under the "+agent+" role is:.*", content1)[0][len(agent)+36:]
    return t0,t1

    
# @backoff.on_exception(backoff.expo,requests.exceptions.RequestException)
def req_check(prompt, model_type):
    url = "<url>"
    headers={
        # "Content-Type": "application/json",
        "Authorization": "sk"
    }
    params={
        "model": model_type,
        "messages": [],
        "temperature":0.7,
        "max_tokens": 1024,
        # "stream": True, 
        "top_p": 1
    }
    message=[]
    message.append({"role": "user", "content":prompt})
    params["messages"] = message
    retry_count = 0
    s0,s1 = -3 , -3
    while retry_count<10 and s0<-2 and s1<-2:    
        try:
            retry_count+=1
            time.sleep(1)
            response = requests.post(url=url, headers=headers,json=params)
            message = response.json()
            content = message['choices'][0]['message']['content']
            print( content )
            s0 = int(re.findall('Score for the same information description:.*', content)[0][43:])
            s1 = int(re.findall('Role matching score:.*', content)[0][20:])
        except Exception as e:
            print("retry:"+ str(retry_count))
    return s0,s1


text = pd.read_csv("raw_data_full.csv")
query_all = text["query"].apply(lambda x:x.strip()).unique()

model1 = "gpt-3.5-turbo-16k"
model2 = "gpt-4"
agent_list = ["Middle-aged woman", "Middle-aged man", "Student", "Old person"]
# agent = agent_list[0]

for query in tqdm(query_all):
    data=[query]
    count_perquery_peragent = 0
    for agent in agent_list:
        query_a = None
        s0 = -2
        s1 = -2
        while s0<0 or s1<0 and count_perquery_peragent<5:
            count_perquery_peragent+=1
            if s0==-2 and s1==-2:
                prompt1 = f"""

                The search query is
                 ```{query}```
                 Please analyze and determine the actual intention or meaning that the person is trying to convey through this search query.                 
                 Assuming you are an {agent}, what changes might you make when rewriting the query, rewrite the question according to your role.
                 Output in the following format.

                 What this query intend to ask is: <What I want to ask>
                 The changes might be: <changes>
                 The rewrite query under the {agent} role is: <rewritten query>

                """
                
            elif s0==-1 and s1>=0:
                prompt1 = f"""

                There is a search query that is
                 ```{query}```
                 Please analyze and determine the actual intention or meaning that the person is trying to convey through this search query.                 
                 Assuming you are an {agent},  please rephrase the query in accordance with your role while preserving the original meaning of the question.
                 Output in the following format.

                 What this query intend to ask is: <What I want to ask>
                 The rewrite query under the {agent} role is: <rewritten query>

                """

            elif s0>=0 and s1==-1:
                prompt1 = f"""
                There is a search query that is
                 ```{query}```
                 Please analyze and determine the actual intention or meaning that the person is trying to convey through this search query.
                 Assuming you are an {agent}, please rephrase the query according to your role and rewrite it more in line with the character's attributes.
                 Output in the following format.

                 What this query intend to ask is: <What I want to ask>
                 The rewrite query under the {agent} role is: <rewritten query>

                """
            else:
                prompt1 = f"""
                There is a search query that is
                 ```{query}```
                 Please output what is the real intent peroson want to ask in this search query,
                 Assuming you are an {agent}, please rephrase the question consistent with your role, maintaining the essence of the original query and aligning it with the character's attributes.
                 Output in the following format.

                 What this query intend to ask is: <What I want to ask>
                 The rewrite query under the {agent} role is: <rewritten query>

                """           

            _,agent_content = req_gen(agent,prompt1, model1)
            prompt2 = f"""
            The original query is: {query}.
            The rephrased query is: {agent_content}.
            
            Evaluate the following:
            1. Are these two queries describing the same information? Give a judgment score -1,0,1. A score of -1 indicates no match at all, 0 indicates an approximate match, and 1 indicates an exact match.
            2. Does the modified query align with the query posed by {agent}? Give judgment scores -1,0,1. A score of -1 indicates no match at all, 0 indicates an approximate match, and 1 indicates an exact match.
            Output in the following format.

            Score for the same information description: <Score for the same information description>
            Role matching score: <role matching description score>
            """
            s0,s1 = req_check(prompt2, model2)
            print(s0,s1)
            if s0<0 or s1 <0:
                agent_content = query
        data.append(agent_content)
    data=[data]
    df = pd.DataFrame(data,columns=['raw_query',"woman", "man", "student", "old"],dtype="str")                   
    df.to_csv('Rewritedquery_eng.csv', index=False, mode="a", header=not os.path.exists("Rewritedquery_eng.csv"))