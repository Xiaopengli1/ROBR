import openai
import os
import re
import pandas as pd
from tqdm import tqdm
import time
import requests

def req_gen(agent,prompt1, model_type):
    url = "<url>"
    headers={
        # "Content-Type": "application/json",
        "Authorization": "<sk>"
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
    t0 = re.findall('这个问题想要问的是：.*', content1)[0][10:]
    t1 = re.findall(agent+'角色下的改写是：.*', content1)[0][len(agent)+8:]
    return t0,t1
    
# @backoff.on_exception(backoff.expo,requests.exceptions.RequestException)
def req_check(prompt, model_type):
    url = "<url>"
    headers={
        # "Content-Type": "application/json",
        "Authorization": "<sk>"
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
            s0 = int(re.findall('相同信息描述打分：.*', content)[0][9:])
            s1 = int(re.findall('角色匹配打分：.*', content)[0][7:])
        except Exception as e:
            print("无评价获得,retry:"+ str(retry_count))
    return s0,s1

def main():
    openai.api_key = "SK"
    text = pd.read_csv("datasets/raw_data_full.csv")
    query_all = text["query"].apply(lambda x:x.strip()).unique()

    model1 = "gpt-3.5-turbo-16k"
    model2 = "gpt-4"
    agent_list = ["中年妇女", "中年男性", "学生", "老年人"]
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
                    现在有一个搜索的问题是
                    ```{query}```
                    请输出这个搜索的问题真实想要问的问题是什么,
                    假设你是一个{agent}，猜测你可能做出哪些改写策略，按照你的角色改写这个问题。
                    按照以下格式输出。

                    这个问题想要问的是：<想要问的是>
                    {agent}角色下的改写方式可能为：<改写方式>
                    {agent}角色下的改写是：<改写后的问题>

                    """
                elif s0==-1 and s1>=0:
                    prompt1 = f"""
                    现在有一个搜索的问题是
                    ```{query}```
                    请输出这个搜索的问题真实想要问的问题是什么,
                    假设你是一个{agent},按照你的角色改写这个问题,改写时不要过分改变原始问题
                    按照以下格式输出。

                    这个问题想要问的是：<想要问的是>
                    {agent}角色下的改写是：<改写后的问题>

                    """
                elif s0>=0 and s1==-1:
                    prompt1 = f"""
                    现在有一个搜索的问题是
                    ```{query}```
                    请输出这个搜索的问题真实想要问的问题是什么,
                    假设你是一个{agent},按照你的角色改写这个问题, 改写时更加契合角色属性一些, 但不要出现{agent}字眼
                    按照以下格式输出。

                    这个问题想要问的是：<想要问的是>
                    {agent}角色下的改写是：<改写后的问题>

                    """
                else:
                    prompt1 = f"""
                    现在有一个搜索的问题是
                    ```{query}```
                    请输出这个搜索的问题真实想要问的问题是什么,
                    假设你是一个{agent},按照你的角色改写这个问题, 改写时不要过分改变原始问题, 并更加契合角色属性一些。
                    按照以下格式输出。

                    这个问题想要问的是：<想要问的是>
                    {agent}角色下的改写是：<改写后的问题>

                    """           

                _,agent_content = req_gen(agent,prompt1, model1)
                prompt2 = f"""
                原始提问是：{query}。
                改写后的提问是：{agent_content}。
                
                请你做出以下判断：
                1.这两个问题是否描述相同的信息?给出判断分数-1,0,1。-1分代表完全不匹配,0分大致匹配,1代表完全匹配。
                2.改写后的query是否符合{agent}口吻下提出的问题？给出判断分数-1,0,1。-1分代表完全不匹配,0分大致匹配,1代表完全匹配。
                按照以下格式输出。
                相同信息描述打分：<相同信息描述打分>
                角色匹配打分：<角色匹配描述打分>
                """
                s0,s1 = req_check(prompt2, model2)
                print(s0,s1)
                if s0<0 or s1 <0:
                    agent_content = query
            data.append(agent_content)
        data=[data]
        df = pd.DataFrame(data,columns=['原始query',"中年妇女", "中年男性", "学生", "老年人"],dtype="str")                   
        df.to_csv('Rewritedquery_v1.csv', index=False, mode="a", header=not os.path.exists("Rewritedquery_v1.csv"))
    

if __name__ == '__main__':
    main()
