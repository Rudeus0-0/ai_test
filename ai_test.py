import os 
from langchain_community.chat_models import ChatSparkLLM
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()
spark_app_id = os.getenv("SPARK_APP_ID")
spark_api_key = os.getenv("SPARK_API_KEY")
spark_api_secret = os.getenv("SPARK_API_SECRET")

# 访问国外api需要的代理地址
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
# 朗史密斯监控
os.environ['LANGCHIAN_TRACING_V2'] = 'true'
os.environ['LANGCHIAN_API_KEY']='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'

#调用大语言模型
chat = ChatSparkLLM(
    spark_app_id=spark_app_id, 
    spark_api_key=spark_api_key, 
    spark_api_secret=spark_api_secret,
    spark_api_url="wss://spark-api.xf-yun.com/v1.1/chat",
    spark_llm_domain="lite",
 )


# 定义提示模板
promp_template = ChatPromptTemplate.from_messages([
    
    ("system", "你是一个非常有帮助的助手,用{language}尽你所能回答所有问题。"),
    MessagesPlaceholder(variable_name="my_msg"),#占位符，用于存放历史聊天记录


]) 


# 得到链
chain = promp_template | chat


#保存聊天的历史记录
store = {}#所有用户的聊天记录都保存到store中，key为session_id,value是历史聊天记录对象

#此函数预期接受一个sessiojn_id,并返回一个历史聊天记录对象
def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


do_message =  RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="my_msg",#每次聊天时候发送消息的键
    
)


# 第一轮
config = {'configurable': {'session_id': 'zs123'}}#给当前会话定义一个sessionid

resp = do_message.invoke(
    {
        'my_msg':[HumanMessage(content='你好！我是老肖')],
        'language':'中文'
    },
    config = config
)

print(resp.content)


# 第二轮
resp = do_message.invoke(
    {
        'my_msg':[HumanMessage(content='请问我的名字是什么')],
        'language':'中文'
    },
    config = config
)

print(resp.content)


