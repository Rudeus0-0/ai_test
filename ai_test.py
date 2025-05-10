import os 
from langchain_community.chat_models import ChatSparkLLM
from langchain_core.messages import HumanMessage,SystemMessage
from dotenv import load_dotenv

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


 
message = HumanMessage(content="用一句话介绍你自己")
result = chat.invoke([message])
print(result.content)