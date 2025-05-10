# --- 第1部分：导入需要的工具 ---
import os # 用来和操作系统打交道，比如读取环境变量
from langchain_community.chat_models import ChatSparkLLM # 这是和讯飞星火大模型说话的工具
from langchain_core.messages import HumanMessage, SystemMessage # 定义不同角色说的话（人说的，系统说的）
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate # 用来创建对话的“剧本”模板
from langchain_core.runnables.history import RunnableWithMessageHistory # 让机器人能记住之前聊过什么的工具
from dotenv import load_dotenv # 从一个特殊文件 (.env) 加载秘密信息（比如API密钥）的工具
from langchain_community.chat_message_histories import ChatMessageHistory # 用来存放聊天记录的工具（这里用的是社区版，也可以用core版）

# --- 第2部分：准备秘密信息和设置 ---
load_dotenv() # 执行命令：去 .env 文件里找找有没有秘密信息，加载进来！
spark_app_id = os.getenv("SPARK_APP_ID") # 从加载进来的信息里，找出名叫 "SPARK_APP_ID" 的值
spark_api_key = os.getenv("SPARK_API_KEY") # 找出 "SPARK_API_KEY"
spark_api_secret = os.getenv("SPARK_API_SECRET") # 找出 "SPARK_API_SECRET"

# 下面这两行是设置网络代理，如果你的电脑直接就能上网和讯飞服务器通讯，这两行可能不需要
# 它们告诉程序：“如果你要访问网络，请通过 127.0.0.1 这个地址的 7890 端口走”
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'

# 下面这两行是给 LangSmith (一个LangChain的调试监控工具) 用的，如果不用LangSmith，可以不管
# "LANGCHIAN_TRACING_V2" 应该是 "LANGCHAIN_TRACING_V2" (拼写小错误，但不影响这个程序的核心功能)
os.environ['LANGCHIAN_TRACING_V2'] = 'true'
os.environ['LANGCHIAN_API_KEY']='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'

# --- 第3部分：创建和配置我们的聊天机器人核心 (LLM) ---
# 我们要创建一个能和讯飞星火聊天的“机器人实例”
chat = ChatSparkLLM(
    spark_app_id=spark_app_id,       # 把我们的AppID告诉它
    spark_api_key=spark_api_key,     # 把我们的API Key告诉它
    spark_api_secret=spark_api_secret, # 把我们的API Secret告诉它
    # 下面这两个参数是更具体的设置，告诉它用讯飞星火的哪个版本/接口
    spark_api_url="wss://spark-api.xf-yun.com/v1.1/chat", # 这是星火 V1.5 版本的接口地址
    spark_llm_domain="lite",        # "lite" 可能是一个轻量版的模型或者特定的接入点。
                                    # 一般来说，讯飞星火通过 "general", "generalv2" 等来区分版本。
                                    # 如果这个配置能用，说明你的APPID对这个 "lite" 版本有权限。
)

# --- 第4部分：设计对话的“剧本”模板 (Prompt Template) ---
# 我们希望机器人能根据一些规则和我们聊天。这个模板就是规则。
promp_template = ChatPromptTemplate.from_messages([
    # 第一条规则：系统消息 (给机器人设定一个角色和任务)
    # "system" 表示这是系统层面的指示
    # "你是一个非常有帮助的助手,用{language}尽你所能回答所有问题。"
    #    - {language} 是一个占位符，我们后面会告诉它具体是什么语言，比如“中文”。
    ("system", "你是一个非常有帮助的助手,用{language}尽你所能回答所有问题。"),

    # 第二条规则：消息占位符 (这里是关键，用来放聊天记录和用户的新消息)
    # MessagesPlaceholder 告诉LangChain：“嘿，这里会有一堆聊天消息，可能是历史记录，也可能是用户刚说的话。”
    # variable_name="my_msg" 给这个占位符取个名字叫 "my_msg"。
    # 这个名字很重要，后面 RunnableWithMessageHistory 会用到它。
    MessagesPlaceholder(variable_name="my_msg"),
])
# 思考一下：这个模板定义了系统会先说一句话（或者在心里想一句话），然后接下来就是一堆消息（历史+当前）。

# --- 第5部分：将“剧本”和“机器人核心”连接起来，形成一个基础的“链” ---
# "|" 符号在 LangChain 里像一根管道，把左边的输出连到右边的输入。
# 意思就是：先把用户的输入和模板结合，形成完整的对话上下文（剧本演到哪了），
# 然后把这个完整的上下文交给 `chat` (我们的讯飞机器人核心) 去理解并生成回复。
chain = promp_template | chat

# --- 第6部分：准备存放聊天记录的地方 ---
# 我们希望机器人能记住之前的对话，所以需要一个地方存这些记录。
store = {} # 创建一个空的字典，像一个空的档案柜。
           # 我们会用它来存放不同用户（或不同对话）的聊天记录。
           # 字典的 "键" (key) 可以是用户的ID (比如 "zs123")，
           # "值" (value) 就是这个用户的聊天记录本身。

# 这个函数是用来获取特定用户聊天记录的。
# session_id 就是用户的ID。
def get_session_history(session_id: str):
    # 检查一下档案柜里有没有这个用户的档案袋
    if session_id not in store:
        # 如果没有，就为这个用户创建一个新的空档案袋 (ChatMessageHistory 对象)
        store[session_id] = ChatMessageHistory() # ChatMessageHistory就像一个专门记录对话的笔记本
    # 不管是找到了旧的还是创建了新的，都把这个用户的档案袋（笔记本）拿出来。
    return store[session_id]

# --- 第7部分：让我们的“链”拥有记忆功能！ ---
# RunnableWithMessageHistory 是一个魔法包装盒。
# 你把一个普通的“链” (我们上面创建的 chain) 放进去，它就能让这个链拥有记忆。
do_message = RunnableWithMessageHistory(
    runnable=chain, # 把我们基础的“剧本+机器人”链放进去
    get_session_history=get_session_history, # 告诉它怎么去拿特定用户的聊天记录 (用我们刚写的函数)
    input_messages_key="my_msg", # 这是非常重要的一环！还记得我们剧本模板里的 MessagesPlaceholder(variable_name="my_msg") 吗？
                                 # 这个 "my_msg" 就是这里的 input_messages_key。
                                 # 它告诉 RunnableWithMessageHistory：“当用户发来新消息时，
                                 # 你要把这些新消息，连同从 get_session_history 拿到的历史消息，
                                 # 一起塞到剧本模板里那个名叫 'my_msg' 的 MessagesPlaceholder 位置上。”
    # history_messages_key (可选): 如果你的模板里有两个MessagesPlaceholder，
    # 一个专门给历史，一个专门给新输入，那这里就需要区分。
    # 但在我们这个例子里，因为 MessagesPlaceholder 的 variable_name 已经和 input_messages_key 对应了，
    # LangChain 会智能地处理：它会把历史消息和通过 'my_msg' 传入的新消息都放到名为 'my_msg' 的占位符里。
)
# 现在，`do_message` 就是一个能记住对话的、更强大的链了！

# --- 第8部分：开始和机器人聊天！ ---

# 第一轮对话
# config 告诉 `do_message` 这次聊天是属于哪个用户的 (session_id)
# 这样它才能找到正确的聊天记录本。
config = {'configurable': {'session_id': 'zs123'}} # 我们给这次对话取个ID叫 "zs123"

print("--- 第一轮对话开始 ---")
# 调用这个带记忆的链来处理用户的输入
resp = do_message.invoke(
    # 这是我们要传递给链的输入。它是一个字典。
    {
        # "my_msg": 对应 RunnableWithMessageHistory 里的 input_messages_key。
        #    这里我们放的是用户这轮说的话。
        #    注意：这里我们传递的是一个 HumanMessage 对象的列表。
        #    因为我们的 MessagesPlaceholder(variable_name="my_msg") 会接收一个消息列表。
        'my_msg': [HumanMessage(content='你好！我是老肖')],

        # "language": 对应我们剧本模板里的 {language} 占位符。
        'language': '中文'
    },
    config=config # 把上面定义的 session_id 配置传进去
)

# 打印机器人回复的内容
print(f"机器人说: {resp.content}")
# 在这一步之后：
# 1. "你好！我是老肖" (HumanMessage) 和机器人的回复 (AIMessage) 都会被自动保存到 "zs123" 这个 session_id 对应的 ChatMessageHistory 里。

# 第二轮对话
print("\n--- 第二轮对话开始 ---")
resp = do_message.invoke(
    {
        'my_msg': [HumanMessage(content='请问我的名字是什么')], # 用户的新问题
        'language': '中文' # 语言还是中文
    },
    config=config # 还是用同一个 session_id "zs123"，所以机器人应该能查到之前的聊天记录
)

# 打印机器人回复的内容
print(f"机器人说: {resp.content}")
# 机器人这时候应该能回答出 "老肖" 或者类似的内容，因为它记住了第一轮的对话！
