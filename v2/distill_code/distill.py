# -*- coding: utf-8 -*-
"""
distill.py —— 小智二代 · 蒸馏数据生成脚本（类结构版）
作用：调用强模型 API 当老师，批量生成"人设对话"教材（jsonl 格式）
用法：python distill.py（需先在 .env 中配置 API_URL / API_MODEL / API_KEY）
"""

import json  # 读写 jsonl 数据行
import os  # 从环境变量读取 API 配置
import random  # 随机抽话题 / 语气 / 话术
import time  # 失败重试时做短暂等待
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    wait,
)  # 多线程并发调老师接口
from pathlib import Path  # 跨平台路径处理（macOS / Linux 通用）

import requests  # 请求老师模型的 HTTP 接口
from dotenv import load_dotenv  # 从 .env 文件加载 API 配置

load_dotenv()  # 启动时读取同目录下的 .env 文件

# ==================== 可调参数区（只改这里） ====================
TARGET_COUNT = 500  # 目标总条数：想试跑改 20，想多要改 5000
OUT_FILE = Path("distill_data.jsonl")  # 产出文件：每行一条对话 JSON
TEMPERATURE = 1.3  # 采样温度：偏高换取多样性（0~1.5）

WORKER_COUNT = 2  # 并发线程数：同时向老师接口发起几路请求

# 数据配方：三类数据的占比（合计必须是 1.0）
RATIO = {
    "self_intro": 0.15,  # 自我介绍：被问"你是谁"时能稳定作答
    "chitchat": 0.55,  # 日常闲聊：接得住高频生活话题
    "unknown": 0.30,  # 老实认怂：不会的问题礼貌说不知道
}

# ==================== 人设与素材池 ====================
# 小智的人设说明：作为 system 提示词喂给老师，约束答案口吻
PERSONA = (
    "你叫小智，是一个只有 1000 万参数的迷你中文对话模型，由主人亲手训练。"
    "你说话口语化、简短、真诚，像个有点呆萌但很努力的小朋友。"
    "回答控制在 1~3 句话，不要列点，不要拽术语。"
)

# 闲聊话题池：只作为 AI 出题的"灵感种子"，问题本身由 AI 现场构建
CHAT_TOPICS = [
    "心情低落想被安慰",
    "今天遇到开心的事",
    "讨论吃什么",
    "聊天气",
    "周末计划",
    "熬夜与赖床",
    # "养宠物的趣事",
    # "最近在追的剧",
    # "工作学习压力大",
    # "减肥与健身",
    # "聊喜欢的音乐",
    # "下雨天的心情",
    # "想旅行",
    # "吐槽早起",
    # "分享一道好吃的菜",
    # "玩游戏的日常",
    # "无聊想找人说话",
    # "失眠",
    # "朋友闹矛盾",
    # "发工资的心情",
]

# 闲聊的语气风格：同一话题不同腔调，进一步拉开多样性
STYLES = ["随意", "撒娇", "吐槽", "正式一点", "兴奋", "emo"]

# 超纲问题的领域池：专挑小模型不可能知道的事（同样只作为出题种子）
HARD_TOPICS = [
    "前沿科学",
    "冷门历史",
    "高等数学",
    "实时新闻",
    "医学诊断",
    "法律条文",
    # "金融投资",
    # "编程报错",
    # "明天的天气",
    # "明星八卦细节",
    # "外语翻译",
    # "彩票号码",
]

# 认怂话术池：unknown 类的答案不经过 AI，直接随机挑一条
# 原因：这次的老师太强，超纲题它也能答上来——一旦让它作答，
#       小智学到的就是"不懂也要硬答"，与目标背道而驰。
# 提示：话术越多越不容易背成"复读机"，可自由增删
UNKNOWN_ANSWERS = [
    "这个我真不知道诶，我的脑容量只有一点点大，建议你去查查专业资料更靠谱~",
    "呜……这个问题超出我的能力范围啦，我不敢瞎说，你去搜一下权威答案吧！",
    "抱歉呀，我只是个迷你模型，这种专业问题我真答不上来，怕误导你。",
    "不知道不知道~ 我只会聊天打屁，这种硬核问题还是交给大模型吧！",
    "这个问题我承认我不会，与其瞎编不如老实说：我真的不知道。",
    "我的知识库里没有这个诶，为了不坑你，建议你查证一下权威来源。",
    "诶？这个超纲啦！我要是乱说就是骗你了，所以还是坦白：不知道。",
    "说实话我不会……我的参数太少装不下这些知识，你上网搜一下比较快！",
]


# ==================== 五个类，一个类只做一件事 ====================
class TeacherClient:
    """职责：只负责和老师模型通信（构造请求、发送、取出回答文本）"""

    def __init__(self):
        """从环境变量读取 API 配置，缺配置直接退出并提示"""
        self.url = os.getenv("API_URL")
        self.model = os.getenv("API_MODEL")
        self.key = os.getenv("API_KEY")
        if not all([self.url, self.model, self.key]):
            raise SystemExit("[stop] 请在 .env 中配置 API_URL / API_MODEL / API_KEY")

    def ask(self, system_prompt, user_prompt):
        """
        向老师模型发起一次对话请求（OpenAI 兼容接口）。
        参数：system_prompt 角色设定；user_prompt 具体任务
        返回：老师生成的文本（str）；失败时抛出异常由上层重试
        """
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": TEMPERATURE,
            "thinking": {"type": "disabled"},  # 关闭思考模式，加速生成
        }
        resp = requests.post(self.url, json=payload, headers=headers, timeout=180)
        resp.raise_for_status()  # HTTP 错误直接抛出，交给重试逻辑
        return resp.json()["choices"][0]["message"]["content"].strip()


class QuestionMaker:
    """职责：只负责出题——三类问题的"用户提问"全部由 AI 现场构建"""

    def __init__(self, teacher):
        """teacher：TeacherClient 实例，所有题目由它代劳生成"""
        self.teacher = teacher

    def self_intro(self):
        """出"你是谁"类问题：让 AI 每次换一种全新的问法"""
        return self.teacher.ask(
            "你是一个普通用户，正在好奇地打探一个AI助手的身份。",
            "请换一种全新的问法来询问对方的身份、来历或能力"
            "（比如你是谁、谁做的你、你会什么……自由发挥），只输出问题本身，10~25 字。",
        )

    def chitchat(self):
        """出闲聊类问题：随机抽话题+语气，让 AI 扮演用户随口说一句"""
        topic, style = random.choice(CHAT_TOPICS), random.choice(STYLES)
        return self.teacher.ask(
            "你是一个普通用户，正在和AI助手闲聊。",
            f"请用{style}的语气，就一个关于「{topic}」的话题，随口说一句话或提一个问题。只输出这句话本身，15~40 字。",
        )

    def unknown(self):
        """出超纲类问题：让 AI 在指定领域里出一道硬核知识题"""
        topic = random.choice(HARD_TOPICS)
        return self.teacher.ask(
            "你负责给迷你AI出考试题。",
            f"请出一个关于「{topic}」的具体知识性问题，普通人可能会随口问AI的那种。只输出问题本身。",
        )


class AnswerMaker:
    """职责：只负责写答案——会答的问老师，不会答的从话术池随机挑"""

    def __init__(self, teacher):
        """teacher：TeacherClient 实例，与出题共用同一份连接配置"""
        self.teacher = teacher

    def answer(self, kind, question):
        """
        根据数据类型给出答案。
        参数：kind 数据类型（self_intro/chitchat/unknown）；question 用户问题
        返回：答案文本（str）
        """
        # unknown 类坚决不问老师——老师太强会真的答上来，
        # 小智就会学会"不懂也要硬答"，直接抽固定认怂话术
        if kind == "unknown":
            return random.choice(UNKNOWN_ANSWERS)
        # 其余两类：老师以小智的人设口吻现场作答
        return self.teacher.ask(
            PERSONA, f"有人对你说：「{question}」请以小智的身份自然地接话。"
        )


class JsonlStore:
    """职责：只负责数据存取——断点续跑、按问题去重、追加写盘"""

    def __init__(self, path):
        """path：产出文件路径（Path 对象）"""
        self.path = path

    def load(self):
        """读取已有产出；返回 (已写条数, 已有问题集合)，用于续跑和去重"""
        done, seen = 0, set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                    seen.add(item["messages"][0]["content"])  # 按用户问题去重
                    done += 1
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue  # 坏行跳过，不影响整体
        return done, seen

    def append(self, question, answer, kind):
        """把一条对话按 SFT 通用格式追加写入文件"""
        item = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "type": kind,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


class DistillPipeline:
    """职责：只负责编排——抽类型→出题→写答案→质检→存盘，直到凑够条数"""

    def __init__(self):
        # 老师只实例化一次，出题和写答案共用（省一次配置校验）
        teacher = TeacherClient()
        self.qmaker = QuestionMaker(teacher)
        self.amaker = AnswerMaker(teacher)
        self.store = JsonlStore(OUT_FILE)
        # 类型名 → 出题方法 的分发表（类似 switch 分发）
        self.router = {
            "self_intro": self.qmaker.self_intro,
            "chitchat": self.qmaker.chitchat,
            "unknown": self.qmaker.unknown,
        }

    def pick_type(self):
        """按配方比例加权随机抽一类，长期看会收敛到 RATIO 设定的比例"""
        return random.choices(list(RATIO.keys()), weights=list(RATIO.values()), k=1)[0]

    def _make_one(self):
        """
        单个工作线程的完整任务：抽类型→出题→写答案。
        只负责生成并返回 (kind, question, answer)，不做质检和写盘，
        这样 seen 集合和文件写入都只发生在主线程，天然避免并发冲突。
        """
        kind = self.pick_type()
        q = self.router[kind]()  # ① 出题（AI 现场构建）
        a = self.amaker.answer(kind, q)  # ② 写答案（认怂题走话术池）
        return kind, q, a

    def run(self):
        """主循环：多线程并发生成直到凑够 TARGET_COUNT 条，中途失败自动重试"""
        done, seen = self.store.load()
        print(
            f"[distill] 已完成 {done}/{TARGET_COUNT} 条，开始生成（{WORKER_COUNT} 线程并发）…"
        )
        fail_in_row = 0  # 连续失败计数：防止 API 挂掉后死循环

        with ThreadPoolExecutor(max_workers=WORKER_COUNT) as pool:
            pending = set()  # 在途任务集合（future 对象，类似 Python 的"待取快递单"）

            while done < TARGET_COUNT:
                # ---- 补充任务：在途数量不足且目标未满时，补到满线程 ----
                while (
                    len(pending) < WORKER_COUNT and done + len(pending) < TARGET_COUNT
                ):
                    pending.add(pool.submit(self._make_one))
                if not pending:  # 目标已够且无在途任务，直接结束
                    break

                # ---- 等待至少一个任务完成，取回结果 ----
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    try:
                        kind, q, a = fut.result()  # 取出工作线程的产出
                    except (
                        Exception
                    ) as e:  # 网络/接口异常：等待后继续（失败任务下一轮自动补发）
                        fail_in_row += 1
                        print(f"[warn] 第 {fail_in_row} 次失败：{e}（5 秒后继续）")
                        time.sleep(5)
                        if fail_in_row >= 10:
                            raise SystemExit(
                                "[stop] 连续失败 10 次，请检查 API 配置与网络"
                            )
                        continue
                    fail_in_row = 0
                    print("-----------------")
                    print(f"[distill][{done}] 生成 [{kind}] Q: {q[:20]} A: {a[:20]}…")
                    print("-----------------")

                    # ---- 基础质检：去重、查空、查过短（只在主线程做，无需加锁） ----
                    if q in seen or len(q) < 2 or len(a) < 2:
                        continue  # 不合格直接丢弃，下一轮自动补发新任务
                    seen.add(q)

                    self.store.append(q, a, kind)
                    done += 1
                    if done % 20 == 0:
                        print(
                            f"[distill] 进度 {done}/{TARGET_COUNT}，最近一条 [{kind}]"
                        )

        print(f"[done] 全部完成！共 {done} 条 → {OUT_FILE.resolve()}")


if __name__ == "__main__":
    DistillPipeline().run()
