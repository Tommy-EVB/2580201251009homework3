import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import re
from datetime import datetime

# ============================================================
# 第一部分：预设关键词词典（用于从问题文本中提取关键词）
# ============================================================

# 全量关键词词表，涵盖AI基础领域核心概念
KEYWORD_VOCAB = [
    # 基础概念
    "人工智能", "AI", "智能", "机器", "学习", "深度", "算法", "模型", "数据",
    "训练", "预测", "分类", "聚类", "回归", "特征", "损失", "梯度",
    "反向传播", "过拟合", "欠拟合", "正则化", "优化",
    # 学习范式
    "监督学习", "无监督学习", "半监督学习", "强化学习", "迁移学习",
    "在线学习", "批量学习", "增量学习",
    # 模型架构
    "神经网络", "卷积", "循环", "CNN", "RNN", "LSTM", "GRU",
    "Transformer", "注意力机制", "生成对抗", "GAN",
    "编码器", "解码器", "嵌入", "词向量",
    # 大模型与生成式AI
    "GPT", "大模型", "LLM", "生成式", "ChatGPT", "BERT",
    "预训练", "微调", "提示", "Token",
    # 应用领域
    "自然语言处理", "NLP", "计算机视觉", "语音识别", "图像识别",
    "自动驾驶", "机器人", "推荐系统", "对话系统", "知识图谱",
    "专家系统", "问答",
    # 经典算法
    "决策树", "随机森林", "SVM", "支持向量机", "贝叶斯",
    "K近邻", "KNN", "逻辑回归", "线性回归",
    # 评估与工程
    "数据集", "训练集", "测试集", "验证集", "交叉验证", "准确率",
    "召回率", "精确率", "F1", "混淆矩阵",
    # 伦理与安全
    "伦理", "安全", "可解释", "偏见", "隐私", "公平",
    # 测试与图灵
    "图灵测试", "强人工智能", "弱人工智能", "通用人工智能",
    # 编程语言与框架
    "Python", "TensorFlow", "PyTorch", "Java", "C++",
    # 公司与产品
    "OpenAI", "谷歌", "Google", "百度", "DeepMind", "微软",
]


# ============================================================
# 第二部分：知识库类（字典存储、集合管理、倒排索引）
# ============================================================

class KnowledgeBase:
    """
    知识库管理器
    - qa_dict:        字典，键为问题字符串，值为答案字符串
    - keyword_dict:   字典，键为问题字符串，值为该问题的核心关键词集合
    - keyword_index:  字典（倒排索引），键为关键词，值为包含该关键词的问题列表
    - all_keywords:   集合，所有问题的核心关键词去重后的大集合
    """

    def __init__(self):
        self.qa_dict = {}          # {问题: 答案}
        self.keyword_dict = {}     # {问题: {关键词集合}}
        self.keyword_index = {}    # {关键词: [问题列表]}（倒排索引，优化查询效率）
        self.all_keywords = set()  # 所有关键词去重集合
        self._build_knowledge_base()

    def _build_knowledge_base(self):
        """构建知识库：预设20个AI基础问答对"""

        # --- 预设知识库（问题→答案，共20条） ---
        raw_data = [
            ("什么是人工智能？",
             "人工智能（Artificial Intelligence, AI）是计算机科学的一个分支，"
             "致力于创建能够模拟人类智能行为的系统，包括学习、推理、感知、"
             "自然语言理解和决策等能力。AI的核心目标是让机器能够像人一样"
             "思考和行动。"),

            ("Python在人工智能中有什么作用？",
             "Python是AI领域最流行的编程语言。原因包括：语法简洁易学，"
             "拥有NumPy、Pandas等科学计算库，以及TensorFlow、PyTorch等"
             "深度学习框架；社区活跃，开源资源丰富；适合快速原型开发和"
             "科学研究，是AI工程师的首选语言。"),

            ("机器学习和深度学习的区别是什么？",
             "机器学习是AI的核心子领域，通过数据训练模型进行预测和决策，"
             "需要人工提取特征。深度学习是机器学习的子集，使用多层神经网络"
             "自动从原始数据中提取特征，特别适合处理图像、语音、文本等"
             "高维复杂数据。深度学习需要更多数据和算力，但表达能力更强。"),

            ("什么是神经网络？",
             "人工神经网络（ANN）是受生物神经系统启发的计算模型，"
             "由大量节点（神经元）分层连接组成。每个神经元接收输入、"
             "进行加权求和并通过激活函数输出。网络通过反向传播算法"
             "不断调整权重，使输出逐渐接近预期结果。常见架构包括"
             "前馈网络、卷积神经网络（CNN）和循环神经网络（RNN）。"),

            ("什么是监督学习？",
             "监督学习是机器学习的一种范式，训练数据包含输入特征和"
             "对应的标签（正确答案）。模型通过学习输入到输出的映射关系，"
             "对新数据进行预测。常见任务包括分类（如垃圾邮件识别）和"
             "回归（如房价预测）。典型算法有决策树、SVM、逻辑回归等。"),

            ("什么是无监督学习？",
             "无监督学习是机器学习的另一种范式，训练数据没有标签，"
             "模型需要自行发现数据中的内在结构和模式。常见任务包括"
             "聚类（如K-Means将数据分组）、降维（如PCA提取主成分）和"
             "异常检测。与监督学习相比，无需人工标注，但结果解释性较弱。"),

            ("什么是强化学习？",
             "强化学习是机器学习的一种范式，智能体通过与环境交互，"
             "根据获得的奖励或惩罚来学习最优策略。核心概念包括状态、"
             "动作、奖励和策略。强化学习在游戏AI（如AlphaGo）、"
             "机器人控制、自动驾驶等领域表现出色。"),

            ("什么是自然语言处理？",
             "自然语言处理（NLP）是AI和语言学的交叉领域，致力于让计算机"
             "理解、生成和处理人类自然语言。核心技术包括分词、词性标注、"
             "命名实体识别、情感分析、机器翻译等。近年来，基于Transformer"
             "的大模型（如GPT、BERT）极大推动了NLP的发展。"),

            ("什么是计算机视觉？",
             "计算机视觉是让计算机理解和解释图像与视频的技术领域。"
             "核心任务包括图像分类、目标检测、语义分割、人脸识别和"
             "姿态估计等。卷积神经网络（CNN）是计算机视觉的核心架构，"
             "广泛应用在自动驾驶、医学影像、安防监控等场景。"),

            ("什么是迁移学习？",
             "迁移学习是将在一个任务上学到的知识迁移到另一个相关任务"
             "的机器学习方法。核心思想是利用预训练模型的通用特征提取"
             "能力，在目标任务上只需少量数据和微调即可获得良好效果。"
             "这大幅降低了训练成本和所需数据量，是当前大模型应用的主流范式。"),

            ("什么是大语言模型？",
             "大语言模型（LLM）是基于Transformer架构、在海量文本数据上"
             "预训练的超大规模神经网络模型，参数量通常达到数十亿到数千亿。"
             "代表产品包括GPT系列、BERT、LLaMA等。LLM具备文本生成、"
             "问答、翻译、代码编写等多种能力，展现出涌现智能的特征。"),

            ("什么是过拟合？如何解决？",
             "过拟合是指模型在训练数据上表现很好，但在新数据上表现很差，"
             "本质是模型过度记忆了训练数据的噪声而非学到通用规律。"
             "常见解决方法包括：增加训练数据量、使用正则化（L1/L2）、"
             "Dropout随机丢弃神经元、早停法（Early Stopping）、"
             "数据增强、交叉验证等。"),

            ("什么是决策树算法？",
             "决策树是一种树形结构的监督学习算法，通过一系列if-else规则"
             "对数据进行分类或回归。每个内部节点代表一个特征上的判断，"
             "每个分支代表判断结果，叶节点代表最终预测。优点是直观可解释，"
             "缺点是容易过拟合。常见变体包括ID3、C4.5、CART算法。"),

            ("什么是卷积神经网络CNN？",
             "卷积神经网络（CNN）是一类专为处理网格状数据（如图像）"
             "设计的深度学习架构。核心组件包括卷积层（提取局部特征）、"
             "池化层（降低维度）和全连接层（分类输出）。CNN通过参数共享"
             "和局部连接大幅减少参数量，在图像分类、目标检测等领域"
             "取得了突破性成果。代表模型有LeNet、AlexNet、ResNet等。"),

            ("什么是Transformer架构？",
             "Transformer是2017年由谷歌提出的序列建模架构，核心创新是"
             "自注意力机制（Self-Attention），能够并行处理序列中所有位置"
             "的依赖关系，克服了RNN的长距离依赖和串行计算瓶颈。"
             "Transformer是当前几乎所有大语言模型（GPT、BERT等）的"
             "基础架构，彻底改变了NLP乃至整个AI领域。"),

            ("什么是生成对抗网络GAN？",
             "生成对抗网络（GAN）由生成器和判别器两个神经网络组成，"
             "通过对抗博弈联合训练。生成器学习生成逼真的假数据，"
             "判别器学习区分真假数据。两者相互博弈、共同提升。"
             "GAN在图像生成、风格迁移、数据增强等方面应用广泛，"
             "代表变体包括DCGAN、StyleGAN、CycleGAN等。"),

            ("什么是知识图谱？",
             "知识图谱是一种用图结构表示知识的技术，以实体为节点、"
             "关系为边，构建大规模语义网络。核心操作包括知识抽取"
             "（从文本中提取实体和关系）、知识融合和知识推理。"
             "知识图谱广泛应用于搜索引擎、智能问答、推荐系统，"
             "典型代表有谷歌Knowledge Graph和百度知心。"),

            ("什么是图灵测试？",
             "图灵测试由艾伦·图灵于1950年提出，是判断机器是否具有"
             "智能的经典标准。测试中，人类评判者通过文本对话区分"
             "机器和人，如果机器能让评判者无法可靠区分，则认为机器"
             "通过了测试。图灵测试引发了关于'机器能否思考'的长期讨论，"
             "至今仍是AI哲学的重要议题。"),

            ("AI存在哪些伦理和安全问题？",
             "AI伦理与安全问题主要包括：（1）算法偏见——训练数据中的"
             "偏见会被模型放大，导致不公平决策；（2）隐私泄露——模型可能"
             "记忆并泄露训练数据中的敏感信息；（3）可解释性差——深度学习"
             "模型是'黑箱'，决策过程难以理解；（4）滥用风险——深度伪造、"
             "自主武器等；（5）就业冲击——自动化取代部分人类岗位。"
             "负责任的AI开发需要兼顾性能与公平、透明、安全。"),

            ("什么是推荐系统？",
             "推荐系统是利用数据分析和算法向用户推荐相关内容或商品的"
             "系统。主要方法包括：（1）协同过滤——基于用户或物品的相似性"
             "推荐；（2）基于内容——根据物品特征匹配用户偏好；"
             "（3）深度学习——用神经网络学习用户-物品交互模式。"
             "推荐系统广泛应用于电商、短视频、音乐等平台，"
             "是AI商业化最成功的应用之一。"),
        ]

        # --- 构建知识库各数据结构 ---
        for question, answer in raw_data:
            # 1. 字典存储：问题→答案
            self.qa_dict[question] = answer

            # 2. 为每个问题提取核心关键词集合
            keywords = self._extract_keywords(question)
            # 同时补充答案中的关键概念（提升召回率）
            keywords |= self._extract_keywords(answer)
            self.keyword_dict[question] = keywords

            # 3. 所有关键词去重集合
            self.all_keywords |= keywords

            # 4. 倒排索引：关键词→包含该关键词的问题列表
            for kw in keywords:
                if kw not in self.keyword_index:
                    self.keyword_index[kw] = []
                self.keyword_index[kw].append(question)

    def _extract_keywords(self, text):
        """
        从文本中提取关键词
        方法：遍历关键词词典，检查文本中是否包含该关键词
        返回：关键词集合（set）
        """
        found = set()
        text_lower = text.lower()
        for kw in KEYWORD_VOCAB:
            if kw.lower() in text_lower:
                found.add(kw)
        return found

    def get_categories(self):
        """
        按关键词对知识库问题进行分类
        返回：字典 {关键词: [问题列表]}（即倒排索引的视图）
        """
        return self.keyword_index


# ============================================================
# 第三部分：问答引擎（集合交集匹配算法）
# ============================================================

class QAEngine:
    """
    问答匹配引擎
    - 核心算法：提取用户问题关键词→与每个知识库问题的关键词集合求交集
      →交集元素最多的问题即为最佳匹配
    - history: 列表，记录用户所有提问
    """

    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.history = []            # 用户所有提问记录（列表）
        self.match_log = []          # 匹配日志（列表）

    def match_question(self, user_input):
        """
        关键词集合交集匹配算法
        
        算法步骤：
        1. 提取用户输入中的关键词 → user_keywords（集合）
        2. 遍历知识库中每个问题的关键词集合
        3. 计算 user_keywords ∩ question_keywords（集合交集）
        4. 交集元素最多的问题即为匹配度最高的问题
        5. 若无交集，返回未匹配提示
        
        返回：(匹配问题, 答案, 匹配关键词集合, 用户关键词集合)
              未匹配时返回 (None, 提示语, set(), 用户关键词集合)
        """
        # 记录用户提问到列表
        self.history.append(user_input)

        # 步骤1：提取用户问题中的关键词
        user_keywords = self.kb._extract_keywords(user_input)

        # 步骤2-4：遍历知识库，计算集合交集，找最大匹配
        best_question = None
        best_intersection = set()

        # 使用倒排索引优化：只检查与用户关键词相关的问题
        candidate_questions = set()
        for kw in user_keywords:
            if kw in self.kb.keyword_index:
                # 倒排索引查询：获取包含该关键词的所有问题
                for q in self.kb.keyword_index[kw]:
                    candidate_questions.add(q)

        # 对候选问题计算交集（若无候选，则遍历全部）
        search_space = candidate_questions if candidate_questions else self.kb.qa_dict.keys()

        for question in search_space:
            # 集合交集操作
            intersection = user_keywords & self.kb.keyword_dict[question]
            if len(intersection) > len(best_intersection):
                best_intersection = intersection
                best_question = question

        # 步骤5：判断是否匹配成功
        if best_question and len(best_intersection) > 0:
            answer = self.kb.qa_dict[best_question]  # 字典键值对查询
            self.match_log.append({
                "用户提问": user_input,
                "匹配问题": best_question,
                "匹配关键词": sorted(best_intersection),
                "时间": datetime.now().strftime("%H:%M:%S")
            })
            return best_question, answer, best_intersection, user_keywords
        else:
            self.match_log.append({
                "用户提问": user_input,
                "匹配问题": None,
                "匹配关键词": [],
                "时间": datetime.now().strftime("%H:%M:%S")
            })
            return None, "抱歉，未找到相关答案，请尝试其他问题。", set(), user_keywords

    def get_history_slice(self, start=None, end=None):
        """
        获取提问历史（支持列表切片）
        参数：start, end 切片索引
        返回：子列表
        """
        if start is not None or end is not None:
            return self.history[start:end]
        return self.history[:]

    def get_question_count(self):
        """统计提问次数（列表计数）"""
        return len(self.history)

    def get_keyword_frequency(self):
        """
        统计用户提问中各关键词出现频次
        返回：字典 {关键词: 出现次数}
        """
        freq = {}
        for question in self.history:
            kws = self.kb._extract_keywords(question)
            for kw in kws:
                freq[kw] = freq.get(kw, 0) + 1
        return freq


# ============================================================
# 第四部分：可视化交互界面（tkinter）
# ============================================================

class QAApp:
    """人工智能基础问答系统 - GUI界面"""

    # 深色主题配色方案
    BG = "#1e1e2e"
    BG_LIGHT = "#2a2a3c"
    BG_INPUT = "#33334d"
    FG = "#e0e0e0"
    FG_DIM = "#8888aa"
    ACCENT = "#7c6ff7"
    ACCENT_HOVER = "#9589fa"
    GREEN = "#50fa7b"
    RED = "#ff5555"
    ORANGE = "#ffb86c"
    CYAN = "#8be9fd"

    def __init__(self, root):
        self.root = root
        self.root.title("AI 基础问答系统")
        self.root.geometry("960x720")
        self.root.configure(bg=self.BG)
        self.root.minsize(800, 600)

        # 初始化核心组件
        self.kb = KnowledgeBase()
        self.engine = QAEngine(self.kb)

        # 构建界面
        self._build_ui()

        # 欢迎信息
        self._append_display(
            "╔══════════════════════════════════════════╗\n"
            "║     AI 基础问答系统 v1.0                 ║\n"
            "║     知识库：{}个问题，{}个关键词            ║\n"
            "╚══════════════════════════════════════════╝\n\n"
            "请输入您的AI相关问题，系统将通过关键词匹配为您找到答案。\n"
            "输入「退出」结束对话。\n".format(
                len(self.kb.qa_dict), len(self.kb.all_keywords)
            ),
            tag="system"
        )

    def _build_ui(self):
        """构建主界面布局"""
        # ---- 顶部标题栏 ----
        header = tk.Frame(self.root, bg=self.BG_LIGHT, height=50)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(
            header, text="🤖 AI 基础问答系统",
            font=("Microsoft YaHei", 14, "bold"),
            bg=self.BG_LIGHT, fg=self.ACCENT
        ).pack(side=tk.LEFT, padx=15, pady=10)

        # 知识库统计（可点击查看详情）
        stats_frame = tk.Frame(header, bg=self.BG_LIGHT)
        stats_frame.pack(side=tk.RIGHT, padx=15)

        lbl_q = tk.Label(
            stats_frame, text="知识库: {}题".format(len(self.kb.qa_dict)),
            font=("Microsoft YaHei", 9),
            bg=self.BG_LIGHT, fg=self.ACCENT,
            cursor="hand2"
        )
        lbl_q.pack(side=tk.LEFT, padx=(0, 2))
        lbl_q.bind("<Button-1>", lambda e: self._show_all_questions())
        lbl_q.bind("<Enter>", lambda e: lbl_q.configure(fg=self.ACCENT_HOVER))
        lbl_q.bind("<Leave>", lambda e: lbl_q.configure(fg=self.ACCENT))

        tk.Label(
            stats_frame, text="|",
            font=("Microsoft YaHei", 9),
            bg=self.BG_LIGHT, fg=self.FG_DIM
        ).pack(side=tk.LEFT, padx=2)

        lbl_kw = tk.Label(
            stats_frame, text="关键词: {}个".format(len(self.kb.all_keywords)),
            font=("Microsoft YaHei", 9),
            bg=self.BG_LIGHT, fg=self.ACCENT,
            cursor="hand2"
        )
        lbl_kw.pack(side=tk.LEFT, padx=(0, 2))
        lbl_kw.bind("<Button-1>", lambda e: self._show_all_keywords())
        lbl_kw.bind("<Enter>", lambda e: lbl_kw.configure(fg=self.ACCENT_HOVER))
        lbl_kw.bind("<Leave>", lambda e: lbl_kw.configure(fg=self.ACCENT))

        tk.Label(
            stats_frame, text="|",
            font=("Microsoft YaHei", 9),
            bg=self.BG_LIGHT, fg=self.FG_DIM
        ).pack(side=tk.LEFT, padx=2)

        lbl_cat = tk.Label(
            stats_frame, text="分类: {}组".format(len(self.kb.keyword_index)),
            font=("Microsoft YaHei", 9),
            bg=self.BG_LIGHT, fg=self.ACCENT,
            cursor="hand2"
        )
        lbl_cat.pack(side=tk.LEFT, padx=(0, 2))
        lbl_cat.bind("<Button-1>", lambda e: self._show_all_categories_detail())
        lbl_cat.bind("<Enter>", lambda e: lbl_cat.configure(fg=self.ACCENT_HOVER))
        lbl_cat.bind("<Leave>", lambda e: lbl_cat.configure(fg=self.ACCENT))

        # ---- 主区域：左右分栏 ----
        main_frame = tk.Frame(self.root, bg=self.BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 左侧：问答区（占70%宽度）
        left_frame = tk.Frame(main_frame, bg=self.BG, width=650)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # 右侧：信息面板（占30%宽度）
        right_frame = tk.Frame(main_frame, bg=self.BG, width=280)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))

        # ---- 左侧：对话显示区 ----
        self.display = scrolledtext.ScrolledText(
            left_frame, wrap=tk.WORD,
            font=("Microsoft YaHei", 11),
            bg=self.BG_LIGHT, fg=self.FG,
            insertbackground=self.FG,
            relief=tk.FLAT, padx=10, pady=10,
            spacing1=2, spacing3=2
        )
        self.display.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # 配置文本标签样式
        self.display.tag_configure("system", foreground=self.CYAN)
        self.display.tag_configure("user", foreground=self.ORANGE, font=("Microsoft YaHei", 11, "bold"))
        self.display.tag_configure("bot", foreground=self.GREEN)
        self.display.tag_configure("miss", foreground=self.RED)
        self.display.tag_configure("meta", foreground=self.FG_DIM, font=("Microsoft YaHei", 9))
        self.display.tag_configure("separator", foreground=self.FG_DIM)

        # ---- 左侧：输入区域 ----
        input_frame = tk.Frame(left_frame, bg=self.BG)
        input_frame.pack(fill=tk.X)

        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(
            input_frame, textvariable=self.input_var,
            font=("Microsoft YaHei", 12),
            bg=self.BG_INPUT, fg=self.FG,
            insertbackground=self.FG,
            relief=tk.FLAT, bd=0
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 8))
        self.input_entry.bind("<Return>", lambda e: self._on_submit())

        # 发送按钮
        send_btn = tk.Button(
            input_frame, text="发送",
            font=("Microsoft YaHei", 11, "bold"),
            bg=self.ACCENT, fg="white",
            activebackground=self.ACCENT_HOVER,
            relief=tk.FLAT, bd=0, cursor="hand2",
            command=self._on_submit
        )
        send_btn.pack(side=tk.LEFT, ipady=6, ipadx=15)

        # 功能按钮栏
        btn_frame = tk.Frame(left_frame, bg=self.BG)
        btn_frame.pack(fill=tk.X, pady=(5, 0))

        btns = [
            ("查看历史", self._show_history),
            ("关键词统计", self._show_keyword_stats),
            ("知识库分类", self._show_categories),
            ("清空对话", self._clear_display),
        ]
        for text, cmd in btns:
            b = tk.Button(
                btn_frame, text=text,
                font=("Microsoft YaHei", 9),
                bg=self.BG_LIGHT, fg=self.FG_DIM,
                activebackground=self.BG_INPUT,
                relief=tk.FLAT, bd=0, cursor="hand2",
                command=cmd
            )
            b.pack(side=tk.LEFT, padx=(0, 8), ipady=3, ipadx=8)

        # ---- 右侧：信息面板 ----
        # 匹配详情
        tk.Label(
            right_frame, text="匹配详情",
            font=("Microsoft YaHei", 11, "bold"),
            bg=self.BG, fg=self.ACCENT
        ).pack(anchor=tk.W, pady=(5, 5))

        self.detail_display = scrolledtext.ScrolledText(
            right_frame, wrap=tk.WORD,
            font=("Microsoft YaHei", 10),
            bg=self.BG_LIGHT, fg=self.FG,
            relief=tk.FLAT, padx=8, pady=8,
            height=12, spacing1=1, spacing3=1
        )
        self.detail_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.detail_display.tag_configure("key", foreground=self.CYAN)
        self.detail_display.tag_configure("val", foreground=self.GREEN)
        self.detail_display.tag_configure("dim", foreground=self.FG_DIM)

        # 统计信息
        tk.Label(
            right_frame, text="交互统计",
            font=("Microsoft YaHei", 11, "bold"),
            bg=self.BG, fg=self.ACCENT
        ).pack(anchor=tk.W, pady=(5, 5))

        self.stats_label = tk.Label(
            right_frame, text="提问次数: 0\n匹配成功: 0\n匹配失败: 0",
            font=("Microsoft YaHei", 10),
            bg=self.BG_LIGHT, fg=self.FG,
            justify=tk.LEFT, anchor=tk.NW,
            padx=10, pady=10
        )
        self.stats_label.pack(fill=tk.X)

        self._success_count = 0
        self._fail_count = 0

        # 初始聚焦输入框
        self.input_entry.focus_set()

    def _append_display(self, text, tag=None):
        """向对话显示区追加文本"""
        self.display.configure(state=tk.NORMAL)
        if tag:
            self.display.insert(tk.END, text, tag)
        else:
            self.display.insert(tk.END, text)
        self.display.see(tk.END)

    def _on_submit(self):
        """处理用户提交"""
        user_input = self.input_var.get().strip()
        if not user_input:
            return

        self.input_var.set("")

        # 检查退出
        if user_input in ("退出", "exit", "quit", "q"):
            self._handle_exit()
            return

        # 显示用户提问
        self._append_display("\n")
        self._append_display("▶ 你：{}\n".format(user_input), tag="user")

        # 执行匹配
        question, answer, intersection, user_keywords = self.engine.match_question(user_input)

        # 显示匹配结果
        if question:
            self._success_count += 1
            self._append_display("◀ AI：{}\n".format(answer), tag="bot")
            self._append_display(
                "  [匹配问题] {}\n".format(question), tag="meta"
            )
            self._append_display(
                "  [匹配关键词] {}\n".format(", ".join(sorted(intersection))),
                tag="meta"
            )
        else:
            self._fail_count += 1
            self._append_display("◀ AI：{}\n".format(answer), tag="miss")

        # 更新右侧匹配详情
        self._update_detail(user_input, question, intersection, user_keywords)
        # 更新统计
        self._update_stats()

    def _update_detail(self, user_input, matched_question, intersection, user_keywords):
        """更新右侧匹配详情面板"""
        self.detail_display.configure(state=tk.NORMAL)
        self.detail_display.delete("1.0", tk.END)

        self.detail_display.insert(tk.END, "用户输入：\n", "dim")
        self.detail_display.insert(tk.END, "  {}\n\n".format(user_input), "val")

        self.detail_display.insert(tk.END, "提取关键词：\n", "dim")
        if user_keywords:
            for kw in sorted(user_keywords):
                self.detail_display.insert(tk.END, "  • {}\n".format(kw), "key")
        else:
            self.detail_display.insert(tk.END, "  （无匹配关键词）\n", "dim")

        self.detail_display.insert(tk.END, "\n匹配结果：\n", "dim")
        if matched_question:
            self.detail_display.insert(tk.END, "  ✓ {}\n".format(matched_question), "val")
            self.detail_display.insert(tk.END, "\n交集关键词：\n", "dim")
            for kw in sorted(intersection):
                self.detail_display.insert(tk.END, "  ★ {}\n".format(kw), "key")
        else:
            self.detail_display.insert(tk.END, "  ✗ 未匹配\n", "key")

        self.detail_display.insert(tk.END, "\n查询方式：集合交集\n", "dim")
        self.detail_display.insert(tk.END, "  用户关键词 ∩ 问题关键词\n", "dim")
        self.detail_display.insert(tk.END, "  = 匹配度最高的问题\n", "dim")

    def _update_stats(self):
        """更新交互统计"""
        total = self._success_count + self._fail_count
        self.stats_label.configure(
            text="提问次数: {}\n匹配成功: {}\n匹配失败: {}\n命中率: {:.0f}%".format(
                total, self._success_count, self._fail_count,
                (self._success_count / total * 100) if total > 0 else 0
            )
        )

    def _handle_exit(self):
        """处理退出逻辑"""
        total = self.engine.get_question_count()
        self._append_display("\n")
        self._append_display("═══ 对话结束 ═══\n", tag="separator")
        self._append_display(
            "共提问 {} 次，匹配成功 {} 次，命中率 {:.0f}%\n".format(
                total, self._success_count,
                (self._success_count / total * 100) if total > 0 else 0
            ),
            tag="meta"
        )
        self._append_display("感谢使用，再见！\n", tag="system")

        # 弹出统计摘要
        if total > 0:
            summary = "对话统计\n\n"
            summary += "总提问数：{}\n".format(total)
            summary += "匹配成功：{}\n".format(self._success_count)
            summary += "匹配失败：{}\n".format(self._fail_count)
            summary += "命中率：{:.0f}%\n\n".format(
                self._success_count / total * 100 if total > 0 else 0
            )

            # 展示最近5条提问（列表切片）
            recent = self.engine.get_history_slice(-5)
            if recent:
                summary += "最近{}条提问：\n".format(len(recent))
                for i, q in enumerate(recent, 1):
                    summary += "  {}. {}\n".format(i, q)

            messagebox.showinfo("对话结束", summary)

    def _show_history(self):
        """展示用户提问历史（体现列表遍历和切片）"""
        history = self.engine.history
        total = len(history)

        self._append_display("\n", tag="separator")
        self._append_display("═══ 提问历史（共{}次） ═══\n".format(total), tag="system")

        if total == 0:
            self._append_display("  暂无提问记录\n", tag="meta")
            return

        # 列表遍历：逐条展示
        for i, q in enumerate(history, 1):
            self._append_display("  {}. {}\n".format(i, q), tag="meta")

        # 列表切片：展示最近5条
        if total > 5:
            recent = history[-5:]  # 列表切片
            self._append_display(
                "\n  [最近5条] {}\n".format(" | ".join(recent)),
                tag="meta"
            )

    def _show_keyword_stats(self):
        """展示关键词出现频次统计（体现字典遍历）"""
        freq = self.engine.get_keyword_frequency()

        self._append_display("\n", tag="separator")
        self._append_display("═══ 关键词统计 ═══\n", tag="system")

        if not freq:
            self._append_display("  暂无关键词数据\n", tag="meta")
            return

        # 字典遍历：按频次降序排列
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        for kw, count in sorted_freq:
            bar = "█" * count  # 简单的柱状图
            self._append_display(
                "  {} {}: {}\n".format(kw, " " * (12 - len(kw)), bar),
                tag="meta"
            )

    def _show_categories(self):
        """展示知识库按关键词分类（体现字典遍历和倒排索引）"""
        categories = self.kb.get_categories()

        self._append_display("\n", tag="separator")
        self._append_display("═══ 知识库分类（倒排索引） ═══\n", tag="system")
        self._append_display(
            "  共 {} 个关键词分类\n\n".format(len(categories)),
            tag="meta"
        )

        # 字典遍历：按包含问题数量降序展示前15个分类
        sorted_cats = sorted(categories.items(), key=lambda x: len(x[1]), reverse=True)
        for kw, questions in sorted_cats[:15]:
            self._append_display("  【{}】({}题)\n".format(kw, len(questions)), tag="meta")
            # 列表切片：每个分类最多显示3个问题
            for q in questions[:3]:
                # 字符串切片：截取问题前20个字符
                display_q = q[:20] + "..." if len(q) > 20 else q
                self._append_display("    → {}\n".format(display_q), tag="meta")
            if len(questions) > 3:
                self._append_display("    ... 等共{}题\n".format(len(questions)), tag="meta")

    def _show_all_questions(self):
        """点击'知识库:N题'：在对话区展示全部知识库问答内容"""
        self._append_display("\n", tag="separator")
        self._append_display(
            "═══ 知识库全部内容（共{}题） ═══\n".format(len(self.kb.qa_dict)),
            tag="system"
        )

        # 字典遍历：逐条展示每个问题及其答案
        for i, (question, answer) in enumerate(self.kb.qa_dict.items(), 1):
            self._append_display("\n  {}. {}\n".format(i, question), tag="user")
            self._append_display("     {}\n".format(answer), tag="bot")

        self._append_display(
            "\n── 共 {} 条知识库记录 ──\n".format(len(self.kb.qa_dict)),
            tag="meta"
        )

    def _show_all_keywords(self):
        """点击'关键词:N个'：在对话区展示全部去重关键词"""
        self._append_display("\n", tag="separator")
        self._append_display(
            "═══ 全部关键词（共{}个，已去重） ═══\n".format(len(self.kb.all_keywords)),
            tag="system"
        )

        # 集合遍历：按字母/拼音排序展示所有关键词
        sorted_kws = sorted(self.kb.all_keywords)
        # 每行展示5个关键词，格式化对齐
        line = "  "
        for i, kw in enumerate(sorted_kws, 1):
            line += "{:<12}".format(kw)
            if i % 5 == 0:
                self._append_display(line + "\n", tag="meta")
                line = "  "
        if line.strip():
            self._append_display(line + "\n", tag="meta")

        self._append_display(
            "\n── 共 {} 个关键词（集合自动去重） ──\n".format(len(self.kb.all_keywords)),
            tag="meta"
        )

    def _show_all_categories_detail(self):
        """点击'分类:N组'：在对话区展示全部关键词分类及对应问题"""
        categories = self.kb.get_categories()

        self._append_display("\n", tag="separator")
        self._append_display(
            "═══ 知识库按关键词分类（共{}组） ═══\n".format(len(categories)),
            tag="system"
        )

        # 字典遍历：按包含问题数量降序展示全部分类
        sorted_cats = sorted(categories.items(), key=lambda x: len(x[1]), reverse=True)
        for idx, (kw, questions) in enumerate(sorted_cats, 1):
            self._append_display(
                "\n  【{}】{} — 共{}题\n".format(idx, kw, len(questions)),
                tag="meta"
            )
            # 列表遍历：展示该分类下的所有问题
            for j, q in enumerate(questions, 1):
                display_q = q[:25] + "..." if len(q) > 25 else q
                self._append_display("    {}. {}\n".format(j, display_q), tag="meta")

        self._append_display(
            "\n── 共 {} 组分类（倒排索引结构） ──\n".format(len(categories)),
            tag="meta"
        )

    def _clear_display(self):
        """清空对话显示区"""
        self.display.configure(state=tk.NORMAL)
        self.display.delete("1.0", tk.END)
        self.input_entry.focus_set()


# ============================================================
# 第五部分：程序入口
# ============================================================

def main():
    """主函数：启动问答系统GUI"""
    root = tk.Tk()
    app = QAApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
