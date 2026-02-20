# ==========================================
# 电商智能推荐系统 - 主程序框架 (main_app.py)
# ==========================================
import time

# 【团队协作提示】：
# 等大家各自的模块写好后，取消下面几行的注释，把真实的类引入进来
# from models.lstm_predictor import UserSequenceModel
# from models.cnn_extractor import ImageFeatureExtractor
# from models.llm_assistant import RecommendationAssistant
# from utils.data_loader import get_user_history, get_candidate_items

class ECommerceRecommender:
    def __init__(self):
        """
        初始化阶段：在这里加载所有预训练好的模型权重和数据库
        """
        print("[系统] 正在启动推荐系统...")
        print("[系统] 正在加载 LSTM 序列模型权重...")
        # self.lstm_model = UserSequenceModel(weight_path='checkpoints/lstm_best.pth')

        print("[系统] 正在加载 CNN 视觉特征库...")
        # self.cnn_extractor = ImageFeatureExtractor(feature_db='data/processed/item_features.npy')

        print("[系统] 正在初始化 LLM 对话接口...")
        # self.llm_agent = RecommendationAssistant(api_key='your_api_key')

        print("[系统] 初始化完成！\n" + "-"*40)

    def get_user_history(self, user_id):
        """模拟：从数据库获取用户历史购买记录 (后期移入 utils/data_loader.py)"""
        # 假设该用户过去买过黑色鼠标、黑色键盘
        return ["黑色无线鼠标", "黑色机械键盘", "深色电脑包"]

    def retrieve_candidates_by_cnn(self, lstm_preference_vector):
        """模拟：基于 LSTM 的偏好，用 CNN 特征在商品库中寻找最匹配的商品"""
        # 假设 CNN 找出了外观风格最符合的三款商品
        return [
            {"name": "Sony WH-1000XM4", "color": "哑光黑", "price": "$250"},
            {"name": "Bose QuietComfort 45", "color": "碳素黑", "price": "$230"},
            {"name": "Anker Soundcore Life Q30", "color": "曜石黑", "price": "$60"} # 这个符合便宜的要求
        ]

    def recommend(self, user_id, user_query):
        """
        核心推荐流水线：将 LSTM, CNN, LLM 串联起来
        """
        print(f"\n[用户 {user_id} 发起对话]: {user_query}")

        # ---------------------------------------------------------
        # 步骤 1: 获取数据 & LSTM 预测序列偏好 (同学 B 负责)
        # ---------------------------------------------------------
        history = self.get_user_history(user_id)
        print(f"[内部流转-LSTM] 分析用户历史: {history}")
        # lstm_vector = self.lstm_model.predict(history)
        lstm_vector = "[模拟的 LSTM 偏好向量: 偏好深色、极简风格电子配件]"
        time.sleep(0.5) # 模拟计算延迟

        # ---------------------------------------------------------
        # 步骤 2: CNN 视觉匹配找回 (同学 A/B 负责)
        # ---------------------------------------------------------
        print(f"[内部流转-CNN] 正在商品库中匹配符合该偏好向量的图片特征...")
        # candidates = self.cnn_extractor.match(lstm_vector, top_k=3)
        candidates = self.retrieve_candidates_by_cnn(lstm_vector)
        time.sleep(0.5)

        # ---------------------------------------------------------
        # 步骤 3: LLM 综合推理与话术生成 (同学 C 负责)
        # ---------------------------------------------------------
        print(f"[内部流转-LLM] 正在融合用户需求与候选商品，生成最终回复...")
        prompt = f"""
        你是一个聪明的AI导购。
        用户的需求是："{user_query}"
        根据用户的购买历史预测，他可能喜欢极简暗色系的电子产品。
        后台CNN系统为你筛选了以下3个候选商品：{candidates}。
        请结合用户需求（便宜）和他的历史偏好，给出最终推荐，并向他解释推荐理由。
        """
        # final_response = self.llm_agent.chat(prompt)

        # 这里模拟 LLM 的完美输出
        final_response = (
            "🤖 AI导购：您好！根据您的要求，我为您强烈推荐 **Anker Soundcore Life Q30 蓝牙耳机**。\n"
            "这款耳机售价仅为 $60，非常符合您对‘便宜实惠’的需求。\n"
            "此外，我注意到您之前购买过黑色的鼠标和键盘，这款耳机的‘曜石黑’极简设计与您现有的桌面配件风格非常搭配！"
        )
        time.sleep(1)

        print("\n================ 最终展示给用户的结果 ================")
        print(final_response)
        print("======================================================")
        return final_response

# ==========================================
# 启动入口
# ==========================================
if __name__ == "__main__":
    # 1. 实例化系统
    system = ECommerceRecommender()

    # 2. 模拟真实场景测试
    test_user = "User_9527"
    test_query = "我想买一个便宜的蓝牙耳机"

    system.recommend(user_id=test_user, user_query=test_query)