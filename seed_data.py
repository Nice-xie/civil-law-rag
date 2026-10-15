# -*- coding: utf-8 -*-
"""向 chat_history 灌入测试提问，供数据分析看板与聚类实验演示。

运行：python seed_data.py
会把 50 条典型民事咨询（5 类 × 每类 10 条）按随机时间（近 30 天）写入 chat_history 表。
脚本可重复运行：会先清空旧的 seed_ 演示数据再插入，避免重复。
"""
import random
import datetime
import pymysql

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "civil_law_rag_db",
    "charset": "utf8mb4",
}

QUESTIONS = [
    # 民间借贷
    "朋友借了我2万元写了借条，到期不还怎么办？",
    "借款利息约定月息3分，超过法定上限了吗？",
    "没有借条只有微信转账记录，能要回借款吗？",
    "借钱时被预先扣了利息，本金怎么算？",
    "套取银行贷款转贷给别人，合同有效吗？",
    "职业放贷人的借款合同是否无效？",
    "夫妻一方借款用于家庭生活，另一方要共同还吗？",
    "借款没约定利息，能主张利息吗？",
    "借条上没写还款日期，出借人能随时要求还款吗？",
    "出借人把借条原件弄丢了，还能起诉要回借款吗？",
    # 婚约彩礼
    "订婚给付彩礼8万元，没领证分手了能要回吗？",
    "结婚登记了但没共同生活，离婚能要求返还彩礼吗？",
    "恋爱期间发红包转账，分手能要回吗？",
    "男方悔婚，彩礼能全额退吗？",
    "彩礼已经用于共同生活开销，返还时要扣除吗？",
    "借结婚名义索取财物，能要求返还吗？",
    "双方都有过错，彩礼返还比例怎么定？",
    "节日礼物算彩礼吗？",
    "订婚后没领证就同居，现在分手彩礼要返还吗？",
    "男方给女方父母的彩礼，离婚时能要求返还吗？",
    # 劳动纠纷
    "上班8个月没签劳动合同，能要求双倍工资吗？",
    "公司无故辞退我，赔偿金怎么算？",
    "公司不交社保，我能解除合同要经济补偿吗？",
    "加班没有加班费，怎么维权？",
    "试用期被无理由辞退合法吗？",
    "工作满10年公司拒签无固定期限合同怎么办？",
    "违反竞业限制要赔违约金吗？",
    "拖欠工资多年，超过仲裁时效了吗？",
    "加班费是按基本工资还是全额工资计算？",
    "公司不给开离职证明，影响我找新工作怎么办？",
    # 合同纠纷
    "付了全款对方不发货，能要求继续履行吗？",
    "提前退租要承担违约金吗？",
    "商家收款后取消订单算违约吗？",
    "服务只完成一半能退款吗？",
    "卖方反悔不卖房，要赔房屋差价吗？",
    "交了定金对方违约，能双倍返还吗？",
    "违约金约定过高能请求减少吗？",
    "合同条款有两种解释，按哪种？",
    "装修合同中途停工，能解除合同要求退款吗？",
    "买二手车发现是事故车，能解除买卖合同吗？",
    # 民事诉讼程序
    "合同纠纷应该去哪个法院起诉？",
    "超过三年没主张债权，还能起诉吗？",
    "谁主张谁举证是什么意思？",
    "被告不到庭能缺席判决吗？",
    "小额诉讼是一审终审吗？",
    "担心对方转移财产，能申请财产保全吗？",
    "微信聊天记录能作为证据吗？",
    "达成调解协议能申请司法确认吗？",
    "一审判决不服，上诉期是多长时间？",
    "申请强制执行需要向哪个法院提交？",
]


def main():
    conn = pymysql.connect(**DB_CONFIG)
    cur = conn.cursor()
    # 幂等：先清掉旧的 seed_ 演示数据，再插入，避免重复
    cur.execute("DELETE FROM chat_history WHERE session_id LIKE 'seed\\_%'")
    now = datetime.datetime.now()
    for i, q in enumerate(QUESTIONS):
        days_ago = random.randint(0, 30)
        create_time = now - datetime.timedelta(
            days=days_ago,
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )
        session_id = f"seed_{i:03d}"
        answer = "（测试数据）本回答仅用于看板演示，详见知识库对应法条。"
        cur.execute(
            "INSERT INTO chat_history(session_id, user_query, ai_answer, create_time) "
            "VALUES(%s, %s, %s, %s)",
            (session_id, q, answer, create_time),
        )
    conn.commit()
    cur.close()
    conn.close()
    print(f"已插入 {len(QUESTIONS)} 条测试提问到 chat_history（5 类 × 每类 10 条）。")


if __name__ == "__main__":
    main()
