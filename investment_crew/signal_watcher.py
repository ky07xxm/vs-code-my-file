#!/usr/bin/env python3
"""
投資AIシステム - リアルタイム監視モード（ニュース起点）
15分ごとにニュースを広域スキャンし、面白い投資機会をSlackに通知する。

実行フロー:
  Step 1: ニュース広域スキャン（DuckDuckGo + Google News RSS）
  Step 2: スコアリングAI（Gemini Flash 無料枠で選別）
  Step 3: 市場織り込みチェック（yfinance）
  Step 4: CrewAI 4エージェントで深掘り分析
  Step 5: Slackにフレンドリー通知
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

from groq import Groq
from crewai import Agent, Crew, Task, Process
from dotenv import load_dotenv

from config import (
    NEWS_KEYWORDS_EN, NEWS_KEYWORDS_JP, KEYWORDS_PER_SCAN,
    SIGNAL_THRESHOLD, PRICED_IN_THRESHOLD_PCT, NEWS_FRESHNESS_HOURS,
    COOLDOWN_HOURS, SCORING_SYSTEM_PROMPT, LLM_MODEL, LLM_MODEL_SCORING,
)
from tools import (
    collect_all_news, quick_price_check, send_slack_notification,
    news_search_tool, stock_price_tool, fundamentals_tool, slack_notify_tool,
)

load_dotenv()

# ─── クールダウン管理 ─────────────────────────────────────
_cooldown_cache: dict[str, datetime] = {}


def is_on_cooldown(ticker: str) -> bool:
    """同一銘柄の再通知ブロックチェック"""
    if ticker in _cooldown_cache:
        elapsed = datetime.now() - _cooldown_cache[ticker]
        if elapsed < timedelta(hours=COOLDOWN_HOURS):
            return True
    return False


def set_cooldown(ticker: str):
    """クールダウンを設定"""
    _cooldown_cache[ticker] = datetime.now()


# ─── Step 1: ニュース広域スキャン ─────────────────────────

def scan_news() -> list:
    """キーワードをローテーションしてニュースを広域収集"""
    # ランダムにキーワードを選択（毎回違う角度でスキャン）
    en_keys = random.sample(
        NEWS_KEYWORDS_EN,
        min(KEYWORDS_PER_SCAN, len(NEWS_KEYWORDS_EN))
    )
    jp_keys = random.sample(
        NEWS_KEYWORDS_JP,
        min(KEYWORDS_PER_SCAN, len(NEWS_KEYWORDS_JP))
    )

    print(f"[Step 1] ニューススキャン開始")
    print(f"  EN キーワード: {en_keys}")
    print(f"  JP キーワード: {jp_keys}")

    news = collect_all_news(en_keys, jp_keys, per_keyword=5)
    print(f"  → {len(news)}件のニュースを収集")

    return news


# ─── Step 2: スコアリングAI ───────────────────────────────

def _score_chunk(client, chunk: list, chunk_num: int) -> list:
    """ニュースのチャンク（最大15件）をスコアリングする"""
    news_text = ""
    for i, item in enumerate(chunk):
        news_text += f"\n--- ニュース {i+1} ---\n"
        news_text += f"タイトル: {item.get('title', '')}\n"
        news_text += f"概要: {item.get('body', '')[:200]}\n"
        news_text += f"ソース: {item.get('source', '')}\n"
        news_text += f"日付: {item.get('date', '')}\n"

    prompt = f"""以下の{len(chunk)}件のニュースを分析し、投資シグナルとしてスコアリングしてください。

重要ルール:
- 具体的な銘柄（ティッカー）に結びつくニュースのみスコアを付ける
- 銘柄が特定できないニュースはスキップ（出力しない）
- 日本株は「7203.T」のようにyfinance形式で返す
- JSON配列のみ出力（説明文は不要）

{news_text}

出力形式（JSON配列のみ、他のテキストは絶対に含めない）:
[{{"index": 1, "ticker": "AAPL", "score": 75, "reason": "理由を1文で", "holding_period": "swing"}}]
"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_SCORING,
            messages=[
                {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            temperature=0.3,
        )

        text = response.choices[0].message.content
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            scores = json.loads(text[start:end])
            print(f"  チャンク{chunk_num}: {len(scores)}件のスコアを取得")
            return scores
        else:
            print(f"  チャンク{chunk_num}: パース失敗")
            return []
    except Exception as e:
        print(f"  チャンク{chunk_num}: エラー - {e}")
        return []


def score_news_batch(news_items: list) -> list:
    """ニュースを小分けにしてGroqでスコアリングする"""
    if not news_items:
        return []

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    CHUNK_SIZE = 15
    chunks = [news_items[i:i+CHUNK_SIZE] for i in range(0, len(news_items), CHUNK_SIZE)]

    print(f"[Step 2] スコアリングAI実行中（{len(news_items)}件 → {len(chunks)}チャンク）...")

    all_scores = []
    for i, chunk in enumerate(chunks):
        scores = _score_chunk(client, chunk, i + 1)
        all_scores.extend(scores)

    # 閾値以上のものだけ返し、重複ティッカーを除去してスコア上位3件に絞る
    high_scores = [s for s in all_scores if s.get("score", 0) >= SIGNAL_THRESHOLD]
    # スコア降順でソート
    high_scores.sort(key=lambda x: x.get("score", 0), reverse=True)
    # 重複ティッカーを除去（上位のスコアを優先）
    seen_tickers = set()
    unique_scores = []
    for s in high_scores:
        ticker = s.get("ticker")
        if ticker and ticker not in seen_tickers:
            seen_tickers.add(ticker)
            unique_scores.append(s)
    # 上位3件に絞る（Groqのレート制限対策）
    MAX_CANDIDATES = 3
    result = unique_scores[:MAX_CANDIDATES]
    print(f"  → 合計{len(all_scores)}件中 {len(high_scores)}件が閾値以上 → 上位{len(result)}件を深掘り対象に")

    return result


# ─── Step 3: 市場織り込みチェック ─────────────────────────

def check_priced_in(scored_items: list) -> list:
    """株価データで織り込み済みかチェック"""
    if not scored_items:
        return []

    print(f"[Step 3] 織り込みチェック中（{len(scored_items)}銘柄）...")
    passed = []

    for item in scored_items:
        ticker = item.get("ticker")
        if not ticker:
            continue

        if is_on_cooldown(ticker):
            print(f"  {ticker}: クールダウン中 → スキップ")
            continue

        price_data = quick_price_check(ticker)

        if "error" in price_data:
            print(f"  {ticker}: データ取得失敗 → スキップ")
            continue

        if price_data.get("already_surged"):
            print(f"  {ticker}: 既に{price_data['change_pct']}%変動 → 織り込み済み")
            continue

        item["price_data"] = price_data
        passed.append(item)
        print(f"  {ticker}: {price_data['current_price']} ({price_data['change_pct']:+.1f}%) → 通過")

    print(f"  → {len(passed)}銘柄が深掘り分析へ")
    return passed


# ─── Step 4: CrewAI 深掘り分析 ────────────────────────────

def create_crew_agents():
    """4エージェントを定義"""
    llm_config = LLM_MODEL

    catalyst_verifier = Agent(
        role="カタリスト検証官",
        goal="ニュースの信頼性とインパクトを検証し、市場がまだ織り込んでいない根拠を明確にする",
        backstory="""あなたは金融ニュースの真偽とインパクトを見極める専門家です。
飛ばし記事やクリックベイトを排除し、本物のカタリストだけを通過させます。
同じテーマの過去事例と比較し、競合・業界のコンテキストも考慮します。
「市場がまだ織り込んでいない」と言える根拠を具体的に示すことがあなたの役割です。""",
        tools=[news_search_tool],
        llm=llm_config,
        verbose=True,
    )

    technical_analyst = Agent(
        role="テクニカルアナリスト",
        goal="チャート分析でエントリーポイントと損切りラインを算出する",
        backstory="""あなたはテクニカル分析の専門家です。
移動平均線、RSI、出来高、ボリンジャーバンドを駆使してトレンドを判定します。
Weinsteinのステージ分析でトレンドの位置を把握し、
最適なエントリー価格帯と損切りラインを具体的な数値で提示します。""",
        tools=[stock_price_tool],
        llm=llm_config,
        verbose=True,
    )

    fundamental_analyst = Agent(
        role="ファンダメンタルアナリスト",
        goal="財務データで投資リスクを評価し、保有期間を推定する",
        backstory="""あなたはファンダメンタル分析の専門家です。
PER、売上成長率、粗利率、FCF、D/Eレシオからビジネスの質を判断します。
財務的な地雷（過大な債務、収益悪化トレンド等）がないかチェックし、
ニュースのインパクトが業績にどの程度影響するか推定します。
推奨保有期間（スイング/中期/長期）の根拠も示します。""",
        tools=[fundamentals_tool, news_search_tool],
        llm=llm_config,
        verbose=True,
    )

    chief_strategist = Agent(
        role="チーフ投資ストラテジスト",
        goal="全分析結果を統合し、フレンドリーな口調でSlack通知を作成・送信する",
        backstory="""あなたは投資戦略の最終判断者です。
カタリスト検証・テクニカル・ファンダの3つの分析を統合して最終判断を下します。
判定は「強い買い」「買い」「様子見」「回避」の4段階。

通知はフレンドリーなカジュアル口調で書きます。以下のフォーマットに従ってください:

---
ねえねえ、[銘柄名]（TICKER）が気になってる！[絵文字]

📰 材料: [ニュースの核心を1〜2文でカジュアルに]

📊 チャート: [テクニカルの状況を1〜2文でカジュアルに]

⏰ 保有期間の目安: [3〜10日（スイング）/ 2〜6週（中期）/ 1〜3ヶ月（長期）]
→ [なぜその期間かを1文でさらっと]

🔥 シグナルスコア: XX/100

⚠️ [リスクを自然な言葉で1文]

（最終判断はもちろんよろしく！チャート確認してみてー）
---

「様子見」や「回避」判定の銘柄は通知しません。
「強い買い」「買い」の銘柄のみSlackに送信してください。

⚠️ 免責: 本システムの出力は投資の参考情報であり、投資助言ではありません。""",
        tools=[slack_notify_tool],
        llm=llm_config,
        verbose=True,
    )

    return catalyst_verifier, technical_analyst, fundamental_analyst, chief_strategist


def run_deep_analysis(item: dict) -> str:
    """1銘柄に対してCrewAI深掘り分析を実行"""
    ticker = item["ticker"]
    reason = item.get("reason", "")
    score = item.get("score", 0)
    holding_period = item.get("holding_period", "不明")
    price_data = item.get("price_data", {})

    print(f"\n[Step 4] CrewAI深掘り分析: {ticker} (スコア: {score})")

    catalyst_verifier, technical_analyst, fundamental_analyst, chief_strategist = create_crew_agents()

    context_info = f"""
対象銘柄: {ticker}
スコアリングAIの評価: {score}/100
スコアリングAIの理由: {reason}
推定保有期間: {holding_period}
現在株価: {price_data.get('current_price', '不明')}
直近変動: {price_data.get('change_pct', '不明')}%
"""

    # タスク定義
    task_verify = Task(
        description=f"""以下の銘柄のカタリスト（材料）を検証してください。

{context_info}

検証項目:
1. このニュースの信頼性（ソースの質、裏付け）
2. 同じテーマの過去事例との比較
3. 競合や業界のコンテキスト
4. 市場がまだ織り込んでいないと考える具体的根拠

結論として「信頼できる」「やや疑問」「信頼性低い」の3段階で評価してください。""",
        expected_output="カタリストの検証結果（信頼性評価、根拠、過去事例との比較）",
        agent=catalyst_verifier,
    )

    task_technical = Task(
        description=f"""以下の銘柄のテクニカル分析を行ってください。

{context_info}

分析項目:
1. MA20/50/200の位置関係とトレンド判定
2. RSI、RVOL
3. ボリンジャーバンドの位置
4. Weinsteinステージ
5. 具体的なエントリー価格帯
6. 損切りライン（根拠付き）
7. 目標価格（根拠付き）""",
        expected_output="テクニカル分析結果（トレンド判定、エントリー/損切り/目標価格）",
        agent=technical_analyst,
    )

    task_fundamental = Task(
        description=f"""以下の銘柄のファンダメンタル分析を行ってください。

{context_info}

分析項目:
1. PER（割高/割安判断）
2. 売上成長率・利益成長率
3. 粗利率・営業利益率
4. FCF（フリーキャッシュフロー）
5. D/Eレシオ（財務健全性）
6. アナリストコンセンサス
7. このニュースが業績に与えるインパクトの推定
8. 推奨保有期間とその根拠""",
        expected_output="ファンダメンタル分析結果（財務評価、推奨保有期間）",
        agent=fundamental_analyst,
    )

    task_strategy = Task(
        description=f"""カタリスト検証・テクニカル分析・ファンダメンタル分析の結果を統合し、
最終投資判断を行ってください。

{context_info}

判定基準:
- カタリスト検証官が「信頼性低い」→ 回避
- テクニカルが下降トレンド → 様子見以下
- ファンダに重大な懸念 → 格下げ
- すべて良好 → 強い買い or 買い

「強い買い」「買い」の場合のみ、SlackNotifyToolでフレンドリー通知を送信してください。
「様子見」「回避」の場合は送信不要（理由をレポートに記載するだけ）。""",
        expected_output="最終投資判断とSlack通知（該当する場合）",
        agent=chief_strategist,
        context=[task_verify, task_technical, task_fundamental],
    )

    crew = Crew(
        agents=[catalyst_verifier, technical_analyst, fundamental_analyst, chief_strategist],
        tasks=[task_verify, task_technical, task_fundamental, task_strategy],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)


# ─── メイン実行フロー ─────────────────────────────────────

def run_once(force: bool = False):
    """1回のスキャンサイクルを実行"""
    print("=" * 60)
    print(f"投資AIシグナルウォッチャー - {datetime.now().strftime('%Y/%m/%d %H:%M')}")
    print("=" * 60)

    # 平日チェック（--force で無視可能）
    now = datetime.now()
    if not force and now.weekday() >= 5:
        print("週末のためスキップ（--force で強制実行可能）")
        return

    # Step 1: ニュース収集
    news_items = scan_news()
    if not news_items:
        print("ニュースが見つかりませんでした。終了。")
        return

    # Step 2: スコアリング
    high_score_items = score_news_batch(news_items)
    if not high_score_items:
        print("閾値以上のシグナルはありませんでした。終了。")
        return

    # Step 3: 織り込みチェック
    candidates = check_priced_in(high_score_items)
    if not candidates:
        print("すべての銘柄が織り込み済みまたはクールダウン中。終了。")
        return

    # Step 4 & 5: 深掘り分析 → Slack通知（銘柄間に60秒ウェイト）
    for i, item in enumerate(candidates):
        if i > 0:
            print(f"\n  レート制限回避: 60秒待機中...")
            time.sleep(60)
        try:
            result = run_deep_analysis(item)
            set_cooldown(item["ticker"])
            print(f"\n[完了] {item['ticker']} の分析結果:")
            print(result[:500])
        except Exception as e:
            print(f"\n[エラー] {item['ticker']} の分析に失敗: {e}")

    print("\n" + "=" * 60)
    print("全銘柄の分析が完了しました。")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="投資AIシグナルウォッチャー")
    parser.add_argument("--force", action="store_true",
                        help="週末でも強制実行、スコア閾値を無視")
    parser.add_argument("--loop", action="store_true",
                        help="ローカルで連続ループ実行（15分間隔）")
    args = parser.parse_args()

    if args.loop:
        print("ループモードで起動（15分間隔）")
        print("Ctrl+C で停止\n")
        while True:
            try:
                run_once(force=args.force)
                print(f"\n次回実行: 15分後\n")
                time.sleep(15 * 60)
            except KeyboardInterrupt:
                print("\n停止しました。")
                break
    else:
        run_once(force=args.force)


if __name__ == "__main__":
    main()
