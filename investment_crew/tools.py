"""
投資AIシステム - カスタムツール群
CrewAIエージェントが使用するツール定義
"""

import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Optional

import requests
import yfinance as yf
import pandas as pd
import numpy as np
from crewai.tools import tool
from ddgs import DDGS

from config import (
    MA_PERIODS, RSI_PERIOD, BOLLINGER_PERIOD, BOLLINGER_STD,
    NEWS_KEYWORDS_EN, NEWS_KEYWORDS_JP, KEYWORDS_PER_SCAN,
)


# ─── ニュース検索ツール ──────────────────────────────────

@tool("NewsSearchTool")
def news_search_tool(query: str, max_results: int = 10) -> str:
    """指定したキーワードで最新ニュースを検索する。
    銘柄名、ティッカー、業界テーマなどを指定可能。
    """
    results = []

    # DuckDuckGo News検索
    try:
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "url": r.get("url", ""),
                    "date": r.get("date", ""),
                    "source": r.get("source", ""),
                })
    except Exception as e:
        results.append({"error": f"DuckDuckGo検索エラー: {str(e)}"})

    return json.dumps(results, ensure_ascii=False, indent=2)


def fetch_google_news_rss(query: str, lang: str = "en", max_results: int = 10) -> list:
    """Google News RSSからニュースを取得する（ツール外部から呼び出し用）"""
    if lang == "ja":
        url = f"https://news.google.com/rss/search?q={query}&hl=ja&gl=JP&ceid=JP:ja"
    else:
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    results = []
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        for item in root.findall(".//item")[:max_results]:
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            pub_date = item.findtext("pubDate", "")
            source = item.findtext("source", "")
            results.append({
                "title": title,
                "url": link,
                "date": pub_date,
                "source": source,
                "body": "",
            })
    except Exception as e:
        results.append({"error": f"Google News RSS エラー: {str(e)}"})

    return results


def collect_all_news(keywords_en: list, keywords_jp: list, per_keyword: int = 5) -> list:
    """英語・日本語キーワードで広域ニュース収集（signal_watcherから呼び出し）"""
    all_news = []
    seen_titles = set()

    for kw in keywords_en:
        # DuckDuckGo
        try:
            with DDGS() as ddgs:
                for r in ddgs.news(kw, max_results=per_keyword):
                    title = r.get("title", "")
                    if title not in seen_titles:
                        seen_titles.add(title)
                        all_news.append({
                            "title": title,
                            "body": r.get("body", ""),
                            "url": r.get("url", ""),
                            "date": r.get("date", ""),
                            "source": r.get("source", ""),
                            "keyword": kw,
                            "lang": "en",
                        })
        except Exception:
            pass

        # Google News RSS
        for item in fetch_google_news_rss(kw, lang="en", max_results=per_keyword):
            title = item.get("title", "")
            if title not in seen_titles and "error" not in item:
                seen_titles.add(title)
                item["keyword"] = kw
                item["lang"] = "en"
                all_news.append(item)

    for kw in keywords_jp:
        # DuckDuckGo（日本語）
        try:
            with DDGS() as ddgs:
                for r in ddgs.news(kw, max_results=per_keyword):
                    title = r.get("title", "")
                    if title not in seen_titles:
                        seen_titles.add(title)
                        all_news.append({
                            "title": title,
                            "body": r.get("body", ""),
                            "url": r.get("url", ""),
                            "date": r.get("date", ""),
                            "source": r.get("source", ""),
                            "keyword": kw,
                            "lang": "ja",
                        })
        except Exception:
            pass

        # Google News RSS（日本語）
        for item in fetch_google_news_rss(kw, lang="ja", max_results=per_keyword):
            title = item.get("title", "")
            if title not in seen_titles and "error" not in item:
                seen_titles.add(title)
                item["keyword"] = kw
                item["lang"] = "ja"
                all_news.append(item)

    return all_news


# ─── 株価・テクニカルツール ───────────────────────────────

@tool("StockPriceTool")
def stock_price_tool(ticker: str) -> str:
    """指定したティッカーの株価データとテクニカル指標を取得する。
    MA20/50/200、RSI14、RVOL、ボリンジャーバンドを計算。
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")

        if hist.empty:
            return json.dumps({"error": f"{ticker}: データ取得失敗"}, ensure_ascii=False)

        close = hist["Close"]
        volume = hist["Volume"]
        current_price = float(close.iloc[-1])

        # 移動平均
        ma = {}
        for p in MA_PERIODS:
            if len(close) >= p:
                ma[f"MA{p}"] = round(float(close.rolling(p).mean().iloc[-1]), 2)

        # RSI
        rsi = None
        if len(close) >= RSI_PERIOD + 1:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            rsi = round(float(rsi_series.iloc[-1]), 1)

        # RVOL（相対出来高）
        rvol = None
        if len(volume) >= 20:
            avg_vol = float(volume.rolling(20).mean().iloc[-1])
            if avg_vol > 0:
                rvol = round(float(volume.iloc[-1]) / avg_vol, 2)

        # ボリンジャーバンド
        bb = {}
        if len(close) >= BOLLINGER_PERIOD:
            sma = close.rolling(BOLLINGER_PERIOD).mean().iloc[-1]
            std = close.rolling(BOLLINGER_PERIOD).std().iloc[-1]
            bb = {
                "upper": round(float(sma + BOLLINGER_STD * std), 2),
                "middle": round(float(sma), 2),
                "lower": round(float(sma - BOLLINGER_STD * std), 2),
            }

        # パーフェクトオーダー判定
        perfect_order = False
        if all(f"MA{p}" in ma for p in MA_PERIODS):
            perfect_order = ma["MA20"] > ma["MA50"] > ma["MA200"]

        # Weinsteinステージ推定
        stage = "不明"
        if "MA200" in ma:
            if current_price > ma["MA200"]:
                if perfect_order:
                    stage = "Stage2（上昇）"
                else:
                    stage = "Stage1（底固め）またはStage2"
            else:
                stage = "Stage3（天井）またはStage4（下降）"

        # 直近5日の変動
        pct_5d = None
        if len(close) >= 6:
            pct_5d = round(float((close.iloc[-1] / close.iloc[-6] - 1) * 100), 2)

        # 前日比
        pct_1d = None
        if len(close) >= 2:
            pct_1d = round(float((close.iloc[-1] / close.iloc[-2] - 1) * 100), 2)

        result = {
            "ticker": ticker,
            "current_price": current_price,
            "change_1d_pct": pct_1d,
            "change_5d_pct": pct_5d,
            "moving_averages": ma,
            "rsi": rsi,
            "rvol": rvol,
            "bollinger_bands": bb,
            "perfect_order": perfect_order,
            "weinstein_stage": stage,
            "volume_latest": int(volume.iloc[-1]) if len(volume) > 0 else None,
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": f"{ticker}: {str(e)}"}, ensure_ascii=False)


def quick_price_check(ticker: str) -> dict:
    """織り込みチェック用の簡易価格データ取得（signal_watcherから呼び出し）"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")

        if hist.empty or len(hist) < 2:
            return {"error": f"{ticker}: データ不足"}

        close = hist["Close"]
        volume = hist["Volume"]

        # 直近の変動率
        pct_change = float((close.iloc[-1] / close.iloc[0] - 1) * 100)

        # RVOL（簡易）
        avg_vol = float(volume.mean())
        latest_vol = float(volume.iloc[-1])
        rvol = round(latest_vol / avg_vol, 2) if avg_vol > 0 else 0

        return {
            "ticker": ticker,
            "current_price": round(float(close.iloc[-1]), 2),
            "change_pct": round(pct_change, 2),
            "rvol": rvol,
            "already_surged": abs(pct_change) > 10,
        }
    except Exception as e:
        return {"error": f"{ticker}: {str(e)}"}


# ─── ファンダメンタルツール ───────────────────────────────

@tool("FundamentalsTool")
def fundamentals_tool(ticker: str) -> str:
    """指定したティッカーのファンダメンタルデータを取得する。
    PER、売上成長率、粗利率、FCF、D/Eレシオ、アナリストコンセンサスなど。
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # 基本情報
        result = {
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName", ticker),
            "sector": info.get("sector", "不明"),
            "industry": info.get("industry", "不明"),
            "market_cap": info.get("marketCap"),
            "currency": info.get("currency", "USD"),
        }

        # バリュエーション
        result["pe_ratio"] = info.get("trailingPE") or info.get("forwardPE")
        result["forward_pe"] = info.get("forwardPE")
        result["peg_ratio"] = info.get("pegRatio")
        result["price_to_book"] = info.get("priceToBook")

        # 成長性
        result["revenue_growth"] = info.get("revenueGrowth")
        result["earnings_growth"] = info.get("earningsGrowth")

        # 収益性
        result["gross_margins"] = info.get("grossMargins")
        result["operating_margins"] = info.get("operatingMargins")
        result["profit_margins"] = info.get("profitMargins")

        # 財務健全性
        result["debt_to_equity"] = info.get("debtToEquity")
        result["current_ratio"] = info.get("currentRatio")
        result["free_cashflow"] = info.get("freeCashflow")

        # アナリスト
        result["target_mean_price"] = info.get("targetMeanPrice")
        result["recommendation"] = info.get("recommendationKey")
        result["number_of_analysts"] = info.get("numberOfAnalystOpinions")

        # 配当
        result["dividend_yield"] = info.get("dividendYield")

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": f"{ticker}: {str(e)}"}, ensure_ascii=False)


# ─── Slack通知ツール ─────────────────────────────────────

@tool("SlackNotifyTool")
def slack_notify_tool(message: str) -> str:
    """Slack Webhookにメッセージを送信する。
    マークダウン形式のメッセージを渡すとSlackに通知される。
    """
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return "エラー: SLACK_WEBHOOK_URL が設定されていません"

    try:
        payload = {"text": message}
        resp = requests.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        return "Slack通知を送信しました"
    except Exception as e:
        return f"Slack通知エラー: {str(e)}"


def send_slack_notification(message: str) -> bool:
    """Slack通知（signal_watcherから直接呼び出し用）"""
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("警告: SLACK_WEBHOOK_URL が未設定")
        return False

    try:
        resp = requests.post(webhook_url, json={"text": message}, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"Slack通知エラー: {e}")
        return False
