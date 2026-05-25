#!/usr/bin/env python3
"""미국 주식 모니터링 대시보드 생성 스크립트"""

import os
import json
import datetime
import traceback
import time
import warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import yfinance as yf

# ─── 상수 ────────────────────────────────────────────────────────────────────
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
TODAY_STR = datetime.date.today().isoformat()  # e.g. "2025-05-25"

# ISM 캐시 파일 (당일 이미 수집했으면 스킵)
ISM_CACHE_FILE = "/tmp/ism_cache.json"

# ─── FRED 데이터 수집 ─────────────────────────────────────────────────────────
def fetch_fred(series_id: str, limit: int = 12) -> list:
    """FRED API에서 시계열 데이터를 가져와 최근 limit개 반환"""
    try:
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={FRED_API_KEY}"
            f"&file_type=json&sort_order=desc&limit={limit}"
        )
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        values = []
        for o in obs:
            try:
                v = float(o["value"])
                values.append({"date": o["date"], "value": v})
            except (ValueError, KeyError):
                pass
        return values
    except Exception as e:
        print(f"[FRED] {series_id} 오류: {e}")
        return []


def latest_fred(series_id: str) -> float | None:
    data = fetch_fred(series_id, limit=5)
    return data[0]["value"] if data else None


def trend_direction(series_id: str, periods: int = 3) -> str:
    """'상승중' / '하락중' / '횡보' 반환"""
    data = fetch_fred(series_id, limit=periods + 2)
    if len(data) < 2:
        return "횡보"
    vals = [d["value"] for d in data[:periods]]
    if vals[0] > vals[-1] * 1.005:
        return "상승중"
    elif vals[0] < vals[-1] * 0.995:
        return "하락중"
    return "횡보"


# ─── yfinance 헬퍼 ────────────────────────────────────────────────────────────
def fetch_yf_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        return df
    except Exception as e:
        print(f"[yfinance] {ticker} 히스토리 오류: {e}")
        return pd.DataFrame()


def fetch_yf_current(ticker: str) -> dict:
    """현재가, 전일종가, 등락률 반환"""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        if hist.empty:
            return {}
        price = float(hist["Close"].iloc[-1])
        prev  = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price
        chg   = (price - prev) / prev * 100
        return {"price": price, "prev": prev, "change_pct": chg}
    except Exception as e:
        print(f"[yfinance] {ticker} 현재가 오류: {e}")
        return {}


# ─── 기술적 지표 계산 ─────────────────────────────────────────────────────────
def calc_rsi(series: pd.Series, period: int = 14) -> float | None:
    try:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, float("nan"))
        rsi   = 100 - (100 / (1 + rs))
        v = float(rsi.iloc[-1])
        return round(v, 1) if not pd.isna(v) else None
    except Exception:
        return None


def calc_macd(series: pd.Series):
    """(macd_line, signal_line, histogram) 반환"""
    try:
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist  = macd - signal
        return float(macd.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])
    except Exception:
        return None, None, None


def calc_bollinger(series: pd.Series, period: int = 20):
    """(upper, middle, lower) 반환"""
    try:
        mid   = series.rolling(period).mean()
        std   = series.rolling(period).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        return float(upper.iloc[-1]), float(mid.iloc[-1]), float(lower.iloc[-1])
    except Exception:
        return None, None, None


def calc_ma(series: pd.Series, period: int) -> float | None:
    try:
        v = float(series.rolling(period).mean().iloc[-1])
        return round(v, 2) if not pd.isna(v) else None
    except Exception:
        return None


def calc_stop_loss(df: pd.DataFrame, current_price: float) -> float | None:
    """손절선: 200MA가 현재가 대비 -5% 이내면 200MA, 아니면 20일 최저가"""
    try:
        ma200 = calc_ma(df["Close"], 200)
        if ma200 and current_price > 0:
            ratio = (ma200 - current_price) / current_price
            if ratio >= -0.05:
                return round(ma200, 2)
        low20 = float(df["Close"].tail(20).min())
        return round(low20, 2)
    except Exception:
        return None


def full_technical(ticker: str, is_korean: bool = False) -> dict:
    """종목 기술적 지표 전체 계산"""
    result = {"ticker": ticker}
    try:
        period = "2y" if not is_korean else "1y"
        df = fetch_yf_history(ticker, period=period)
        if df.empty:
            raise ValueError("히스토리 없음")

        close = df["Close"]
        price = float(close.iloc[-1])
        prev  = float(close.iloc[-2]) if len(close) >= 2 else price
        chg   = (price - prev) / prev * 100

        result["price"]      = round(price, 2)
        result["change_pct"] = round(chg, 2)
        result["rsi"]        = calc_rsi(close)

        macd, signal, hist = calc_macd(close)
        result["macd"]        = round(macd, 4)   if macd   is not None else None
        result["macd_signal"] = round(signal, 4) if signal is not None else None
        result["macd_hist"]   = round(hist, 4)   if hist   is not None else None

        bb_upper, bb_mid, bb_lower = calc_bollinger(close)
        result["bb_upper"] = round(bb_upper, 2) if bb_upper else None
        result["bb_mid"]   = round(bb_mid, 2)   if bb_mid   else None
        result["bb_lower"] = round(bb_lower, 2) if bb_lower else None

        ma50  = calc_ma(close, 50)
        ma200 = calc_ma(close, 200)
        result["ma50"]  = ma50
        result["ma200"] = ma200

        # 골든/데드 크로스
        if ma50 and ma200:
            if ma50 > ma200:
                result["cross"] = "골든크로스"
            else:
                result["cross"] = "데드크로스"
        else:
            result["cross"] = "N/A"

        # 200MA 대비 위치
        if ma200:
            result["vs_ma200"] = "위" if price > ma200 else "아래"
        else:
            result["vs_ma200"] = "N/A"

        # BB 위치
        if bb_upper and bb_lower:
            bb_range = bb_upper - bb_lower
            if bb_range > 0:
                bb_pct = (price - bb_lower) / bb_range * 100
                result["bb_position"] = round(bb_pct, 1)
            else:
                result["bb_position"] = 50.0
        else:
            result["bb_position"] = None

        # 매수선/매도선/손절선
        result["buy_line"]  = round(bb_lower, 2) if bb_lower else None
        result["sell_line"] = round(bb_upper, 2) if bb_upper else None
        result["stop_loss"] = calc_stop_loss(df, price)

        # 거래량 (20일 평균 대비)
        if "Volume" in df.columns and len(df) >= 21:
            vol_today = float(df["Volume"].iloc[-1])
            vol_avg20 = float(df["Volume"].iloc[-21:-1].mean())
            result["vol_ratio"] = round(vol_today / vol_avg20, 2) if vol_avg20 > 0 else None
        else:
            result["vol_ratio"] = None

        # 기술적 종합 신호
        buy_signals = 0
        if result["rsi"] is not None and result["rsi"] < 40:
            buy_signals += 1
        if macd is not None and signal is not None and macd > signal:
            buy_signals += 1
        if bb_lower and price <= bb_lower * 1.02:
            buy_signals += 1
        if ma200 and price > ma200:
            buy_signals += 1

        if buy_signals >= 3:
            result["tech_signal"] = "매수"
        elif buy_signals <= 1:
            result["tech_signal"] = "매도"
        else:
            result["tech_signal"] = "중립"

        # 진입 신호 (탭 3·4 용)
        entry_signals = 0
        if result["rsi"] is not None and result["rsi"] <= 30:
            entry_signals += 1
        if macd is not None and signal is not None and macd > signal:
            entry_signals += 1
        if bb_lower and price <= bb_lower * 1.01:
            entry_signals += 1
        if ma200 and price > ma200:
            entry_signals += 1
        result["entry_signals"] = entry_signals

    except Exception as e:
        print(f"[기술적지표] {ticker} 오류: {e}")

    return result


# ─── 거시지표 수집 ────────────────────────────────────────────────────────────
def collect_macro() -> dict:
    m = {}

    # VIX
    try:
        vix_data = fetch_yf_current("^VIX")
        m["vix"] = round(vix_data.get("price", 0), 2) if vix_data else None
    except Exception as e:
        print(f"[VIX] 오류: {e}")
        m["vix"] = None

    # 공포탐욕지수 (CNN 1차 → alternative.me 백업)
    try:
        score, rating = None, None
        # 1차: CNN (URL 2가지 시도)
        cnn_urls = [
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{datetime.date.today().isoformat()}",
        ]
        for url in cnn_urls:
            try:
                r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                r.raise_for_status()
                fd = r.json()
                fg = fd.get("fear_and_greed") or fd
                score = fg.get("score") or fg.get("now", {}).get("score")
                rating = fg.get("rating") or fg.get("now", {}).get("text", "")
                if score is not None:
                    break
            except Exception:
                continue
        # 2차 백업: alternative.me
        if score is None:
            r2 = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            r2.raise_for_status()
            fng = r2.json()["data"][0]
            score = float(fng["value"])
            rating = fng["value_classification"]
            print(f"[공포탐욕] alternative.me 백업 사용: {score}")
        m["fear_greed"] = {"score": round(float(score), 1), "rating": rating}
    except Exception as e:
        print(f"[공포탐욕] 오류: {e}")
        m["fear_greed"] = None

    # 풋/콜 비율 (^VIX 대용 — CBOE는 직접 API 없음, yfinance 옵션 체인 활용)
    try:
        spy = yf.Ticker("SPY")
        exp_dates = spy.options
        if exp_dates:
            chain = spy.option_chain(exp_dates[0])
            total_put_vol  = chain.puts["volume"].sum()
            total_call_vol = chain.calls["volume"].sum()
            pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else None
            m["put_call"] = round(float(pc_ratio), 3) if pc_ratio else None
        else:
            m["put_call"] = None
    except Exception as e:
        print(f"[풋콜비율] 오류: {e}")
        m["put_call"] = None

    # FRED 데이터들
    fred_map = {
        "yield_spread": "T10Y2Y",
        "hy_spread":    "BAMLH0A0HYM2",
        "rate_10y":     "DGS10",
        "rate_2y":      "DGS2",
        "cpi":          "CPIAUCSL",
        "pce":          "PCEPI",
        "unemployment": "UNRATE",
        "jobless":      "ICSA",
        "nfp":          "PAYEMS",
        "m2":           "M2SL",
        "fed_balance":  "WALCL",
        "reserves":     "WRESBAL",
        "rrp":          "RRPONTSYD",
        "tga":          "WTREGEN",
        "stlfsi":       "STLFSI4",
    }
    for key, sid in fred_map.items():
        try:
            m[key] = latest_fred(sid)
        except Exception as e:
            print(f"[FRED:{sid}] 오류: {e}")
            m[key] = None

    # 방향 지표
    try:
        m["hy_spread_trend"]    = trend_direction("BAMLH0A0HYM2")
        m["yield_spread_trend"] = trend_direction("T10Y2Y")
        m["unemployment_trend"] = trend_direction("UNRATE")
        m["cpi_trend"]          = trend_direction("CPIAUCSL")
        m["m2_trend"]           = trend_direction("M2SL")
        m["fed_balance_trend"]  = trend_direction("WALCL")
    except Exception as e:
        print(f"[방향지표] 오류: {e}")

    # DXY
    try:
        dxy = fetch_yf_current("DX-Y.NYB")
        m["dxy"] = round(dxy.get("price", 0), 3) if dxy else None
        # DXY 방향 (5일 비교)
        dxy_hist = fetch_yf_history("DX-Y.NYB", "1mo")
        if not dxy_hist.empty and len(dxy_hist) >= 5:
            cur = float(dxy_hist["Close"].iloc[-1])
            old = float(dxy_hist["Close"].iloc[-5])
            if cur > old * 1.005:
                m["dxy_trend"] = "강세급등"
            elif cur < old * 0.995:
                m["dxy_trend"] = "약세전환"
            else:
                m["dxy_trend"] = "횡보"
        else:
            m["dxy_trend"] = "횡보"
    except Exception as e:
        print(f"[DXY] 오류: {e}")
        m["dxy"] = None
        m["dxy_trend"] = "횡보"

    # SOX
    try:
        sox = fetch_yf_current("^SOX")
        m["sox"] = round(sox.get("price", 0), 2) if sox else None
        m["sox_change"] = round(sox.get("change_pct", 0), 2) if sox else None
    except Exception as e:
        print(f"[SOX] 오류: {e}")
        m["sox"] = None
        m["sox_change"] = None

    # 구리
    try:
        cu = fetch_yf_current("HG=F")
        m["copper"] = round(cu.get("price", 0), 4) if cu else None
        m["copper_change"] = round(cu.get("change_pct", 0), 2) if cu else None
    except Exception as e:
        print(f"[구리] 오류: {e}")
        m["copper"] = None
        m["copper_change"] = None

    # 원달러 환율
    try:
        krw = fetch_yf_current("KRW=X")
        m["krw_usd"] = round(krw.get("price", 0), 2) if krw else None
        m["krw_change"] = round(krw.get("change_pct", 0), 2) if krw else None
    except Exception as e:
        print(f"[환율] 오류: {e}")
        m["krw_usd"] = None
        m["krw_change"] = None

    # Polymarket FOMC 예측
    try:
        poly_url = "https://clob.polymarket.com/markets"
        r = requests.get(
            poly_url,
            params={"tag_slug": "fomc", "limit": 5},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        r.raise_for_status()
        markets = r.json().get("data", [])
        if markets:
            top = markets[0]
            m["polymarket_fomc"] = {
                "question": top.get("question", "FOMC"),
                "yes_price": top.get("tokens", [{}])[0].get("price", "N/A"),
            }
        else:
            m["polymarket_fomc"] = None
    except Exception as e:
        print(f"[Polymarket] 오류: {e}")
        m["polymarket_fomc"] = None

    # ISM PMI (Claude Haiku 웹서치)
    m["ism_pmi"] = fetch_ism_pmi()

    return m


# ─── ISM PMI (Claude Haiku) ──────────────────────────────────────────────────
def fetch_ism_pmi() -> float | None:
    """하루 1회만 호출. 캐시 파일로 관리."""
    try:
        if os.path.exists(ISM_CACHE_FILE):
            with open(ISM_CACHE_FILE) as f:
                cache = json.load(f)
            if cache.get("date") == TODAY_STR:
                print(f"[ISM PMI] 캐시 사용: {cache.get('value')}")
                return cache.get("value")
    except Exception:
        pass

    if not ANTHROPIC_API_KEY:
        print("[ISM PMI] ANTHROPIC_API_KEY 없음, 스킵")
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=256,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 1,
            }],
            messages=[{
                "role": "user",
                "content": (
                    "Search for the most recent ISM Manufacturing PMI value. "
                    "Reply with ONLY the numeric value (e.g. 49.2). "
                    "No explanation needed."
                )
            }]
        )
        # 응답에서 숫자 추출
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
        import re
        numbers = re.findall(r"\b\d{2}\.\d\b", text)
        if numbers:
            val = float(numbers[0])
            with open(ISM_CACHE_FILE, "w") as f:
                json.dump({"date": TODAY_STR, "value": val}, f)
            print(f"[ISM PMI] 새 값 수집: {val}")
            return val
        print(f"[ISM PMI] 숫자 파싱 실패. 응답: {text[:200]}")
        return None
    except Exception as e:
        print(f"[ISM PMI] 오류: {e}")
        return None


# ─── 점수 계산 ───────────────────────────────────────────────────────────────
def calc_scores(m: dict) -> dict:
    short_score = 0
    mid_score   = 0
    long_score  = 0

    # ── 단기 ──
    vix = m.get("vix")
    if vix is not None:
        if vix < 20:   short_score += 25
        elif vix < 30: short_score += 12

    fg = m.get("fear_greed")
    fg_score_val = fg["score"] if fg else None
    if fg_score_val is not None:
        if fg_score_val <= 25:  short_score += 25
        elif fg_score_val <= 60: short_score += 12

    pc = m.get("put_call")
    if pc is not None:
        if pc >= 1.2:   short_score += 25
        elif pc >= 0.7: short_score += 12

    hy_trend = m.get("hy_spread_trend", "횡보")
    if hy_trend == "하락중":   short_score += 15
    elif hy_trend == "횡보":   short_score += 7

    dxy_trend = m.get("dxy_trend", "횡보")
    if dxy_trend == "약세전환": short_score += 10
    elif dxy_trend == "횡보":   short_score += 5

    # ── 중기 ──
    ys_trend = m.get("yield_spread_trend", "횡보")
    ys_val   = m.get("yield_spread")
    if ys_trend == "상승중":  mid_score += 20  # 역전 해소 중
    elif ys_trend == "횡보":  mid_score += 10

    hy = m.get("hy_spread")
    if hy is not None:
        if hy <= 4:   mid_score += 20
        elif hy <= 6: mid_score += 10

    unemp_trend = m.get("unemployment_trend", "횡보")
    if unemp_trend == "하락중": mid_score += 20
    elif unemp_trend == "횡보": mid_score += 10

    nfp = m.get("nfp")
    if nfp is not None:
        nfp_k = nfp / 1000  # 단위: 천명 → 만명
        if nfp_k >= 200:   mid_score += 15
        elif nfp_k >= 100: mid_score += 7

    cpi_trend = m.get("cpi_trend", "횡보")
    if cpi_trend == "하락중":  mid_score += 15
    elif cpi_trend == "횡보":  mid_score += 7

    ism = m.get("ism_pmi")
    if ism is not None:
        if ism >= 52:   mid_score += 10
        elif ism >= 50: mid_score += 5

    # ── 장기 ──
    m2_trend = m.get("m2_trend", "횡보")
    if m2_trend == "상승중":   long_score += 30
    elif m2_trend == "횡보":   long_score += 15

    fed_trend = m.get("fed_balance_trend", "횡보")
    if fed_trend == "상승중":  long_score += 25   # QE
    elif fed_trend == "횡보":  long_score += 12   # 중립/감속

    if ys_val is not None:
        ys_pct = ys_val  # T10Y2Y 단위 이미 %
        if ys_pct >= 0.5:   long_score += 20
        elif ys_pct >= 0:   long_score += 10

    unemp_data = fetch_fred("UNRATE", limit=4)
    if len(unemp_data) >= 4:
        u_vals = [d["value"] for d in unemp_data[:4]]
        if u_vals[0] < u_vals[-1]:  long_score += 15  # 하락 지속
        elif u_vals[0] == u_vals[-1]: long_score += 7

    stlfsi = m.get("stlfsi")
    if stlfsi is not None:
        if stlfsi < 0:   long_score += 10
        elif stlfsi < 1: long_score += 5

    # 역발상 보정
    hy_abs = m.get("hy_spread", 999)
    if vix is not None and stlfsi is not None:
        if 30 <= vix < 40 and stlfsi < 1:
            long_score += 10
        if vix >= 40 and stlfsi < 3:
            long_score += 20
    if fg_score_val is not None and hy_abs is not None:
        if fg_score_val <= 15 and hy_abs < 10:
            long_score += 10

    def label(score):
        if score >= 70:   return ("강한 매수", "green")
        elif score >= 50: return ("약한 매수", "yellow")
        elif score >= 30: return ("중립/약한 매도", "orange")
        return ("강한 매도", "red")

    return {
        "short":  {"score": min(short_score, 100), "label": label(short_score)[0], "color": label(short_score)[1]},
        "mid":    {"score": min(mid_score,   100), "label": label(mid_score)[0],   "color": label(mid_score)[1]},
        "long":   {"score": min(long_score,  100), "label": label(long_score)[0],  "color": label(long_score)[1]},
    }


# ─── 섹터 종목 수집 ──────────────────────────────────────────────────────────
US_SECTORS = {
    "반도체":        ["NVDA", "AMD", "TSM", "ASML", "SNDK"],
    "전력·인프라":   ["NEE", "VST", "CEG", "TSLA", "ETN", "BE"],
    "우주":          ["RKLB", "LUNR", "PL"],
    "소프트웨어·AI": ["MSFT", "PLTR", "CDNS"],
}
US_SCALP   = ["NVDA", "TSLA", "SOFI", "MARA"]
US_SCALP_NAMES = {"NVDA": "엔비디아", "TSLA": "테슬라", "SOFI": "소파이", "MARA": "마라홀딩스"}
KR_SCALP   = ["086520", "068270", "035720", "058470", "000720"]
KR_NAMES   = {
    "086520": "에코프로", "068270": "셀트리온", "035720": "카카오",
    "058470": "리노공업", "000720": "현대건설",
}


def fetch_earnings_info(ticker: str) -> dict:
    """다음 실적발표일, EPS 서프라이즈 정보"""
    info = {
        "next_earnings": None, "earnings_soon": False,
        "eps_actual": None, "eps_estimate": None,
        "eps_surprise_pct": None, "target_price": None,
    }
    try:
        t = yf.Ticker(ticker)

        # 캘린더 — 최신 yfinance는 dict 반환
        try:
            cal = t.calendar
            next_date = None
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date", None)
                if ed is not None:
                    if isinstance(ed, (list, tuple)) and len(ed) > 0:
                        next_date = pd.Timestamp(ed[0]).date()
                    elif not isinstance(ed, (list, tuple)):
                        next_date = pd.Timestamp(ed).date()
            elif cal is not None and hasattr(cal, "empty") and not cal.empty:
                if "Earnings Date" in cal.index:
                    ed = cal.loc["Earnings Date"]
                    next_date = pd.Timestamp(ed.iloc[0] if hasattr(ed, "iloc") else ed).date()
            if next_date:
                info["next_earnings"] = str(next_date)
                info["earnings_soon"] = 0 <= (next_date - datetime.date.today()).days <= 7
        except Exception as e:
            print(f"[캘린더] {ticker}: {e}")

        # EPS 히스토리
        try:
            earnings = t.earnings_history
            eps_actual = eps_estimate = None
            if earnings is not None:
                if hasattr(earnings, "empty") and not earnings.empty:
                    last = earnings.iloc[-1]
                    eps_actual   = last.get("epsActual",   None)
                    eps_estimate = last.get("epsEstimate", None)
                elif isinstance(earnings, list) and earnings:
                    last = earnings[-1]
                    eps_actual   = last.get("epsActual",   None)
                    eps_estimate = last.get("epsEstimate", None)
            info["eps_actual"]   = float(eps_actual)   if eps_actual   is not None else None
            info["eps_estimate"] = float(eps_estimate) if eps_estimate is not None else None
            if eps_actual and eps_estimate and eps_estimate != 0:
                info["eps_surprise_pct"] = round((eps_actual - eps_estimate) / abs(eps_estimate) * 100, 1)
        except Exception as e:
            print(f"[EPS] {ticker}: {e}")

        # 애널리스트 목표주가
        try:
            rec = t.analyst_price_targets
            if isinstance(rec, dict):
                info["target_price"] = rec.get("mean", None)
        except Exception as e:
            print(f"[목표주가] {ticker}: {e}")

    except Exception as e:
        print(f"[실적정보] {ticker} 전체 오류: {e}")

    return info


def collect_us_sectors(short_score: int) -> dict:
    result = {}
    for sector, tickers in US_SECTORS.items():
        sector_data = []
        for tk in tickers:
            print(f"  [섹터] {sector}/{tk} 수집 중...")
            tech = full_technical(tk)
            earn = fetch_earnings_info(tk)
            item = {**tech, **earn, "sector": sector, "short_score_warn": short_score <= 30}
            sector_data.append(item)
        result[sector] = sector_data
    return result


def collect_us_scalp() -> list:
    result = []
    for tk in US_SCALP:
        print(f"  [미국단타] {tk} 수집 중...")
        tech = full_technical(tk)
        result.append(tech)
    return result


def collect_kr_scalp() -> list:
    result = []
    try:
        from pykrx import stock as pykrx_stock
        end_date   = datetime.date.today().strftime("%Y%m%d")
        start_date = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y%m%d")

        for code in KR_SCALP:
            print(f"  [한국단타] {code} 수집 중...")
            item = {"ticker": code, "name": KR_NAMES.get(code, code)}
            try:
                # 주가 데이터
                ohlcv = pykrx_stock.get_market_ohlcv_by_date(start_date, end_date, code)
                if ohlcv.empty:
                    raise ValueError("데이터 없음")

                close = ohlcv["종가"]
                price = float(close.iloc[-1])
                prev  = float(close.iloc[-2]) if len(close) >= 2 else price
                item["price"]      = price
                item["change_pct"] = round((price - prev) / prev * 100, 2)
                item["rsi"]        = calc_rsi(close)

                macd, signal, hist = calc_macd(close)
                item["macd"]        = round(macd, 2)   if macd   is not None else None
                item["macd_signal"] = round(signal, 2) if signal is not None else None

                bb_upper, bb_mid, bb_lower = calc_bollinger(close)
                item["bb_upper"] = round(bb_upper, 0) if bb_upper else None
                item["bb_lower"] = round(bb_lower, 0) if bb_lower else None

                ma200 = calc_ma(close, 200)
                item["ma200"]   = ma200
                item["vs_ma200"] = "위" if (ma200 and price > ma200) else "아래"

                # 볼린저 위치
                if bb_upper and bb_lower:
                    bb_range = bb_upper - bb_lower
                    item["bb_position"] = round((price - bb_lower) / bb_range * 100, 1) if bb_range > 0 else 50.0
                else:
                    item["bb_position"] = None

                # 매수선/손절선
                item["buy_line"]  = round(bb_lower, 0) if bb_lower else None
                tmp_df = pd.DataFrame({"Close": close.values})
                item["stop_loss"] = calc_stop_loss(tmp_df, price)

                # 거래량
                if "거래량" in ohlcv.columns and len(ohlcv) >= 21:
                    vol_today = float(ohlcv["거래량"].iloc[-1])
                    vol_avg20 = float(ohlcv["거래량"].iloc[-21:-1].mean())
                    item["vol_ratio"] = round(vol_today / vol_avg20, 2) if vol_avg20 > 0 else None
                else:
                    item["vol_ratio"] = None

                # 기관/외국인 순매수
                try:
                    trade_df = pykrx_stock.get_market_trading_value_by_date(
                        start_date, end_date, code
                    )
                    if not trade_df.empty and len(trade_df) >= 3:
                        # 기관
                        inst_col = [c for c in trade_df.columns if "기관" in c]
                        if inst_col:
                            inst_last3 = trade_df[inst_col[0]].iloc[-3:]
                            item["inst_buy_3d"] = bool((inst_last3 > 0).all())
                        else:
                            item["inst_buy_3d"] = False
                        # 외국인
                        for_col = [c for c in trade_df.columns if "외국인" in c]
                        if for_col:
                            for_last3 = trade_df[for_col[0]].iloc[-3:]
                            item["for_buy_3d"] = bool((for_last3 > 0).all())
                        else:
                            item["for_buy_3d"] = False
                    else:
                        item["inst_buy_3d"] = False
                        item["for_buy_3d"]  = False
                except Exception as e2:
                    print(f"  [pykrx 매매동향] {code} 오류: {e2}")
                    item["inst_buy_3d"] = False
                    item["for_buy_3d"]  = False

                # 진입 신호 카운트
                entry_signals = 0
                if item["rsi"] is not None and item["rsi"] <= 30:
                    entry_signals += 1
                if macd is not None and signal is not None and macd > signal:
                    entry_signals += 1
                if bb_lower and price <= bb_lower * 1.01:
                    entry_signals += 1
                if ma200 and price > ma200:
                    entry_signals += 1
                item["entry_signals"] = entry_signals
                item["strong_entry"]  = entry_signals >= 3 and (item["inst_buy_3d"] or item["for_buy_3d"])

            except Exception as e:
                print(f"  [한국단타] {code} 데이터 오류: {e}")
                item["price"]       = None
                item["change_pct"]  = None
                item["rsi"]         = None
                item["entry_signals"] = 0
                item["strong_entry"]  = False

            result.append(item)
            time.sleep(0.5)  # KRX 서버 과부하 방지

    except ImportError:
        print("[pykrx] 라이브러리 없음. 한국 종목 N/A 처리")
        for code in KR_SCALP:
            result.append({
                "ticker": code, "name": KR_NAMES.get(code, code),
                "price": None, "error": "pykrx 없음"
            })
    except Exception as e:
        print(f"[pykrx 전체] 오류: {e}")

    return result


# ─── HTML 생성 ───────────────────────────────────────────────────────────────
def fmt(val, decimals=2, suffix="") -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}{suffix}"


def color_class(val, low_bad=True) -> str:
    """단순 색상 클래스. low_bad=True이면 낮을수록 빨강"""
    if val is None:
        return "neutral"
    return "green" if val else "red"


def score_color(score: int) -> str:
    if score >= 70: return "#4caf50"
    if score >= 50: return "#ffeb3b"
    if score >= 30: return "#ff9800"
    return "#f44336"


def signal_badge(signal: str) -> str:
    colors = {"매수": "#4caf50", "중립": "#ffeb3b", "매도": "#f44336"}
    c = colors.get(signal, "#9e9e9e")
    return f'<span style="background:{c};color:#000;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600">{signal}</span>'


def change_span(pct) -> str:
    if pct is None:
        return "N/A"
    color = "#4caf50" if pct >= 0 else "#f44336"
    arrow = "▲" if pct >= 0 else "▼"
    return f'<span style="color:{color}">{arrow}{abs(pct):.2f}%</span>'


def generate_html(macro: dict, scores: dict, sectors: dict, us_scalp: list, kr_scalp: list) -> str:
    now_kst = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M KST")
    short_s = scores["short"]
    mid_s   = scores["mid"]
    long_s  = scores["long"]

    # ── 탭 1: 거시경제 카드 생성 ──────────────────────────────────────────────
    def macro_card(title, value, sub="", signal="neutral"):
        sig_colors = {"green": "#4caf50", "yellow": "#ffeb3b", "orange": "#ff9800", "red": "#f44336", "neutral": "#9e9e9e"}
        border_c = sig_colors.get(signal, "#9e9e9e")
        return f"""
        <div class="card" style="border-left:4px solid {border_c}">
          <div class="card-title">{title}</div>
          <div class="card-value">{value}</div>
          <div class="card-sub">{sub}</div>
        </div>"""

    vix     = macro.get("vix")
    fg      = macro.get("fear_greed")
    fg_s    = fg["score"] if fg else None
    fg_r    = fg["rating"] if fg else "N/A"
    pc      = macro.get("put_call")
    ys      = macro.get("yield_spread")
    hy      = macro.get("hy_spread")
    r10     = macro.get("rate_10y")
    r2      = macro.get("rate_2y")
    cpi     = macro.get("cpi")
    pce     = macro.get("pce")
    unemp   = macro.get("unemployment")
    jobless = macro.get("jobless")
    nfp     = macro.get("nfp")
    ism     = macro.get("ism_pmi")
    m2      = macro.get("m2")
    fedbal  = macro.get("fed_balance")
    res     = macro.get("reserves")
    rrp     = macro.get("rrp")
    tga     = macro.get("tga")
    stlfsi  = macro.get("stlfsi")
    dxy     = macro.get("dxy")
    sox     = macro.get("sox")
    sox_chg = macro.get("sox_change")
    copper  = macro.get("copper")
    cop_chg = macro.get("copper_change")
    krw     = macro.get("krw_usd")
    krw_chg = macro.get("krw_change")
    poly    = macro.get("polymarket_fomc")

    def vix_sig(v): return "green" if v and v < 20 else ("yellow" if v and v < 30 else "red")
    def fg_sig(v):  return "green" if v and v <= 25 else ("yellow" if v and v <= 60 else "red")
    def pc_sig(v):  return "green" if v and v >= 1.2 else ("yellow" if v and v >= 0.7 else "red")
    def ys_sig(v):  return "green" if v and v >= 0.5 else ("yellow" if v and v >= 0 else "red")
    def hy_sig(v):  return "green" if v and v <= 4 else ("yellow" if v and v <= 6 else "red")
    def stl_sig(v): return "green" if v and v < 0 else ("yellow" if v and v < 1 else "red")

    nfp_k = (nfp / 1000) if nfp else None  # 천명

    cards_macro = ""
    cards_macro += macro_card("VIX 공포지수",    fmt(vix,2), "20↓매수 30↑매도", vix_sig(vix))
    cards_macro += macro_card("공포탐욕지수",     fmt(fg_s,1), fg_r, fg_sig(fg_s))
    cards_macro += macro_card("풋/콜 비율",       fmt(pc,3), "1.2↑매수 0.7↓매도", pc_sig(pc))
    cards_macro += macro_card("장단기 금리차",    fmt(ys,2,"%"), "10Y-2Y", ys_sig(ys))
    cards_macro += macro_card("하이일드 스프레드",fmt(hy,2,"%"), "4%↓매수 6%↑매도", hy_sig(hy))
    cards_macro += macro_card("달러인덱스(DXY)",  fmt(dxy,3), macro.get("dxy_trend",""), "neutral")
    cards_macro += macro_card("미10년물 금리",    fmt(r10,2,"%"), "", "neutral")
    cards_macro += macro_card("미2년물 금리",     fmt(r2,2,"%"), "", "neutral")
    cards_macro += macro_card("CPI",              fmt(cpi,1), macro.get("cpi_trend",""), "neutral")
    cards_macro += macro_card("PCE",              fmt(pce,1), "", "neutral")
    cards_macro += macro_card("실업률",           fmt(unemp,1,"%"), macro.get("unemployment_trend",""), "neutral")
    cards_macro += macro_card("신규 실업수당",    fmt(jobless,0,"천"), "", "neutral")
    cards_macro += macro_card("NFP 비농업고용",   fmt(nfp_k,0,"만명"), "", "neutral")
    cards_macro += macro_card("ISM PMI",          fmt(ism,1), "52↑매수 50↓매도", "green" if ism and ism>=52 else ("yellow" if ism and ism>=50 else "red"))
    cards_macro += macro_card("M2 통화량",        fmt(m2,0,"억$") if m2 else "N/A", macro.get("m2_trend",""), "neutral")
    cards_macro += macro_card("연준 대차대조표",  fmt(fedbal,0,"억$") if fedbal else "N/A", macro.get("fed_balance_trend",""), "neutral")
    cards_macro += macro_card("연준 지급준비금",  fmt(res,0,"억$") if res else "N/A", "", "neutral")
    cards_macro += macro_card("역레포 잔액(RRP)", fmt(rrp,0,"억$") if rrp else "N/A", "", "neutral")
    cards_macro += macro_card("TGA 잔액",         fmt(tga,0,"억$") if tga else "N/A", "", "neutral")
    cards_macro += macro_card("금융스트레스지수", fmt(stlfsi,3), "0↓안정 1↑위험", stl_sig(stlfsi))
    cards_macro += macro_card("SOX 반도체지수",   fmt(sox,2), change_span(sox_chg), "neutral")
    cards_macro += macro_card("구리 선물",        fmt(copper,4,"$"), change_span(cop_chg), "neutral")
    cards_macro += macro_card("원/달러 환율",     fmt(krw,2,"원"), change_span(krw_chg), "neutral")

    poly_text = "N/A"
    if poly:
        q = poly.get("question","FOMC")[:30]
        y = poly.get("yes_price","N/A")
        poly_text = f"{y} ({q})"
    cards_macro += macro_card("Polymarket FOMC", poly_text, "", "neutral")

    # ── 탭 2: 섹터 종목 카드 ──────────────────────────────────────────────────
    def sector_stock_card(item: dict, short_warn: bool) -> str:
        tk    = item.get("ticker","")
        price = item.get("price")
        chg   = item.get("change_pct")
        rsi   = item.get("rsi")
        macd  = item.get("macd")
        sig   = item.get("macd_signal")
        bb_pos= item.get("bb_position")
        cross = item.get("cross","N/A")
        vs200 = item.get("vs_ma200","N/A")
        tsig  = item.get("tech_signal","중립")
        buy   = item.get("buy_line")
        sell  = item.get("sell_line")
        stop  = item.get("stop_loss")
        ne    = item.get("next_earnings","")
        soon  = item.get("earnings_soon", False)
        eps_a = item.get("eps_actual")
        eps_e = item.get("eps_estimate")
        eps_s = item.get("eps_surprise_pct")
        tgt   = item.get("target_price")

        rsi_color = "#4caf50" if rsi and rsi<=30 else ("#f44336" if rsi and rsi>=70 else "#ccc")
        macd_sig_txt = "상승" if (macd and sig and macd>sig) else "하락"
        bb_txt = f"{fmt(bb_pos,1)}%" if bb_pos is not None else "N/A"
        cross_color = "#4caf50" if cross=="골든크로스" else "#f44336"
        vs200_color = "#4caf50" if vs200=="위" else "#f44336"

        earn_str = f'<span style="color:#f44336;font-weight:700">{ne} ⚠️실적임박</span>' if soon else (ne or "N/A")
        eps_str  = f"{fmt(eps_a,2)} / 예상 {fmt(eps_e,2)}"
        if eps_s is not None:
            surp_c = "#4caf50" if eps_s>=0 else "#f44336"
            eps_str += f' <span style="color:{surp_c}">(서프라이즈 {eps_s:+.1f}%)</span>'

        warn = '<div style="background:#b71c1c;color:#fff;padding:6px;border-radius:4px;margin-top:8px;font-size:12px">⚠️ 거시 위험 높음 — 포지션 크기 축소 권장</div>' if short_warn else ""

        return f"""
        <div class="stock-card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <span style="font-size:18px;font-weight:700">{tk}</span>
            {signal_badge(tsig)}
          </div>
          <div style="font-size:22px;font-weight:700">${fmt(price,2)}</div>
          <div style="margin:4px 0">{change_span(chg)}</div>
          <hr style="border-color:#333;margin:8px 0">
          <div class="stock-row"><span title="{TOOLTIP_RSI}" style="cursor:help;border-bottom:1px dotted #666">RSI(14) ℹ</span><span style="color:{rsi_color};font-weight:600">{fmt(rsi,1)}</span></div>
          <div class="stock-row"><span title="{TOOLTIP_MACD}" style="cursor:help;border-bottom:1px dotted #666">MACD ℹ</span><span>{macd_sig_txt}</span></div>
          <div class="stock-row"><span title="{TOOLTIP_BB}" style="cursor:help;border-bottom:1px dotted #666">볼린저밴드 위치 ℹ</span><span>{bb_txt}</span></div>
          <div class="stock-row"><span title="{TOOLTIP_MA}" style="cursor:help;border-bottom:1px dotted #666">MA 크로스 ℹ</span><span style="color:{cross_color}">{cross}</span></div>
          <div class="stock-row"><span title="{TOOLTIP_MA}" style="cursor:help;border-bottom:1px dotted #666">200MA 대비 ℹ</span><span style="color:{vs200_color}">{vs200}</span></div>
          <hr style="border-color:#333;margin:8px 0">
          <div class="stock-row"><span>매수선</span><span style="color:#4caf50">${fmt(buy,2)}</span></div>
          <div class="stock-row"><span>매도선</span><span style="color:#f44336">${fmt(sell,2)}</span></div>
          <div class="stock-row"><span>손절선</span><span style="color:#ff9800">${fmt(stop,2)}</span></div>
          <hr style="border-color:#333;margin:8px 0">
          <div class="stock-row"><span>다음 실적</span><span>{earn_str}</span></div>
          <div class="stock-row small-text"><span>EPS</span><span>{eps_str}</span></div>
          <div class="stock-row"><span>목표주가</span><span>${fmt(tgt,2) if tgt else "N/A"}</span></div>
          {warn}
        </div>"""

    html_sectors = ""
    for sector_name, items in sectors.items():
        html_sectors += f'<div class="sector-title">{sector_name}</div><div class="card-grid">'
        for item in items:
            html_sectors += sector_stock_card(item, short_warn=short_s["score"] <= 30)
        html_sectors += "</div>"

    TOOLTIP_RSI  = "과매수/과매도 측정 (0~100). 30↓ 과매도(매수신호), 70↑ 과매수(매도신호). 14일 기준"
    TOOLTIP_MACD = "단기(12일) - 장기(26일) 이동평균 차이. Signal선(9일) 상향 돌파 시 매수신호"
    TOOLTIP_BB   = "20일 평균 ± 2표준편차. 하단 터치는 통계적 과매도, 상단 터치는 과매수 구간"
    TOOLTIP_MA   = "50일선이 200일선을 상향 돌파 = 골든크로스(강세), 하향 = 데드크로스(약세)"

    # ── 탭 3: 미국 단타 카드 ─────────────────────────────────────────────────
    def us_scalp_card(item: dict) -> str:
        tk     = item.get("ticker","")
        kr_name = US_SCALP_NAMES.get(tk, "")
        price  = item.get("price")
        chg    = item.get("change_pct")
        rsi    = item.get("rsi")
        macd   = item.get("macd")
        sig    = item.get("macd_signal")
        bb_pos = item.get("bb_position")
        bb_low = item.get("bb_lower")
        vs200  = item.get("vs_ma200","N/A")
        vol    = item.get("vol_ratio")
        esig   = item.get("entry_signals",0)
        buy    = item.get("buy_line")
        stop   = item.get("stop_loss")

        rsi_color = "#4caf50" if rsi and rsi<=30 else ("#f44336" if rsi and rsi>=70 else "#ccc")
        rsi_label = " 🟢과매도" if rsi and rsi<=30 else (" 🔴과매수" if rsi and rsi>=70 else "")
        macd_gc   = macd is not None and sig is not None and macd > sig
        bb_touch  = bb_low is not None and price is not None and price <= bb_low * 1.01
        vol_surge = vol is not None and vol >= 2.0
        vs200_color = "#4caf50" if vs200=="위" else "#f44336"

        badge = ""
        if esig >= 3:
            badge = '<div style="background:#1565c0;color:#fff;padding:6px 12px;border-radius:6px;text-align:center;font-weight:700;margin-bottom:8px">🎯 진입 신호</div>'

        return f"""
        <div class="stock-card">
          {badge}
          <div style="font-size:20px;font-weight:700;margin-bottom:2px">{tk}</div>
          <div style="color:#9e9e9e;font-size:13px;margin-bottom:4px">{kr_name}</div>
          <div style="font-size:22px;font-weight:700">${fmt(price,2)}</div>
          <div style="margin:4px 0">{change_span(chg)}</div>
          <hr style="border-color:#333;margin:8px 0">
          <div class="stock-row"><span title="{TOOLTIP_RSI}" style="cursor:help;border-bottom:1px dotted #666">RSI(14) ℹ</span><span style="color:{rsi_color};font-weight:600">{fmt(rsi,1)}{rsi_label}</span></div>
          <div class="stock-row"><span title="{TOOLTIP_MACD}" style="cursor:help;border-bottom:1px dotted #666">MACD 골든크로스 ℹ</span><span style="color:{'#4caf50' if macd_gc else '#f44336'}">{'✓ 발생' if macd_gc else '미발생'}</span></div>
          <div class="stock-row"><span title="{TOOLTIP_BB}" style="cursor:help;border-bottom:1px dotted #666">볼린저 하단 터치 ℹ</span><span style="color:{'#4caf50' if bb_touch else '#ccc'}">{'✓ 터치' if bb_touch else '미터치'}</span></div>
          <div class="stock-row"><span title="{TOOLTIP_MA}" style="cursor:help;border-bottom:1px dotted #666">200MA 대비 ℹ</span><span style="color:{vs200_color}">{vs200}</span></div>
          <div class="stock-row"><span>거래량 비율</span><span style="color:{'#ff9800' if vol_surge else '#ccc'}">{fmt(vol,2,"x")}{' 🔥급증' if vol_surge else ''}</span></div>
          <hr style="border-color:#333;margin:8px 0">
          <div class="stock-row"><span>매수선</span><span style="color:#4caf50">${fmt(buy,2)}</span></div>
          <div class="stock-row"><span>손절선</span><span style="color:#ff9800">${fmt(stop,2)}</span></div>
        </div>"""

    html_us_scalp = '<div class="card-grid">'
    for item in us_scalp:
        html_us_scalp += us_scalp_card(item)
    html_us_scalp += "</div>"

    # ── 탭 4: 한국 단타 카드 ─────────────────────────────────────────────────
    def kr_scalp_card(item: dict) -> str:
        tk    = item.get("ticker","")
        name  = item.get("name", tk)
        price = item.get("price")
        chg   = item.get("change_pct")
        rsi   = item.get("rsi")
        macd  = item.get("macd")
        sig   = item.get("macd_signal")
        bb_pos= item.get("bb_position")
        bb_low= item.get("bb_lower")
        vs200 = item.get("vs_ma200","N/A")
        vol   = item.get("vol_ratio")
        inst  = item.get("inst_buy_3d", False)
        for_  = item.get("for_buy_3d",  False)
        strong= item.get("strong_entry", False)
        buy   = item.get("buy_line")
        stop  = item.get("stop_loss")

        rsi_color = "#4caf50" if rsi and rsi<=30 else ("#f44336" if rsi and rsi>=70 else "#ccc")
        rsi_label = " 🟢과매도" if rsi and rsi<=30 else (" 🔴과매수" if rsi and rsi>=70 else "")
        macd_gc  = macd is not None and sig is not None and macd > sig
        bb_touch = bb_low is not None and price is not None and price <= bb_low * 1.01
        vol_surge= vol is not None and vol >= 2.0
        vs200_color = "#4caf50" if vs200=="위" else "#f44336"

        badge = ""
        if strong:
            badge = '<div style="background:#6a1b9a;color:#fff;padding:6px 12px;border-radius:6px;text-align:center;font-weight:700;margin-bottom:8px">🎯 강한 진입 신호</div>'

        price_fmt = f"{price:,.0f}원" if price else "N/A"
        buy_fmt   = f"{buy:,.0f}원"   if buy   else "N/A"
        stop_fmt  = f"{stop:,.0f}원"  if stop  else "N/A"

        return f"""
        <div class="stock-card">
          {badge}
          <div style="font-size:18px;font-weight:700;margin-bottom:2px">{name}</div>
          <div style="color:#888;font-size:13px;margin-bottom:4px">{tk}</div>
          <div style="font-size:22px;font-weight:700">{price_fmt}</div>
          <div style="margin:4px 0">{change_span(chg)}</div>
          <hr style="border-color:#333;margin:8px 0">
          <div class="stock-row"><span title="{TOOLTIP_RSI}" style="cursor:help;border-bottom:1px dotted #666">RSI(14) ℹ</span><span style="color:{rsi_color};font-weight:600">{fmt(rsi,1)}{rsi_label}</span></div>
          <div class="stock-row"><span title="{TOOLTIP_MACD}" style="cursor:help;border-bottom:1px dotted #666">MACD 골든크로스 ℹ</span><span style="color:{'#4caf50' if macd_gc else '#f44336'}">{'✓ 발생' if macd_gc else '미발생'}</span></div>
          <div class="stock-row"><span title="{TOOLTIP_BB}" style="cursor:help;border-bottom:1px dotted #666">볼린저 하단 터치 ℹ</span><span style="color:{'#4caf50' if bb_touch else '#ccc'}">{'✓ 터치' if bb_touch else '미터치'}</span></div>
          <div class="stock-row"><span title="{TOOLTIP_MA}" style="cursor:help;border-bottom:1px dotted #666">200MA 대비 ℹ</span><span style="color:{vs200_color}">{vs200}</span></div>
          <div class="stock-row"><span>거래량 비율</span><span style="color:{'#ff9800' if vol_surge else '#ccc'}">{fmt(vol,2,"x")}{' 🔥급증' if vol_surge else ''}</span></div>
          <div class="stock-row"><span>기관 3일 순매수</span><span style="color:{'#4caf50' if inst else '#ccc'}">{'✓ 3일 연속' if inst else '아니오'}</span></div>
          <div class="stock-row"><span>외국인 3일 순매수</span><span style="color:{'#4caf50' if for_ else '#ccc'}">{'✓ 3일 연속' if for_ else '아니오'}</span></div>
          <hr style="border-color:#333;margin:8px 0">
          <div class="stock-row"><span>매수선</span><span style="color:#4caf50">{buy_fmt}</span></div>
          <div class="stock-row"><span>손절선</span><span style="color:#ff9800">{stop_fmt}</span></div>
        </div>"""

    html_kr_scalp = '<div class="card-grid">'
    for item in kr_scalp:
        html_kr_scalp += kr_scalp_card(item)
    html_kr_scalp += "</div>"

    # ── 전체 HTML 조립 ────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>미국 주식 모니터링 대시보드</title>
<style>
  :root {{
    --bg: #121212;
    --surface: #1e1e1e;
    --surface2: #252525;
    --text: #e0e0e0;
    --text-sub: #9e9e9e;
    --border: #333;
    --accent: #90caf9;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    min-height: 100vh;
  }}

  /* ── 탭 (데스크탑/태블릿: 상단 고정) ── */
  .tab-nav {{
    display: flex;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 100;
  }}
  .tab-btn {{
    flex: 1;
    padding: 14px 8px;
    background: none;
    border: none;
    color: var(--text-sub);
    font-size: 14px;
    cursor: pointer;
    border-bottom: 3px solid transparent;
    transition: all 0.2s;
    white-space: nowrap;
  }}
  .tab-btn.active {{
    color: var(--accent);
    border-bottom-color: var(--accent);
  }}
  .tab-btn:hover {{ color: var(--text); }}

  .tab-panel {{ display: none; padding: 16px; }}
  .tab-panel.active {{ display: block; }}

  /* ── 점수 카드 ── */
  .score-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }}
  .score-card {{
    background: var(--surface);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
  }}
  .score-title {{ font-size: 13px; color: var(--text-sub); margin-bottom: 8px; }}
  .score-value {{ font-size: 48px; font-weight: 700; line-height: 1; }}
  .score-label {{ font-size: 14px; margin-top: 8px; font-weight: 600; }}

  /* ── 거시 카드 그리드 ── */
  .card-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 24px;
  }}
  .card {{
    background: var(--surface);
    border-radius: 10px;
    padding: 14px;
    border-left: 4px solid #9e9e9e;
  }}
  .card-title {{ font-size: 12px; color: var(--text-sub); margin-bottom: 4px; }}
  .card-value {{ font-size: 20px; font-weight: 700; }}
  .card-sub   {{ font-size: 12px; color: var(--text-sub); margin-top: 4px; }}

  /* ── 종목 카드 ── */
  .stock-card {{
    background: var(--surface);
    border-radius: 10px;
    padding: 16px;
  }}
  .stock-row {{
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    padding: 3px 0;
    border-bottom: 1px solid var(--border);
  }}
  .stock-row:last-child {{ border-bottom: none; }}
  .stock-row.small-text {{ font-size: 11px; }}
  .sector-title {{
    font-size: 18px;
    font-weight: 700;
    color: var(--accent);
    margin: 20px 0 12px;
    padding-bottom: 6px;
    border-bottom: 2px solid var(--border);
  }}

  /* ── 업데이트 시각 ── */
  .update-bar {{
    background: var(--surface2);
    padding: 8px 16px;
    font-size: 13px;
    color: var(--text-sub);
    text-align: right;
    border-bottom: 1px solid var(--border);
  }}

  /* ── 태블릿 (768~1024px) ── */
  @media (max-width: 1024px) {{
    .card-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .score-grid {{ grid-template-columns: repeat(3, 1fr); }}
  }}
  @media (max-width: 900px) {{
    .card-grid {{ grid-template-columns: repeat(2, 1fr); }}
  }}

  /* ── 모바일 (≤768px) ── */
  @media (max-width: 768px) {{
    .tab-nav {{
      position: fixed;
      bottom: 0;
      top: auto;
      left: 0;
      right: 0;
      border-top: 1px solid var(--border);
      border-bottom: none;
      z-index: 200;
    }}
    .tab-btn {{
      padding: 12px 4px 8px;
      font-size: 11px;
      min-height: 56px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }}
    body {{ padding-bottom: 64px; }}
    .tab-panel {{ padding: 12px; }}
    .card-grid {{ grid-template-columns: 1fr; }}
    .score-grid {{ grid-template-columns: 1fr; gap: 12px; }}
    .score-value {{ font-size: 40px; }}
    .card-value {{ font-size: 18px; }}
    * {{ font-size: 14px; }}
    .tab-btn, .stock-row, .card-sub {{ min-height: 0; }}
  }}
</style>
</head>
<body>

<div class="update-bar">마지막 업데이트: {now_kst}</div>

<div class="tab-nav">
  <button class="tab-btn active" onclick="switchTab(0)">📊 거시경제</button>
  <button class="tab-btn" onclick="switchTab(1)">🏢 섹터 종목</button>
  <button class="tab-btn" onclick="switchTab(2)">⚡ 단타 (미국)</button>
  <button class="tab-btn" onclick="switchTab(3)">🇰🇷 단타 (한국)</button>
</div>

<!-- 탭 1: 거시경제 개요 -->
<div class="tab-panel active" id="tab0">
  <div class="score-grid">
    <div class="score-card">
      <div class="score-title">단기 점수 (심리·모멘텀)</div>
      <div class="score-value" style="color:{score_color(short_s['score'])}">{short_s['score']}</div>
      <div class="score-label" style="color:{score_color(short_s['score'])}">{short_s['label']}</div>
    </div>
    <div class="score-card">
      <div class="score-title">중기 점수 (펀더멘털·정책)</div>
      <div class="score-value" style="color:{score_color(mid_s['score'])}">{mid_s['score']}</div>
      <div class="score-label" style="color:{score_color(mid_s['score'])}">{mid_s['label']}</div>
    </div>
    <div class="score-card">
      <div class="score-title">장기 점수 (유동성·구조)</div>
      <div class="score-value" style="color:{score_color(long_s['score'])}">{long_s['score']}</div>
      <div class="score-label" style="color:{score_color(long_s['score'])}">{long_s['label']}</div>
    </div>
  </div>

  <div class="card-grid">
    {cards_macro}
  </div>
</div>

<!-- 탭 2: 섹터별 관심종목 -->
<div class="tab-panel" id="tab1">
  {html_sectors}
</div>

<!-- 탭 3: 단타 종목 (미국) -->
<div class="tab-panel" id="tab2">
  <p style="color:var(--text-sub);font-size:13px;margin-bottom:16px">
    ※ 기술적 신호 3개 이상 충족 시 🎯 진입 신호 배지 표시
  </p>
  {html_us_scalp}
</div>

<!-- 탭 4: 단타 종목 (한국) -->
<div class="tab-panel" id="tab3">
  <p style="color:var(--text-sub);font-size:13px;margin-bottom:16px">
    ※ 기술적 신호 3개 이상 + 기관/외국인 매수 동시 충족 시 🎯 강한 진입 신호 배지 표시
  </p>
  {html_kr_scalp}
</div>

<script>
function switchTab(idx) {{
  document.querySelectorAll('.tab-panel').forEach((p, i) => {{
    p.classList.toggle('active', i === idx);
  }});
  document.querySelectorAll('.tab-btn').forEach((b, i) => {{
    b.classList.toggle('active', i === idx);
  }});
}}
</script>
</body>
</html>"""
    return html


# ─── 메인 ────────────────────────────────────────────────────────────────────
def main():
    print(f"=== 대시보드 생성 시작 ({TODAY_STR}) ===")

    print("[1/4] 거시지표 수집 중...")
    macro = collect_macro()

    print("[2/4] 점수 계산 중...")
    scores = calc_scores(macro)
    print(f"  단기: {scores['short']['score']} | 중기: {scores['mid']['score']} | 장기: {scores['long']['score']}")

    print("[3/4] 종목 데이터 수집 중...")
    sectors  = collect_us_sectors(scores["short"]["score"])
    us_scalp = collect_us_scalp()
    kr_scalp = collect_kr_scalp()

    print("[4/4] HTML 생성 중...")
    html = generate_html(macro, scores, sectors, us_scalp, kr_scalp)

    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"=== 완료: {out_path} ===")


if __name__ == "__main__":
    main()
