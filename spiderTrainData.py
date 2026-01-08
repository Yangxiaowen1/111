from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests


HEADERS = {
    "accept": "*/*",
    "accept-language": "zh-CN,zh;q=0.9",
    "content-type": "application/json",
    "cookieorigin": "https://you.ctrip.com",
    "origin": "https://you.ctrip.com",
    "priority": "u=1, i",
    "referer": "https://you.ctrip.com/",
    "sec-ch-ua": "\"Chromium\";v=\"142\", \"Google Chrome\";v=\"142\", \"Not_A Brand\";v=\"99\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    "x-ctx-currency": "CNY",
    "x-ctx-locale": "zh-CN",
    "x-ctx-ubt-pageid": "10650142842",
    "x-ctx-ubt-pvid": "13",
    "x-ctx-ubt-sid": "2",
    "x-ctx-ubt-vid": "1763963515081.9998JVbalV8i",
    "x-ctx-wclient-req": "35fca3817524cb62d32ec0a3a155ee4a",
    "cookie": "GUID=09031146412762781768; UBT_VID=1763963515081.9998JVbalV8i; MKT_CKID=1763963515148.mk89s.q3dw; _RGUID=33dff6cd-dfab-4377-8c14-08cf30fa6281; nfes_isSupportWebP=1; Hm_lvt_a8d6737197d542432f4ff4abc6e06384=1763963515,1764138593; HMACCOUNT=D88074E5CC8BEFC6; Hm_lpvt_a8d6737197d542432f4ff4abc6e06384=1764138608; Session=smartlinkcode=U130026&smartlinklanguage=zh&SmartLinkKeyWord=&SmartLinkQuary=&SmartLinkHost=; Union=AllianceID=4897&SID=130026&OUID=&createtime=1764138608&Expires=1764743407509; _ga=GA1.1.661598682.1764138609; MKT_Pagesource=PC; _ga_9BZF483VNQ=GS2.1.s1764138608$o1$g0$t1764138612$j56$l0$h0; _ga_5DVRDQD429=GS2.1.s1764138608$o1$g0$t1764138612$j56$l0$h1147704621; _ga_B77BES1Z8Z=GS2.1.s1764138608$o1$g0$t1764138612$j56$l0$h0; StartCity_Pkg=PkgStartCity=2; ibulanguage=ZH-CN; ibulocale=zh_cn; cookiePricesDisplayed=CNY; _bfa=1.1763963515081.9998JVbalV8i.1.1764139383721.1764139388277.2.18.10650142842; _jzqco=%7C%7C%7C%7C1764137772256%7C1.1459682470.1764137772119.1764139383879.1764139388452.1764139383879.1764139388452.undefined.0.0.18.18",
}

URL = "https://m.ctrip.com/restapi/soa2/18109/json/getAttractionList"
PARAMS = {
    "_fxpcqlniredt": "09031146412762781768",
    "x-traceID": "09031146412762781768-1764138739146-634505",
}


def build_payload(district_id: int, page_index: int, count: int) -> str:
    """构造请求体，按需设置城市和分页参数。"""
    payload = {
        "head": {
            "cid": "09031146412762781768",
            "ctok": "",
            "cver": "1.0",
            "lang": "01",
            "sid": "8888",
            "syscode": "999",
            "auth": "",
            "xsid": "",
            "extension": [],
        },
        "scene": "online",
        "districtId": district_id,
        "index": page_index,
        "sortType": 1,
        "count": count,
        "filter": {"filterItems": []},
        "returnModuleType": "product",
    }
    return json.dumps(payload, separators=(",", ":"))


def fetch_attraction_page(district_id: int, page_index: int, count: int) -> List[Dict[str, Any]]:
    """拉取指定城市某一页的景点数据。"""
    response = requests.post(
        URL,
        headers=HEADERS,
        params=PARAMS,
        data=build_payload(district_id, page_index, count),
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("attractionList", [])


def _stringify(value: Any) -> Any:
    """将复杂数据类型转为 JSON 字符串，保证 CSV 可落地所有字段。"""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return json.dumps(list(value), ensure_ascii=False)
    return value


def parse_card(card: Dict[str, Any]) -> Dict[str, Any]:
    """提取单条景点卡片的全部字段信息。"""
    inner = card.get("card") if isinstance(card, dict) and "card" in card else card
    normalized: Dict[str, Any] = {}
    for key, value in (inner or {}).items():
        normalized[key] = _stringify(value)
    return normalized


def write_csv(records: List[Dict[str, Any]], output_path: Path) -> None:
    """将数据累积写入 CSV，已存在则读取旧内容并合并后整体写回。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    historical: List[Dict[str, Any]] = []
    if output_path.exists():
        with output_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            historical = list(reader)

    merged = historical + records
    if not merged:
        return

    fieldnames: List[str] = sorted({field for record in merged for field in record.keys()})
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)


def crawl_and_export(
    district_id: int,
    district_name: str,
    pages: int,
    count: int,
    output: Path,
    delay: float,
) -> None:
    """批量爬取多页数据并保存到 CSV。"""
    all_cards: List[Dict[str, Any]] = []
    for page in range(1, pages + 1):
        try:
            cards = fetch_attraction_page(district_id, page, count)
        except requests.RequestException as exc:
            print(f"第 {page} 页请求失败：{exc}")
            break
        if not cards:
            print(f"第 {page} 页无数据，提前结束。")
            break
        for card in cards:
            record = parse_card(card)
            record["crawlDistrictId"] = district_id
            record["crawlDistrictName"] = district_name or record.get("districtName")
            all_cards.append(record)
        print(f"已获取第 {page} 页，累计 {len(all_cards)} 条记录。")
        if len(cards) < count:
            print("当前页数量不足 count，可能已经到最后一页。")
            break
        time.sleep(max(delay, 0))

    if all_cards:
        write_csv(all_cards, output)
        print(f"成功写入 {len(all_cards)} 条数据到 {output}.")
    else:
        print("未获取到任何数据，未生成 CSV。")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="携程景点列表爬虫")
    parser.add_argument("--district", type=int, default=21, help="城市 districtId，例如 2 表示上海")
    parser.add_argument("--city", type=str, default="厦门", help="城市中文名称，便于后续分析")
    parser.add_argument("--pages", type=int, default=100, help="需要爬取的页数，默认 1")
    parser.add_argument("--count", type=int, default=10, help="每页条数，接口默认 10，最大 10")
    parser.add_argument("--output", type=str, default="data/attractions.csv", help="CSV 输出路径")
    parser.add_argument("--delay", type=float, default=0.5, help="连续请求之间的延时（秒）")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    crawl_and_export(
        district_id=args.district,
        district_name=args.city,
        pages=max(1, args.pages),
        count=max(1, min(10, args.count)),
        output=Path(args.output),
        delay=args.delay,
    )