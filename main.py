import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import config
from analyzer import analyze_vacancies_batch
from fetcher import fetch_vacancies_from_api, load_vacancies_from_file
from scheduler import run_scheduled


def save_report(output_path: str, items: List[Dict[str, Any]], source_mode: str, api_url: str) -> None:
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": source_mode,
        "source": api_url,
        "total_vacancies": len(items),
        "vacancies": items,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_live_once(api_url: str, max_vacancies: int, page_limit: int, output_path: str) -> None:
    print(f"[LIVE] Загрузка вакансий из {api_url} (max={max_vacancies}, page_limit={page_limit})...")
    vacancies, error = fetch_vacancies_from_api(
        api_url=api_url,
        max_vacancies=max_vacancies,
        page_limit=page_limit,
        min_pages=2,
    )

    if error:
        print(f"[ERROR] {error}")
        if os.path.exists(output_path):
            print(f"[INFO] Сохранён последний успешный отчёт: {output_path}")
        else:
            print("[INFO] Предыдущий успешный отчёт отсутствует, новый отчёт не перезаписан.")
        return

    report_items = analyze_vacancies_batch(
        vacancies=vacancies,
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        model=config.LLM_MODEL,
    )
    save_report(output_path=output_path, items=report_items, source_mode="live", api_url=api_url)
    print(f"[OK] Обработано вакансий: {len(report_items)}. Отчёт сохранён в {output_path}")


def run_check_mode(input_path: str, output_path: str) -> None:
    print(f"[CHECK] Загрузка локального JSON из {input_path}...")
    vacancies, error = load_vacancies_from_file(input_path)
    if error:
        print(f"[ERROR] {error}")
        return

    report_items = analyze_vacancies_batch(
        vacancies=vacancies,
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        model=config.LLM_MODEL,
    )
    save_report(output_path=output_path, items=report_items, source_mode="check", api_url=input_path)
    print(f"[OK] Проверочный режим завершён. Вакансий в отчёте: {len(report_items)}. Файл: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Прозрачная вакансия — анализатор условий труда")
    parser.add_argument(
        "--mode",
        choices=["live", "schedule", "check"],
        default="live",
        help="live — разовый запуск с API, schedule — запуск по расписанию, check — локальный JSON",
    )
    parser.add_argument("--api-url", default=config.API_URL, help="URL API источника")
    parser.add_argument("--max-vacancies", type=int, default=config.MAX_VACANCIES, help="Максимум вакансий")
    parser.add_argument("--page-limit", type=int, default=config.PAGE_LIMIT, help="Размер страницы API")
    parser.add_argument("--interval", type=int, default=config.SCHEDULE_INTERVAL_MINUTES, help="Интервал в минутах")
    parser.add_argument("--input", dest="input_file", default="sample_data/test_vacancies.json", help="Путь к JSON для check")
    parser.add_argument("--output", dest="output_file", default=config.OUTPUT_PATH, help="Путь к выходному отчёту JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "check":
        run_check_mode(input_path=args.input_file, output_path=args.output_file)
    elif args.mode == "schedule":
        run_scheduled(
job_fn=lambda: run_live_once(
                api_url=args.api_url,
                max_vacancies=args.max_vacancies,
                page_limit=args.page_limit,
                output_path=args.output_file,
            ),
            interval_minutes=args.interval,
            run_immediately=True,
        )
    else:
        run_live_once(
            api_url=args.api_url,
            max_vacancies=args.max_vacancies,
            page_limit=args.page_limit,
            output_path=args.output_file,
        )


if __name__ == "__main__":
    main()
