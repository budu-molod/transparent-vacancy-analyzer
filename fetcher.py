import json
import os
from typing import Any, Dict, List, Optional, Tuple
import requests


def extract_vacancies_from_payload(payload: Any) -> List[Dict[str, Any]]:
    """Извлекает список словарей вакансий из разных вариантов структуры ответа API."""
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, dict):
            vacancies = results.get("vacancies")
            if isinstance(vacancies, list):
                items = vacancies
            else:
                items = []
        elif isinstance(results, list):
            items = results
        elif isinstance(payload.get("vacancies"), list):
            items = payload["vacancies"]
        else:
            items = []
    else:
        items = []

    normalized: List[Dict[str, Any]] = []
    for entry in items:
        if isinstance(entry, dict) and isinstance(entry.get("vacancy"), dict):
            normalized.append(entry["vacancy"])
        elif isinstance(entry, dict):
            normalized.append(entry)
    return normalized


def fetch_vacancies_from_api(
    api_url: str,
    max_vacancies: int = 20,
    page_limit: int = 10,
    min_pages: int = 2,
    timeout: int = 20,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Загружает вакансии из API с поддержкой пагинации.
    Старается пройти минимум min_pages страниц, если позволяет объём и лимит.
    Возвращает (список_вакансий, сообщение_об_ошибке_или_None).
    """
    collected: List[Dict[str, Any]] = []
    seen_ids = set()
    page = 0

    try:
        while len(collected) < max_vacancies:
            params = {
                "limit": page_limit,
                "offset": page * page_limit,
            }
            resp = requests.get(api_url, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            batch = extract_vacancies_from_payload(data)
            if not batch:
                break

            added_in_page = 0
            for vac in batch:
                vac_id = str(vac.get("id") or vac.get("vacancy_id") or len(collected))
                if vac_id in seen_ids:
                    continue
                seen_ids.add(vac_id)
                collected.append(vac)
                added_in_page += 1
                if len(collected) >= max_vacancies:
                    break

            page += 1
            if added_in_page == 0:
                break
            # Если уже прошли минимум страниц и набрали нужное число — выходим;
            # если набрали мало, но страниц ещё много — продолжаем до max_vacancies
            if page >= min_pages and len(collected) >= max_vacancies:
                break

        return collected[:max_vacancies], None

    except Exception as exc:
        return [], f"Ошибка при обращении к API ({api_url}): {exc}"


def load_vacancies_from_file(file_path: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Загружает вакансии из локального JSON-файла в проверочном режиме."""
    if not os.path.exists(file_path):
        return [], f"Файл не найден: {file_path}"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        vacancies = extract_vacancies_from_payload(payload)
        return vacancies, None
    except Exception as exc:
        return [], f"Ошибка чтения локального JSON ({file_path}): {exc}"