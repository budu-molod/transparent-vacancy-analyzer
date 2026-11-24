import time
from typing import Callable
import schedule


def run_scheduled(job_fn: Callable[[], None], interval_minutes: int, run_immediately: bool = True) -> None:
    """Запускает функцию обработки вакансий по расписанию каждые interval_minutes минут."""
    if run_immediately:
        job_fn()

    schedule.every(interval_minutes).minutes.do(job_fn)
    print(f"[SCHEDULER] Запущен рабочий режим. Интервал: каждые {interval_minutes} мин. Для остановки нажмите Ctrl+C.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[SCHEDULER] Работа планировщика остановлена пользователем.")