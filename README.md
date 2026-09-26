# LoRA-классификатор роли участника по ИНН

Классификация роли участника (истец / ответчик / третье лицо) в судебном
решении по его ИНН. Энкодер `kazzand/ru-longformer-base-4096` заморожен,
дообучается только LoRA-адаптер (query/value в attention) + голова классификации.

## Структура

- `prepare_data.py` — превращает `annotations.json` в `train.jsonl` / `val.jsonl` / `test.jsonl`
- `model.py` — архитектура модели (Longformer + LoRA + голова)
- `dataset.py` — PyTorch Dataset поверх *.jsonl
- `train.py` — обучение, валидация по эпохам, финальная оценка на test

## Порядок запуска

Все команды — из папки проекта, с активным окружением, где установлены зависимости.

1. **Подготовить датасет:**
   ```bash
   python prepare_data.py
   ```

   На выходе — `train.jsonl`, `val.jsonl`, `test.jsonl` в этой же папке.

2. **Обучение:**
   ```bash
   python train.py
   ```
   Модель и датасеты подтягиваются автоматически (`model.py`/`dataset.py` импортируются как модули). По ходу обучения:
   - печатается `train loss` / `val loss` по эпохам и `classification_report` на val
   - после каждого улучшения `val loss` сохраняется чекпоинт: `lora_adapter/` (LoRA-веса) + `classifier_head.pt` (голова)
   - в конце — **одна** финальная оценка на `test.jsonl` с `classification_report` и confusion matrix, на **лучшем** сохранённом чекпоинте (не на последней эпохе)

## Оценка качества

`train.py` сам по себе покрывает весь цикл, который требовался изначально:
обучение → сохранение чекпоинта → инференс на test.jsonl (без обучения,
`model.eval()` + `torch.no_grad()`) → classification_report + confusion matrix.
Отдельно ничего запускать не нужно, всё в одном `python train.py`.
