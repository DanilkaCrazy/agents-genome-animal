## Genome AI Agents — end-to-end data pipeline

Этот репозиторий закрывает задания 1–4 (агенты) и финальный пайплайн одной командой.

### ML-задача (очень подробно, в стиле GenomeAI)

**Тема:** «Прогнозирование здоровья и продуктивности КРС на основе разнородных данных».

На практике “разнородные данные” на ферме выглядят так:
- **табличные записи** (кормление, события здоровья, показатели продуктивности, лабораторные анализы);
- **сигналы/симптомы** (иногда как бинарные признаки, иногда как текстовые заметки ветврача/зоотехника);
- **внешний контекст** (например, цены на корма/молоко, которые влияют на управленческие решения и интерпретацию).

В рамках учебного пайплайна мы делаем “Deep Tech”-версию, но сохраняем контракт задания: **каждый источник приводим к единой схеме** `text/label/source/collected_at`.

#### Что именно предсказываем (label)
В нашем шаблоне `label` — это **состояние здоровья / заболевание** (класс).
Если в источнике уже есть метка (табличный датасет с “health/disease/prognosis”) — мы кладём её в `label`.
Если метки нет (скрапинг цен) — `label="unknown"` и источник работает как “контекстные тексты”.

#### Что такое `text` при табличных данных
Чтобы унифицировать источники под контракт, мы превращаем одну строку табличного датасета в текстовый “паспорт наблюдения”:

`text = "cattle_record: feature1=value1; feature2=value2; ..."`

Это позволяет:
- использовать единый интерфейс агентов (все видят `text`);
- запускать EDA по длине “текстов” и топ-словам/токенам (требование задания);
- использовать `AnnotationAgent` (авторазметка) для строк без меток.

### Источники данных (Kaggle + Hugging Face + scraping агро-портала)

В `config.yaml` по умолчанию настроены **3 источника по теме**:

1) **Kaggle Dataset (табличные данные про КРС)**
- **Источник:** `shahhet2812/cattle-health-and-feeding-data`
- **Роль в пайплайне:** основной “фермерский журнал” (кормление/состояние/признаки → метка здоровья).
- **Как приводим к контракту:** если в CSV нет готового текстового столбца, строим `text` из всех колонок (кроме `label_col`), а `label` берём из `label_col`.

2) **Hugging Face Dataset (симптомы → диагноз)**
- **Источник:** `roshan8312/cattle-disease-prediction`
- **Роль в пайплайне:** второй независимый открытый источник (симптомы/признаки → `prognosis`).
- **Как приводим к контракту:** строим `text` из набора симптомов/признаков (`symptoms: col=value; ...`), а `label` = `prognosis`.

3) **Scraping (внешний контекст: цены на корма)**
- **Источник:** публичная страница IndexMundi с таблицей по кормам (soybean meal):
  `https://www.indexmundi.com/commodities/?commodity=soybean-meal&months=12`
- **Роль в пайплайне:** показать, что агент умеет получать данные “снаружи” (scraping), и добавить контекстные записи (без меток).
- **Как приводим к контракту:** каждую строку таблицы превращаем в `text`, `label="unknown"`, `source="scrape:<url>"`.

### Стандартная схема датасета (контракт между агентами)
Агенты 1–2 обязаны возвращать `pd.DataFrame` со стандартными колонками:
- **`text`**: текст (в этом шаблоне используется для `modality=text`)
- **`audio`**: audio-поле (в text-only пайплайне всегда `null`)
- **`image`**: image-поле (в text-only пайплайне всегда `null`)
- **`label`**: метка (может быть `unknown` до разметки)
- **`source`**: строка-источник (`hf:...`, `api:...`, `scrape:...`)
- **`collected_at`**: UTC timestamp (ISO)

### Установка

```bash
python -m pip install -r requirements.txt
```

### Запуск пайплайна одной командой

```bash
python run_pipeline.py
```

Для борьбы с дисбалансом и утечками в AL-настройках (`config.yaml -> active_learning`) добавлены параметры:
- `dedup_before_split` — дедупликация по `text` до train/test split
- `unknown_policy` — `drop|cap|keep` для класса `unknown`
- `unknown_cap_ratio` — доля `unknown` при `unknown_policy=cap`
- `min_class_count` — минимальный размер класса для участия в AL

Артефакты сохраняются в `artifacts/run_<run_id>/`:
- `data_raw.csv` — собранные данные
- `data_clean.csv` — очищенные данные
- `data_auto_labeled.csv` — авторазметка + confidence
- `review_queue.csv` — HITL очередь (confidence ниже порога)
- `review_queue_corrected.csv` — файл, который человек создаёт/правит вручную
- `data_labeled_final.csv` — итоговая таблица с `label_final`
- `labelstudio_import.json` — импорт в LabelStudio
- `reports/quality_report.json` — найденные проблемы качества
- `reports/quality_compare.csv` — сравнение до/после
- `reports/annotation_spec.md` — спецификация разметки
- `reports/annotation_metrics.json` — метрики качества разметки
- `reports/al_history.csv` и `reports/learning_curve.png` — AL эксперимент (если данных достаточно)
- `run_manifest.json` — manifest прогона (run_id + пути)

Также Pipeline пишет “latest” артефакты в структуру проекта:
- `data/raw/data_raw.csv`
- `data/labeled/data_clean.csv`, `data/labeled/data_auto_labeled.csv`, `data/labeled/data_labeled_final.csv`
- `review_queue.csv` (очередь HITL для удобства)
- `reports/quality_report.md`, `reports/annotation_report.md`, `reports/al_report.md`, `reports/final_report.md`
- `models/final_model.pkl` и `models/vectorizer.pkl` (если AL не пропущен)

### Human-in-the-loop (обязательная точка)
1) После первого запуска откройте `review_queue.csv`.
2) Заполните `label_human` и сохраните как `review_queue_corrected.csv` в той же папке `artifacts/run_<run_id>/`.
3) Перезапустите:

```bash
python run_pipeline.py --resume-hitl
```

### Ноутбуки (EDA и AL)
- `notebooks/eda.ipynb`: распределение классов, длины текстов, top-20 слов.
- `notebooks/al_experiment.ipynb`: сравнение стратегий AL (entropy vs random) и кривая обучения.

