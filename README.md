# Search Ranking — гибридный поиск статей для RAG

Проект ранжирует статьи базы знаний по пользовательскому запросу для дальнейшего
использования в RAG-пайплайне. Поиск объединяет BM25, TF-IDF, BGE-M3 и Query-kNN,
после чего GTE cross-encoder уточняет порядок найденных статей.

Результат работы — `answer.csv` с десятью `article_id` для каждого запроса
из `test.f`. Форматы входных данных и результата описаны в
[документации](docs/data-format.md).

## Как воспроизвести answer.csv

Требуется Python 3.11 или новее.

```bash
python -m pip install -r requirements.txt
python generate_answer.py
```

Команда создаёт `answer.csv` с колонками `query_id,answer`: ровно одна строка
для каждого из 500 test-запросов и 10 уникальных существующих статей в строке.

Быстрый запуск работает офлайн. Он повторно вычисляет нормализацию, смешивание
scores, ранжирование и форматирование CSV из сохранённых результатов моделей. Готовые строки
`answer` в артефакте не хранятся. Артефакт содержит top-100 кандидатов для
каждого запроса и GTE scores только для позиций 1-30.

```text
SHA-256 answer.csv: dc2c19cf7c1971f9ad4f2ac398f97e571f30a55f93d1c2216ec7c14ae7f24be7
```


Повторная сборка retrieval с проверенным GTE score-cache:

```bash
python -m pip install -r requirements.txt
python generate_answer.py --rebuild
```

Первый полный запуск скачивает закреплённые revisions BGE-M3 и GTE, затем
строит retrieval index. GTE scores переиспользуются только при полном совпадении
model/revisions и hash всех top-30 входных текстов.

Принудительный пересчёт GTE scores:

```bash
python generate_answer.py --rebuild --force-model
```

CUDA используется автоматически, при её отсутствии остаётся CPU fallback.

Полный end-to-end запуск, включая повторную обработку исходного HTML:

```bash
python generate_answer.py --rebuild --reprocess --force-index
```

Быстрый режим и `--rebuild` с валидированным score-cache формируют побайтно
одинаковый `answer.csv`. `--force-model` предназначен для проверки самой модели:
низкоуровневые FP16 CUDA kernels могут давать минимальные численные отличия между
GPU/версиями драйверов.

## Подход

Pipeline состоит из четырёх этапов:

1. HTML-препроцессинг статей.
2. Формирование retrieval chunks.
3. Гибридный candidate generation.
4. Zero-shot reranking top-30 и выбор top-10.

### Обработка HTML

HTML разбирается как DOM, а не регулярными выражениями. Каждая статья
преобразуется в последовательность типизированных блоков:

- `h1-h6`, `headline`, заголовки `chunk`, вкладок и спойлеров;
- абзацы и элементы списков;
- предупреждения и factoid-блоки;
- строки таблиц с сохранением заголовков колонок;
- `alt` отдельных изображений и обычный текст вне известных контейнеров.

`script`, `style`, `noscript` и элементы ввода удаляются. Inline-ссылки и `alt`
изображений сохраняются в тексте блока. Внутренние ссылки вида `/articles/id`
извлекаются и фильтруются по множеству реально существующих статей.

Для каждого блока сохраняется `section_path`. Стек обновляется по уровням
заголовков, а `chunk`, spoiler и tab добавляют scoped-сегменты только для своего
содержимого. Это сохраняет локальный смысл одинаковых абзацев в разных разделах.

Текст очищается в двух формах:

- `clean_text`: HTML entities, Unicode NFC и нормализованные пробелы;
- `normalized_text`: Unicode NFKC, `casefold`, `ё -> е`, пунктуация в пробел.

Последовательные дубли объединяются только при одинаковом `section_path`.
Глобальная дедупликация не выполняется. Таблицы превращаются в строки вида
`Заголовок: значение | ...`; вложенные таблицы отдельно не индексируются.

Для retrieval соседние блоки одного раздела упаковываются в chunks до 384
токенов с overlap 64 токена. Длинные блоки делятся token-window.

### Candidate generation

В production используются четыре поисковых канала:

- BM25 по chunks;
- char TF-IDF `char_wb`, n-граммы 3-5;
- dense retrieval по BGE-M3;
- supervised Query-kNN по размеченным calibration-запросам.

Query-kNN смешивает word и char TF-IDF similarity. Выбранные параметры:

```text
neighbors = 75
word_weight = 0.25
char_weight = 0.75
frequency_penalty = 0.5
voting = similarity_sum
```

Score каждого канала нормализуется внутри запроса. Итоговый linear fusion:

```text
BM25 chunks  = 0.1875
char TF-IDF  = 0.0625
BGE-M3       = 1.0000
Query-kNN    = 0.2500
```

BM25 по заголовкам участвовал в OOF-подборе, получил нулевой вес и исключён из
production build/search. Его исторические scores сохранены только в OOF feature
matrix для точного воспроизведения эксперимента. Candidate generation возвращает
top-100.

Dense-модель зафиксирована по revision:

```text
BAAI/bge-m3
5617a9f61b028005a4858fdac845db406aefb181
```

### Reranking

Для первых 30 кандидатов zero-shot cross-encoder получает:

```text
query + article title + best BM25 chunk / best dense chunk
```

Одинаковые chunks не дублируются. Для статьи выбирается максимальный score среди
её представлений. Затем применяется консервативная смесь:

```text
final_score = 0.25 * minmax(GTE score) + 0.75 * minmax(fusion score)
```

Позиции 31-100 не оцениваются cross-encoder и сохраняют исходный порядок.
Сортировка активной части стабильная: `final_score desc`, исходный candidate
rank, `article_id`.
Используется модель из 305 959 681 параметра:

```text
Alibaba-NLP/gte-multilingual-reranker-base
8215cf04918ba6f7b6a62bb44238ce2953d8831c
```

Перед загрузкой reranker число параметров считывается из `safetensors`. Запуск
останавливается при превышении 1 млрд параметров. Remote-code repo и revision
хранятся в `configs/reranker.toml`, входят в cache identity и передаются в
`CrossEncoder` через `code_revision`; ветка `main` не используется.

## Проверка качества на calibration.f

Основные метрики: `MAP@10` и `Recall@K`. В `ground_truth` может быть несколько
статей, поэтому MAP лучше отражает качество полного ранжированного top-10, чем
обычная top-1 accuracy.

Использовалась grouped nested 5-fold OOF-схема с seed 42:

- нормализованные дубли запросов всегда находятся в одном fold;
- outer fold используется только для финальной оценки;
- параметры Query-kNN, fusion и `alpha` выбираются без labels outer fold;
- Query-kNN и частоты статей вычисляются только по train-folds.

Проверено 315 конфигураций Query-kNN. Итоговые честные OOF-результаты:

| Вариант | MAP@10 | Recall@10 | Recall@50 | Recall@100 |
|---|---:|---:|---:|---:|
| Fixed BGE fusion | 0.690908 | 0.932333 | 0.986333 | 0.993000 |
| Tuned Query-kNN fusion | 0.700449 | 0.939667 | 0.986333 | 0.993000 |
| GTE top-30 + fusion | **0.704632** | **0.945667** | 0.986333 | 0.993000* |

`*` Recall@100 задаётся candidate generation; reranker меняет порядок только
внутри top-30. Query-kNN улучшил fusion на всех пяти outer folds. Nested OOF
выбрал `alpha=0.25` на каждом fold; GTE top-30 улучшил MAP@10 на четырёх из
пяти folds, поэтому его вес оставлен небольшим.

Пофолдовые результаты находятся в `metrics/`.

Полное воспроизведение 315-конфигурационного Query-kNN nested search:

```bash
python evaluate.py query-knn
```

Повторный расчёт nested OOF-метрик reranker из поставленных top-50 пар:

```bash
python evaluate.py reranker
```

Последовательный запуск обоих этапов:

```bash
python evaluate.py all
```

OOF score-cache проверяется по model/revision, remote-code repo/revision и hash
всех текстов query/article/chunks. При несовпадении scores рассчитываются заново.

## Анализ ошибок

Ошибки разделялись по этапу возникновения.

### Candidate generation

`Recall@100 = 0.993`: небольшая часть релевантных статей отсутствует даже в
top-100, и reranker уже не может её восстановить. Чаще это короткие или
неоднозначные запросы без терминов из нужной статьи.

Что сделано:

- добавлен BGE-M3 для семантических перефразировок;
- сохранены BM25 и char TF-IDF для точных формулировок, чисел и опечаток;
- Query-kNN использует похожие размеченные запросы;
- fusion сохраняет разнообразие каналов и высокий Recall@100.

### Reranker

Для GTE top-30 разобраны все 60 запросов, на которых AP@10 ухудшился:

| Тип ошибки | Количество | Наблюдение |
|---|---:|---|
| Потеря второй релевантной статьи | 42 | Cross-encoder поднимает наиболее очевидную статью, но опускает дополнительную инструкцию или правило |
| Переоценка общего тематического текста | 15 | Общая статья выглядит семантически похожей, хотя не отвечает на конкретное условие запроса |
| Смешение близких статей | 3 | Модель путает документы одной темы с разными сценариями или ограничениями |

Что сделано:

- reranker ограничен top-30, неизменённый хвост не теряется;
- GTE score смешивается с retrieval score, а не заменяет его;
- выбран небольшой `alpha = 0.25`, чтобы снизить переоценку общего текста;
- сохранён Query-kNN, который помогает удерживать несколько релевантных статей;
- depth и `alpha` едины для всех запросов; per-query настройки отсутствуют.

## Структура репозитория

```text
search-ranking/
|-- data/                 # articles.f, calibration.f, test.f
|-- docs/                 # форматы данных и результата поиска
|-- artifacts/            # блоки статей, model scores, manifest с SHA-256
|-- configs/              # production-конфигурации retrieval и reranker
|-- preprocessing/        # DOM -> статья и типизированные блоки
|-- retrieval/            # chunks, BM25, TF-IDF, BGE-M3, Query-kNN, fusion
|-- reranking/            # zero-shot GTE и parameter guard
|-- metrics/              # итоговые и пофолдовые OOF-метрики
|-- tests/                # unit-тесты ключевой логики и формата ответа
|-- generate_answer.py    # единая точка запуска
|-- evaluate.py           # воспроизведение Query-kNN и reranker OOF
|-- submission.py         # финальная нормализация, blend и валидация CSV
|-- rebuild.py            # полный пересчёт retrieval и reranker
`-- answer.csv            # воспроизведённый итоговый файл
```

## Тесты

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q
```

Проверяются HTML-правила, дедупликация chunks, parameter guard, cache identity,
нормализация, стабильная сортировка, уникальность статей и полное покрытие
test-запросов. Текущий результат: `31 passed`.
