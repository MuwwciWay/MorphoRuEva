import json
import re
import pandas as pd
from transformers import pipeline, AutoTokenizer
import logging
import numpy as np

def parse_json(file_path):
    # Загружаем JSON файл
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Будем использовать список для хранения информации
    extracted_data = []

    for item in data:
        # Проверяем заголовок задания
        if item.get("title") == "Расставьте знаки препинания.":
            # Извлекаем описание и определяем нужный знак препинания
            description = item.get("description", "")
            punctuation_sign = extract_punctuation_sign(description)
            
            # Извлекаем само задание и ответ ученика
            task_text = item.get("task", "")
            student_answer = item.get("answer", "")
            answer = item.get("input_answer", "")
            
            # Сохраняем данные в виде словаря
            extracted_data.append({
                "description": description,
                "punctuation_sign": punctuation_sign,
                "task_text": task_text,
                "student_answer": student_answer,
                "answer": answer
            })

    # Конвертируем список в DataFrame для удобства анализа
    df = pd.DataFrame(extracted_data)
    return df

def extract_punctuation_sign(description):
    """Функция для извлечения знака препинания из описания задания."""
    if "запятые" in description.lower():
        return ","
    elif "тире" in description.lower():
        return "—"
    elif "двоеточие" in description.lower():
        return ":"
    elif "точка с запятой" in description.lower():
        return ";"
    return None

# Пример вызова функции
file_path = 'dataset.json'
df = parse_json(file_path)

# Отключаем все логирование
logging.disable(logging.CRITICAL)

# Загрузка модели для расстановки пунктуации
pt = "RUPunct/RUPunct_big"

tk = AutoTokenizer.from_pretrained(pt, strip_accents=False, add_prefix_space=True)
classifier = pipeline("ner", model=pt, tokenizer=tk, aggregation_strategy="first")


import re

def process_token(token, label, prev_char, is_start_of_sentence):
    """Обработка токенов в зависимости от их метки без изменения регистра, с капитализацией в начале предложения."""
    new_token = token
    substitution = None

    if label == "LOWER_O":
        return token, None
    if label in ["LOWER_COMMA", "UPPER_COMMA", "UPPER_TOTAL_COMMA"]:
        if prev_char != ',':
            new_token += ","
            substitution = ","
    elif label in ["LOWER_QUESTION", "UPPER_QUESTION", "UPPER_TOTAL_QUESTION"]:
        if prev_char != '?':
            new_token += "?"
            substitution = "?"
    elif label == "LOWER_TIRE":
        new_token = " " + token + " —"
        substitution = " —"
    elif label == "UPPER_TIRE":
        new_token = " " + token.capitalize() + " —"
        substitution = " —"
    elif label == "LOWER_DVOETOCHIE":
        new_token += ":"
        substitution = ":"
    elif label == "LOWER_VOSKL":
        new_token += "!"
        substitution = "!"
    elif label == "LOWER_PERIODCOMMA":
        new_token += ";"
        substitution = ";"
    elif label == "LOWER_DEFIS":
        new_token += "-"
        substitution = "-"
    elif label == "LOWER_MNOGOTOCHIE":
        new_token += "..."
        substitution = "..."
    elif label == "LOWER_QUESTIONVOSKL":
        new_token += "?!"
        substitution = "?!"
    elif label == "UPPER_O":
        new_token = token.capitalize()
    elif label == "UPPER_TOTAL_O":
        new_token = token.upper()

    # Если это начало предложения, делаем первую букву заглавной
    if is_start_of_sentence and new_token:
        new_token = new_token.capitalize()

    return new_token, substitution

def punctuate_and_get_symbols(sentence):
    """Функция для расстановки пунктуации и генерации списка вставленных знаков для одного предложения."""
    modified_sentence = re.sub(r'\(\d+\)', '', sentence).strip()
    modified_sentence = re.sub(r'[^\w\s]', '', modified_sentence)

    preds = classifier(modified_sentence)

    output = ""
    prev_char = ""
    symbols = {}
    symbol_counter = 1  # Счетчик для нумерации вставленных знаков

    for index, item in enumerate(preds):
        is_start_of_sentence = (index == 0 or prev_char in ".!?")  # Определяем, является ли токен началом предложения
        token, substitution = process_token(item['word'].strip(), item['entity_group'], prev_char, is_start_of_sentence)

        if substitution:
            symbols[symbol_counter] = substitution
            symbol_counter += 1  # Увеличиваем счетчик только при добавлении знака

        if token and token[0] not in ",!?;:":  # Не добавляем лишние пробелы
            output += " " + token
        else:
            output += token

        prev_char = token[-1] if token else prev_char

    output = output.strip()
    return output, symbols



def process_text(input_text):
    """Функция для обработки всего текста, разбивая его на предложения."""
    sentences = re.split(r'(?<=[.!?]) +', input_text)
    processed_sentences = []
    all_symbols = {}

    for i, sentence in enumerate(sentences):
        processed_sentence, symbols = punctuate_and_get_symbols(sentence)

        # Добавляем точку в конце, если она отсутствует
        if processed_sentence and processed_sentence[-1] not in ".!?":
            processed_sentence += "."
        
        processed_sentences.append(processed_sentence)

        for num, symbol in symbols.items():
            all_symbols[len(all_symbols) + 1] = symbol  # Нумерация знаков по порядку

    final_output = ' '.join(processed_sentences)
    return final_output, all_symbols

def extract_punctuation_sign(task_description):
    """Функция для извлечения знака препинания из задания."""
    if "запятые" in task_description.lower():
        return ","
    elif "тире" in task_description.lower():
        return "—"
    elif "двоеточие" in task_description.lower():
        return ":"
    elif "точка с запятой" in task_description.lower():
        return ";"
    return None

def find_punctuation(input_text, task_description):
    """Функция для нахождения позиций нужного знака."""
    required_sign = extract_punctuation_sign(task_description)
    if required_sign is None:
        return []

    processed_text, symbols = process_text(input_text)
    
    # Получаем все метки из текста, даже если они пусты
    markers = sorted(int(num) for num in re.findall(r'\((\d+)\)', input_text))
    
    # Заполняем позиции, включая "N" для пустых меток
    all_positions = {marker: symbols.get(marker, 'N') for marker in markers}

    return processed_text, all_positions

def compare_punctuation(input_text, punctuation_indices):
    """Сравнивает, какие метки были заменены на какие знаки."""
    markers = re.findall(r'\((\d+)\)', input_text)

    changes = {}
    for marker in markers:
        marker_num = int(marker)
        original_marker = f"({marker_num})"
        added_symbol = punctuation_indices.get(marker_num, "N")

        # Сопоставление замены
        changes[original_marker] = added_symbol

    return changes

def extract_labels(text):
    """Извлекает метки в формате (число) из текста."""
    labels = re.findall(r'\(\d+\)', text)
    return labels

def compare_labels(original_text, modified_text):
    """Сравнивает метки и определяет, что с ними произошло в модифицированном тексте."""
    changes = {}

    # Разделяем оригинальный текст на слова
    original_words = original_text.split()

    # Проходим по словам и ищем метки
    for i in range(len(original_words)):
        word = original_words[i]
        
        # Если это метка
        if '(' in word and ')' in word:
            label = word  # Сохраняем метку

            # Берем слово после метки как правую границу
            right_word = original_words[i + 1] if i < len(original_words) - 1 else ''

            # Ищем правую границу в изменённом тексте
            right_index = modified_text.find(right_word)

            if right_index == -1:
                changes[label] = 'Nan'  # Если правое слово не найдено
                continue

            # Ищем левую границу
            left_word = original_words[i - 1] if i > 0 else ''

            left_index = modified_text.rfind(left_word, 0, right_index)

            if left_index == -1:
                changes[label] = 'Nan'  # Если левое слово не найдено
                continue
            
            # Проверяем расстояние между границами
            while True:
                # Ищем пробелы и специальные символы между границами
                distance = right_index - (left_index + len(left_word))  # Рассчитываем расстояние

                if distance <= 3:  # Если расстояние корректное
                    # Извлекаем текст между словами
                    text_between = modified_text[left_index + len(left_word):right_index].strip()

                    # Проверяем наличие специальных знаков
                    if any(char in text_between for char in " ,:;-—"):
                        changes[label] = text_between
                    else:
                        changes[label] = 'Nan'  # Если специальные знаки не найдены
                    break  # Выходим из цикла после успешного извлечения текста

                # Если расстояние слишком большое, ищем новую левую границу
                left_index = modified_text.rfind(left_word, 0, left_index)  # Ищем левую границу заново
                
                if left_index == -1 or left_index >= right_index:
                    changes[label] = 'Nan'  # Если не нашли соответствия
                    break  # Выходим из цикла, если левое слово не найдено

                left_word = modified_text[left_index:left_index + len(original_words[i - 1])].strip()

    return changes


def process_task_text(row):
    input_text = row['task_text']
    task_description = row['description']

    # Применяем ваш алгоритм
    processed_text, punctuation_indices = find_punctuation(input_text, task_description)

    label_changes = compare_labels(input_text, processed_text)
    
    # Возвращаем как кортеж или словарь, если нужно больше данных
    return processed_text, label_changes  # Или любой другой формат

# Обновляем DataFrame для хранения нескольких выходных данных
df[['model_answer', 'punctuation_indices']] = df.apply(process_task_text, axis=1, result_type='expand')

def filter_punctuation_indices(row):
    sign = row['punctuation_sign']
    indices = row['punctuation_indices']

    # Фильтруем индексы по заданному знаку пунктуации
    filtered_indices = [key[1] for key, value in indices.items() if value == sign]

    # Объединяем индексы в строку
    return ''.join(filtered_indices)

# Применяем фильтрацию к каждому ряду датафрейма
df['filtered_punctuation_indices'] = df.apply(filter_punctuation_indices, axis=1)

# Сравнение значений и создание нового столбца 'mark'
df['mark'] = df.apply(lambda row: f"{sum(1 for i in row['filtered_punctuation_indices'] if i in row['student_answer'])}/{len(row['student_answer'])}" if len(row['student_answer']) > 0 else "0/0", axis=1)

df['procent'] = df.apply(
    lambda row: (sum(1 for i in row['filtered_punctuation_indices'] if i in row['answer']) / len(row['answer']) * 100) if len(row['answer']) > 0 else 0, 
    axis=1
)

# Округляем до ближайшего целого числа и преобразуем в целочисленный тип
df['procent'] = df['procent'].round(0).astype(int)

# Формирование JSON выходных данных
assignments = []
for index, row in df.iterrows():
    assignment = {
        "id": index + 1,  # Индексация с 1
        "mark": row['mark'],
        "feedback": row['procent']
    }
    assignments.append(assignment)

output_json = {
    "assignments": assignments
}

# Сохранение в файл JSON
output_file = 'output.json'
with open(output_file, 'w') as json_file:
    json.dump(output_json, json_file, indent=3)
