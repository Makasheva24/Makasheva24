# Проекты Python

🟢 Basic (Новичок) ✅ Синтаксис Python (переменные, типы данных)
                        ✅ Условные конструкции (if, else, elif)  
                        ✅ Циклы (for, while)
                        ✅ Функции (def, return)
                        ✅ Работа со списками, словарями, множествами
                        ✅ Обработка исключений (try/except)


## **🚀 Проект: "Менеджер задач (To-Do List) в консоли"**

📝 **Описание:**

Ты создашь **приложение в консоли**, которое позволяет пользователю **добавлять, удалять, просматривать и сохранять задачи.**

### **📌 Что ты здесь прокачаешь?**

✅ **Переменные и типы данных** → задачи будут храниться в **списке или словаре.**

✅ **Условные конструкции (if/else)** → будем проверять ввод пользователя.

✅ **Циклы (for, while)** → нужен бесконечный цикл для работы программы.

✅ **Функции (def, return)** → весь код разделён на функции.

✅ **Обработка исключений (try/except)** → защита от неверного ввода.

✅ **Работа со списками, словарями** → задачи хранятся в удобной структуре.

---

### **🚀 Функционал проекта**

✅ **1. Добавление задач**

✅ **2. Просмотр списка задач**

✅ **3. Удаление задач**

✅ **4. Сохранение задач в файл и загрузка при запуске**

✅ **5. Обработка ошибок (неверный ввод, удаление несуществующей задачи)**

---

### **🔧 Как будет выглядеть программа?**

Пользователь **видит меню**, выбирает действие (1-5), и программа выполняет команду.

```

===== To-Do List Manager =====
1. Добавить задачу
2. Посмотреть список задач
3. Удалить задачу
4. Сохранить и выйти
Выберите действие (1-4): _

```

---

### **🔥 Начальный код (основа проекта)**

Этот код **запускает бесконечный цикл** и позволяет **пользователю управлять задачами.**

```python

import json

# Загружаем задачи из файла (если файл существует)
def load_tasks():
    try:
        with open("tasks.json", "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

# Сохраняем задачи в файл
def save_tasks(tasks):
    with open("tasks.json", "w") as file:
        json.dump(tasks, file)

# Добавление задачи
def add_task(tasks):
    task = input("Введите новую задачу: ")
    tasks.append(task)
    print(f"Задача '{task}' добавлена!")

# Просмотр всех задач
def view_tasks(tasks):
    if not tasks:
        print("Нет активных задач.")
    else:
        print("\nВаши задачи:")
        for i, task in enumerate(tasks, 1):
            print(f"{i}. {task}")

# Удаление задачи
def delete_task(tasks):
    view_tasks(tasks)
    try:
        task_num = int(input("Введите номер задачи для удаления: ")) - 1
        if 0 <= task_num < len(tasks):
            removed_task = tasks.pop(task_num)
            print(f"Задача '{removed_task}' удалена.")
        else:
            print("Некорректный номер задачи.")
    except ValueError:
        print("Введите корректный номер!")

# Основной цикл программы
def main():
    tasks = load_tasks()

    while True:
        print("\n===== To-Do List Manager =====")
        print("1. Добавить задачу")
        print("2. Посмотреть список задач")
        print("3. Удалить задачу")
        print("4. Сохранить и выйти")

        choice = input("Выберите действие (1-4): ")

        if choice == "1":
            add_task(tasks)
        elif choice == "2":
            view_tasks(tasks)
        elif choice == "3":
            delete_task(tasks)
        elif choice == "4":
            save_tasks(tasks)
            print("Задачи сохранены. Выход.")
            break
        else:
            print("Некорректный ввод. Попробуйте снова.")

if __name__ == "__main__":
    main()

```

---

### **🚀 Что дальше?**

1️⃣ **Скопируй код и запусти его в терминале.**

2️⃣ **Попробуй добавить, удалить и посмотреть задачи.**

3️⃣ **Добавь что-то своё:**



🔵 Medium (Средний)	 ✅ ООП (Классы, объекты, наследование, инкапсуляция)
                    ✅ Работа с файлами (JSON, CSV, TXT)
                    ✅ Регулярные выражения (re)
                    ✅ Лямбда-функции, map, filter, reduce
                    ✅ Работа с API (requests, BeautifulSoup, Selenium)
                    ✅ Основы асинхронности (asyncio, threading)

### 1. **Проект: Создание менеджера задач с использованием ООП**

**Задача:** Напиши программу для управления задачами с возможностью создания, редактирования, удаления и отображения задач.

**Что нужно сделать:**

- Реализовать **классы** для задачи (Task), списка задач (TaskList).
- Каждый **класс** должен содержать методы для добавления, удаления, редактирования задач.
- Включить **инкапсуляцию** — например, данные о задаче должны быть доступны только через методы.
- Используй **наследование**: сделай класс `PriorityTask`, который наследует от `Task` и добавляет возможность задания приоритета.
- Используй **регулярные выражения**, чтобы фильтровать задачи по имени или описанию.

**Дополнительные шаги:**

- Сохраняй задачи в **JSON** файле и давай пользователю возможность загружать/сохранять их в файл.
- Добавь возможность поиска задач с использованием регулярных выражений.
- Реализуй команду **сортировки задач** по приоритету или по дате создания, используя **lambda-функции**.

```python
import json
import re

class Task:
    """Класс, представляющий обычную задачу"""
    
    def __init__(self, name):
        self.name = name
        
    def to_dict(self):
        return {"name":self.name, "type":"Task"}
    
    @staticmethod
    def from_dict(data):
        return Task(data["name"])
    
class PriorityTask(Task):
    """Класс для приоритетных задач"""
    
    def __init__(self, name, priority):
        super().__init__(name)
        self.priority = priority
        
    def to_dict(self):
        return{"name":self.name, "priority": self.priority, "type":"PriorityTask"}
    
    @staticmethod
    def from_dict(data):
        return PriorityTask(data["name"], data["priority"])

    def __str__(self):
        return f"[{self.priority.upper()}] {self.name}"

class TaskList:
    """Класс для управлениея списком задач"""
    
    FILE_NAME = "tasks.json"
    
    def __init__(self):
        self.tasks = self.load_tasks()
        
    def load_tasks(self):
        """Загрузка задач из файла"""
        try:
            with open(self.FILE_NAME, "r", encoding="utf-8") as file:
                data = json.load(file)
                print("DEBUG: Загруженные данные:", data) 
                
            if not isinstance(data, list):  # Если JSON неожиданного формата
                print("⚠ Ошибка: JSON не является списком!")
                return []                
                
                tasks = []
                for task_data in data:
                    if isinstance(task_data, dict) and "type" in task_data:
                        if task_data["type"] == "PriorityTask":
                            tasks.append(PriorityTask.from_dict(task_data))
                        else:
                            tasks.append(Task.from_dict(task_data))
                    else:
                        print(f"⚠ Пропускаем некорректную запись: {task_data}")
                return tasks
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        
    def save_tasks(self):
        """Сохранение задач в файл"""
        if self.tasks is None:  # Если self.tasks по какой-то причине None, делаем его списком
            print("⚠ Ошибка: self.tasks оказался None! Сбрасываю в пустой список.")
            self.tasks = []
        
        
        with open(self.FILE_NAME, "w", encoding="utf-8") as file:
            json.dump([task.to_dict() for task in self.tasks], file, ensure_ascii=False, indent=4)
            
    def add_task(self):
        """Добавление новой задачи"""
        name = input("Введите новую задачу: ")
        self.tasks.append(Task(name))
        print(f"Задача '{name}' добавлена!")
        
    def add_priority_task(self):
        """Добавление приоритетной задачи"""
        name = input("Введите название приоритетной задачи: ")
        priority = input("Введите приоритет (высокий, средний, низкий): ").strip().lower()
        if priority not in ["высокий", "средний", "низкий"]:
            print("Некорректный приоритет! Используйте: высокий, средний, низкий.")
            return
        self.tasks.append(PriorityTask(name, priority))
        print(f"Приоритетная задача '{name}' добавлена с приоритетом '{priority}'.")

    def view_tasks(self):
        """Просмотр всех задач"""
        if not self.tasks:
            print("Нет активных задач.")
        else:
            print("\nВаши задачи:")
            for i, task in enumerate(self.tasks, 1):
                print(f"{i}.{task}")
                
    def delete_task(self):
        """Удаление задачи"""
        self.view_task()
        try:
            task_num = int(input("Введите номер задачи для удаления: ")) - 1
            if 0 <= task_num < len(self.tasks):
                removed_task = self.tasks.pop(task_num)
                print(f"Задача '{removed_task.name}' удалена.")
            else:
                print("Некорректный номер задачи.")
        except ValueError:
            print("Введите корректный номер!")

    def search_tasks(self):
        """Поиск задач по названию с использованием регулярных выражений"""
        keyword = input("Введите ключевое слово для поиска: ")
        pattern = re.compile(keyword, re.IGNORECASE)
        results = [task for task in self.tasks if pattern.search(task.name)]
        if results:
            print("\nНайденные задачи:")
            for task in results:
                print(task)
        else:
            print("Задачи не найдены.")

    def sort_tasks(self):
        """Сортировка задач по приоритету (приоритетные задачи идут первыми)"""
        priority_order = {"высокий": 0, "средний": 1, "низкий": 2}
        self.tasks.sort(key=lambda task: priority_order.get(task.priority, 99))
        print("Задачи отсортированы по приоритету!")

def main():
    """Основной цикл программы"""
    task_list = TaskList()

    while True:
        print("\n===== To-Do List Manager =====")
        print("1. Добавить задачу")
        print("2. Посмотреть список задач")
        print("3. Удалить задачу")
        print("4. Добавить приоритетную задачу")
        print("5. Поиск задач")
        print("6. Сортировка по приоритету")
        print("7. Сохранить и выйти")

        choice = input("Выберите действие (1-7): ")

        if choice == "1":
            task_list.add_task()
        elif choice == "2":
            task_list.view_tasks()
        elif choice == "3":
            task_list.delete_task()
        elif choice == "4":
            task_list.add_priority_task()
        elif choice == "5":
            task_list.search_tasks()
        elif choice == "6":
            task_list.sort_tasks()
        elif choice == "7":
            task_list.save_tasks()
            print("Задачи сохранены. Выход.")
            break
        else:
            print("Некорректный ввод. Попробуйте снова.")

if __name__ == "__main__":
    main()

```

## 🔐 **Project Codename: “DataVault Hunter”**

> “Не просто парсер. Это охотник за ценными данными, который адаптируется, фильтрует, сохраняет и анализирует.”
> 

---

### 🧩 Суть проекта:

Создаём **интерактивный Python-инструмент**, который:

- 🔎 Получает данные с веб-сайтов (API / парсинг HTML)
- 📁 Сохраняет в **JSON / CSV / TXT**
- 🎯 Использует **регулярки** для фильтрации и поиска ключевых паттернов
- 🧠 Организует всё в **ООП-архитектуре**
- 🧬 Добавляет **lambda + map/filter/reduce** для обработки
- 🔁 Делает **асинхронный сбор данных** (через `asyncio`, `aiohttp`, `threading`)
- 📜 Предоставляет консольный UI/меню (или GUI — опционально)

---

### ✅ Темы, которые ты закроешь:

| Тема | Как применяется |
| --- | --- |
| **ООП** | Классы `DataHunter`, `APIClient`, `Parser`, `FileWriter`, `FilterManager` |
| **Работа с файлами** | `with open(...)` для JSON/CSV/TXT, сохранение логов |
| **Регулярки** | Поиск email, ключевых слов, ID, паттернов |
| **Lambda/map/filter/reduce** | Фильтрация, чистка, трансформация данных |
| **API + BeautifulSoup** | Сбор с сайтов/открытых API |
| **Асинхронность** | Сбор с нескольких источников, параллельный запуск |

---

### 🔥 Пример сценария:

```
text
КопироватьРедактировать
[1] Введи ключевое слово → GPT
[2] Выбери источник → Хабр, Википедия, API YouTube
[3] Собираем данные...
[4] Найдено 152 совпадения
[5] Сохранить в? → data/output_gpt.json
[6] Найдено 27 email-адресов, 12 номеров телефонов

```

---

### 🎮 Уровень: Medium

📈 XP за проект: **+80 XP**

🚀 Бонус: легко расширяется до **GUI-версии** и прокачки до **"Senior Tool"** с ML/AI-функциями.

```python

import aiohttp
import asyncio
import json
import re
from bs4 import BeautifulSoup
import wikipedia
import nest_asyncio
nest_asyncio.apply()

class NewParser:
    file_name = "GPT.json"
    
    def __init__(self):
        self.data = []
        
    async def fetch(self, url):
        """Асинхронный запрос для получения контента с сайта"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                elif response.status == 404:
                    print("Страница не найдена!")
                elif response.status == 500:
                    print("Ошибка на сервере!")
                else:
                    print(f"Ошибка запроса {url}: {response.status}")
                    return None

    async def parse_site(self, url):

        html = await self.fetch(url)
        if html:
            soup = BeautifulSoup(html, 'html.parser')
            headers = soup.find_all(['h1', 'h2', 'h3', 'title'])  
            if not headers:
                print(f"Нет заголовков на странице {url}")
            for header in headers:
                header_text = header.get_text().strip()
                if header_text:  # Фильтрация пустых заголовков
                    title = re.sub(r'\s+', ' ', header_text)  # Убираем лишние пробелы
                    self.data.append({"title": title, "source": url})
            print(f"Заголовки для {url}: {[header.get_text() for header in headers]}")  # Выводим заголовки

    async def get_data(self, urls):
        info = []
        for url in urls:
            info.append(self.parse_site(url))
        await asyncio.gather(*info)

    def save_data(self):
        """Сохраняем данные в JSON файл"""
        if self.data:
            with open(self.file_name, "w", encoding="utf-8") as file:
                json.dump(self.data, file, ensure_ascii=False, indent=4)
            print(f"Данные сохранены в {self.file_name}")
        else:
            print("⚠️ Нет данных для сохранения.")

    def load_data(self):
        """Загружаем данные из JSON файла"""
        try:
            with open(self.file_name, "r", encoding="utf-8") as file:
                self.data = json.load(file)
                print(f"Данные успешно загружены ✅: {self.data}")
        except (FileNotFoundError, json.JSONDecodeError):
            print("Ошибка при загрузке данных ❌. Файл не найден или поврежден.")
            self.data = []

    def check_and_fill_data(self, urls):
        if not self.data:
            print("Нет данных в файле. Запускаем парсинг...")
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.get_data(urls))
        else:
            print("Данные загружены, продолжаем.")

# URLs для парсинга
urls = [
    "https://habr.com/ru/articles/",  
    "https://www.youtube.com/",  
]

# Запуск парсера
parser = NewParser()

# Загружаем старые данные, если они есть
parser.load_data()

# Проверяем, есть ли данные, если нет - парсим новые
parser.check_and_fill_data(urls)

# Фильтрация новостей по ключевому слову "GPT"
keyword = "GPT"
filtered_data = list(filter(lambda x: keyword.lower() in x['title'].lower(), parser.data))

# Вывод отфильтрованных новостей
print("Отфильтрованные данные:")
for data in filtered_data:
    print(data)

# Сохранение новых данных
parser.save_data()

# 📬 Email и телефон из заголовков
def extract_emails(text):
    return re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', text)

def extract_phone_numbers(text):
    return re.findall(r'\b\d{10,12}\b', text)

# 🔍 Обработка текстов из заголовков
all_titles = ' '.join([entry['title'] for entry in parser.data])
emails = extract_emails(all_titles)
phones = extract_phone_numbers(all_titles)

print(f"\n📧 Найдено email-адресов: {emails}")
print(f"📱 Найдено номеров телефонов: {phones}")

# 📖 Wikipedia по ключевому слову
def get_wikipedia_data(keyword):
    wikipedia.set_lang("ru")
    try:
        summary = wikipedia.summary(keyword, sentences=3)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"🔀 Найдено несколько значений: {e.options}"
    except wikipedia.exceptions.HTTPTimeoutError:
        return "⏳ Ошибка соединения"

wiki_summary = get_wikipedia_data(keyword)
print(f"\n📘 Wikipedia:\n{wiki_summary}")
```