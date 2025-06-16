from flask import Flask, abort, jsonify, render_template, request, redirect, url_for, session, json
import numpy as np
from fuzzy_logic import calculate_deviance, generate_combined_plot
import os
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime
import pytz  

app = Flask(__name__, static_url_path='/static')
app.jinja_env.globals.update(zip=zip)  # Добавляем функцию zip в шаблоны
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))

# Конфигурация БД 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///results.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

def get_moscow_time():
    return datetime.now(pytz.timezone('Europe/Moscow'))

# Модель для хранения результатов
class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    deviance_level = db.Column(db.Float)
    date = db.Column(db.DateTime, default=get_moscow_time)

    def __repr__(self):
        return f'<Result {self.id}>'

question_blocks = {
    "Агрессивное и рискованное поведение": [
        "Я легко ввязываюсь в ссоры или драки.",
        "Мне нравится чувствовать адреналин, даже если это опасно.",
        "Мне сложно сдержать агрессию, если кто-то меня задевает.",
        "Я не чувствую вины, если сделал(а) что-то плохое в ответ на обиду.",
        "Я считаю, что физическая сила — лучший способ решить конфликт.",
        "Мне нравится, когда другие боятся меня.",
        "Я часто говорю обидные или уничижительные слова другим.",
        "Когда я злюсь, я могу сломать вещи.",
        "Мне сложно контролировать себя в конфликтных ситуациях.",
        "Иногда мне хочется ударить человека, если он меня сильно бесит."
    ],
    "Делинквентное (противоправное) поведение": [
        "Я когда-либо воровал(а) что-то в магазине или у других.",
        "Я прогуливал(а) школу без уважительной причины.",
        "Я употреблял(а) алкоголь, несмотря на возрастные ограничения.",
        "Я курил(а) сигареты или вейп.",
        "Я бы мог(ла) сделать что-то незаконное, если был(а) уверен(а), что меня не поймают.",
        "Я участвовал(а) в драках, несмотря на запрет.",
        "Мне интересно нарушать правила — это даёт ощущение свободы.",
        "Я совершал(а) что-то запрещённое ради «прикола» или желания выделиться.",
        "Я помогал(а) другим совершать противоправные действия.",
        "Я распространял(а) слухи или ложную информацию о других."
    ],
    "Демонстративное игнорирование социальных норм": [
        "Мне скучно на школьных уроках, поэтому я часто отвлекаюсь или мешаю другим.",
        "Я регулярно опаздываю в школу или на другие обязательные мероприятия.",
        "Мне неинтересно помогать родителям или выполнять домашние обязанности.",
        "Я избегаю участия в школьных или общественных мероприятиях.",
        "Я часто игнорирую просьбы взрослых.",
        "Я считаю, что можно не выполнять обещания, если не хочется.",
        "Я нарушаю распорядок дня или другие режимные требования, даже если это вредно.",
        "Я не считаю нужным помогать другим, даже если у меня есть возможность.",
        "Я чувствую раздражение, когда меня просят соблюдать правила или быть «как все».",
        "Мне неинтересно, как моё поведение влияет на окружающих."
    ],
    "Аутоагрессия и депрессивные тенденции": [
        "Я иногда причиняю себе боль, чтобы почувствовать облегчение или контроль.",
        "Я часто чувствую, что не стоит продолжать делать что-либо, потому что это бесполезно.",
        "Я не забочусь о своем здоровье или внешности.",
        "Я делаю вещи, которые могут быть опасными для меня, хотя знаю, что это может мне навредить.",
        "Я иногда не могу проснуться утром, потому что не хочу сталкиваться с реальностью.",
        "Я чувствую себя беспомощным(ой) и не могу изменить свою жизнь.",
        "Я не люблю смотреть на себя в зеркало, потому что мне не нравится, что я вижу.",
        "Я часто чувствую, что никто не поймёт меня, и мне легче быть одному(ой).",
        "Иногда мне кажется, что жизнь не имеет смысла.",
        "Я склонен(на) обвинять себя в происходящих проблемах, даже если они не зависят от меня."
    ],
    "Манипулятивное и аморальное поведение": [
        "Я часто вру, чтобы избежать ответственности за свои поступки.",
        "Мне не важно, как мои действия влияют на чувства других людей.",
        "Я считаю, что ложь — это нормальный способ добиться своих целей.",
        "Я скрываю свои поступки, даже если они причиняют боль другим, чтобы избежать наказания.",
        "Я не чувствую вины за обман или манипуляции, если это помогает мне достичь своих целей.",
        "Я готов(а) использовать других людей ради своих интересов.",
        "Я часто говорю плохие вещи о других за их спинами.",
        "Мне кажется, что люди должны поступать так, как я считаю правильным, независимо от их чувств.",
        "Я не считаю, что нужно соблюдать правила честности в отношениях с людьми.",
        "Я считаю, что обманывать и манипулировать — это нормальная часть жизни."
    ]
}

@app.route('/', methods=['GET', 'POST'])
def start():
    if request.method == 'POST':
        session.clear()
        
        # Проверяем, нажата ли кнопка тестового режима
        is_test = 'test_mode' in request.form
        
        # Получаем данные из формы
        session['gender'] = request.form.get('gender')
        session['age'] = request.form.get('age')
        
        if is_test:
            # Тестовый режим
            session['result_saved'] = False  # Не сохранять в БД
            
            # Тестовые баллы
            session['scores'] = {
                'Агрессивное и рискованное поведение': 4.5,
                'Делинквентное (противоправное) поведение': 3.0,
                'Демонстративное игнорирование социальных норм': 5.5,
                'Аутоагрессия и депрессивные тенденции': 2.0,
                'Манипулятивное и аморальное поведение': 8.0
            }
            
            # Тестовые ответы
            session['answers'] = {
                block: ['yes' if i % 3 == 0 else 'sometimes' if i % 3 == 1 else 'no' 
                        for i in range(len(questions))]
                for block, questions in question_blocks.items()
            }
            
            return redirect(url_for('result'))
        else:
            # Обычный режим
            session['current_block'] = 0
            session['current_question'] = 0
            session['answers'] = {block: [] for block in question_blocks}
            session['scores'] = {block: 0 for block in question_blocks}
            return redirect(url_for('question'))
    
    return render_template('start.html')



@app.route('/question', methods=['GET', 'POST'])
def question():
    if 'current_block' not in session:
        return redirect(url_for('start'))
    
    block_names = list(question_blocks.keys())
    
    # Проверка выхода за границы блоков
    if session['current_block'] >= len(block_names):
        return redirect(url_for('result'))
    
    current_block_name = block_names[session['current_block']]
    questions = question_blocks[current_block_name]

    if request.method == 'POST':
        if 'back' in request.form:
            # Обработка кнопки "Назад"
            if session['current_question'] > 0:
                session['current_question'] -= 1
                if session['answers'][current_block_name]:
                    session['answers'][current_block_name].pop()
                
                # Уменьшаем балл
                last_answer = request.form.get('last_answer')
                if last_answer == 'yes':
                    session['scores'][current_block_name] -= 1
                elif last_answer == 'sometimes':
                    session['scores'][current_block_name] -= 0.5
                    
            elif session['current_block'] > 0:
                # Переход к предыдущему блоку
                session['current_block'] -= 1
                prev_block_name = block_names[session['current_block']]
                prev_questions = question_blocks[prev_block_name]
                session['current_question'] = len(prev_questions) - 1
                
                if session['answers'][prev_block_name]:
                    session['answers'][prev_block_name].pop()
                
                last_answer = request.form.get('last_answer')
                if last_answer == 'yes':
                    session['scores'][prev_block_name] -= 1
                elif last_answer == 'sometimes':
                    session['scores'][prev_block_name] -= 0.5
                    
            session.modified = True
            return redirect(url_for('question'))

        # Обработка ответа пользователя
        answer = request.form.get('answer')
        if answer:
            session['answers'][current_block_name].append(answer)

            if answer == 'yes':
                session['scores'][current_block_name] += 1
            elif answer == 'sometimes':
                session['scores'][current_block_name] += 0.5

            session['current_question'] += 1
            session.modified = True

        # Проверка завершения блока
        if session['current_question'] >= len(questions):
            session['current_block'] += 1
            session['current_question'] = 0
            
            # Проверка завершения всех блоков
            if session['current_block'] >= len(question_blocks):
                return redirect(url_for('result'))

        return redirect(url_for('question'))

    # Получение текущего вопроса
    current_q = session['current_question']
    
    # Проверка выхода за границы вопросов
    if current_q >= len(questions):
        session['current_question'] = len(questions) - 1
        return redirect(url_for('question'))
    
    # Расчет прогресса
    total_questions = sum(len(q) for q in question_blocks.values())
    current_question_number = sum(len(q) for q in list(question_blocks.values())[:session['current_block']]) + current_q + 1
    progress = (current_question_number / total_questions) * 100

    # Проверка наличия предыдущего ответа
    last_answer = ""
    if session['answers'][current_block_name] and session['current_question'] > 0:
        last_answer = session['answers'][current_block_name][-1]

    return render_template('question.html',
                           question=questions[current_q],
                           question_num=current_question_number,
                           total_questions=total_questions,
                           progress=progress,
                           current_block=session['current_block'],
                           current_question=session['current_question'],
                           last_answer=last_answer)


def get_level_description(score, max_score):
    percentage = (score / max_score) * 100
    levels = [
        (75, "Высокий уровень", "#e74c3c"),
        (50, "Средний уровень", "#e67e22"),
        (25, "Умеренный уровень", "#f1c40f"),
        (0, "Низкий уровень", "#2ecc71"),
    ]
    for threshold, desc, color in levels:
        if percentage >= threshold:
            return desc, color


@app.route('/result', methods=['GET', 'POST'])
def result():
    if 'scores' not in session:
        return redirect(url_for('start'))

    max_scores = {block: len(questions) for block, questions in question_blocks.items()}
    results = []

    # Расчет нечеткого вывода
    fuzzy_data = calculate_deviance(session['scores'], max_scores)
    
    # Сохраняем результаты в БД
    if not session.get('result_saved'):
        try:
            new_result = Result(
                age=session.get('age'),
                gender=session.get('gender'),
                deviance_level=fuzzy_data.get('value', 0)
            )
            db.session.add(new_result)
            db.session.commit()
            session['result_saved'] = True
        except Exception as e:
            print(f"Ошибка при сохранении в БД: {e}")

    # Формируем данные для отображения
    for block, score in session['scores'].items():
        max_score = max_scores[block]
        level, color = get_level_description(score, max_score)
        results.append({
            'block': block,
            'score': score,
            'max_score': max_score,
            'level': level,
            'color': color
        })

    # Генерируем график
    combined_plot = generate_combined_plot(fuzzy_data['normalized_scores'])
    
    # Сохраняем в сессии данные
    session['fuzzy_result'] = {
        'value': fuzzy_data['value'],
        'normalized_scores': fuzzy_data['normalized_scores']
    }

    return render_template('result.html',
                         results=results,
                         fuzzy_result=session['fuzzy_result'],
                         combined_plot=combined_plot,  
                         gender=session.get('gender'),
                         age=session.get('age'))

@app.route('/other_page')
def other_page():
    if 'answers' not in session:
        return redirect(url_for('start'))
    
    # Используем данные из сессии
    user_data = {
        'gender': session.get('gender'),
        'age': session.get('age'),
        'answers': session.get('answers'),
        'scores': session.get('scores'),
        'fuzzy_result': session.get('fuzzy_result', {})
    }

    return render_template(
        'other_page.html',
        user_data=user_data,
        question_blocks=question_blocks
    )

@app.route('/other_pages')
def other_pages():
    try:
        results = Result.query.order_by(Result.date.desc()).all()

        if not results:
            return render_template('other_pages.html',
                                results=[],
                                chart_data={},
                                stats={})

        # Подготовка данных для графиков
        chart_data = {
            'dates': [r.date.strftime('%Y-%m-%d') for r in results],
            'scores': [float(r.deviance_level) for r in results],
            'levels': [
                sum(1 for r in results if r.deviance_level >= 8.5),
                sum(1 for r in results if 7.5 <= r.deviance_level < 8.5),
                sum(1 for r in results if 5.5 <= r.deviance_level < 7.5),
                sum(1 for r in results if 4 <= r.deviance_level < 5.5),
                sum(1 for r in results if 2.5 <= r.deviance_level < 4),
                sum(1 for r in results if r.deviance_level < 2.5)
            ]
        }

        # Статистика
        stats = {
            'min': min(chart_data['scores']),
            'max': max(chart_data['scores']),
            'avg': sum(chart_data['scores'])/len(chart_data['scores']),
            'total': len(results)
        }

        return render_template('other_pages.html',
                            results=results,
                            chart_data=json.dumps(chart_data),
                            stats=json.dumps(stats))

    except Exception as e:
        app.logger.error(f"Error in other_pages: {str(e)}", exc_info=True)
        return render_template('error.html',
                            error_message="Ошибка загрузки данных",
                            error_details=str(e)), 500
        
        
# Эндпоинты API
@app.route('/api/start', methods=['POST'])
def api_start():
    data = request.json
    session['age'] = data['age']
    session['gender'] = data['gender']
    return jsonify({'status': 'ok'})

@app.route('/api/submit_answer', methods=['POST'])
def api_submit_answer():
    data = request.json
    # Логика обработки ответа...
    return jsonify({'next_question': question, 'progress': 75})

@app.route('/api/results', methods=['GET'])
def api_results():
    results = Result.query.all()
    return jsonify([r.to_dict() for r in results])
        
if __name__ == '__main__':
    app.run(debug=True)