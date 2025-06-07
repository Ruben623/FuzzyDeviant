import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def setup_fuzzy_system():
    universe = np.arange(0, 10.1, 0.1)

    # Входные переменные
    aggression = ctrl.Antecedent(universe, 'aggression')
    delinquency = ctrl.Antecedent(universe, 'delinquency')
    norms = ctrl.Antecedent(universe, 'norms')
    autoaggression = ctrl.Antecedent(universe, 'autoaggression')
    manipulation = ctrl.Antecedent(universe, 'manipulation')

    # Настраиваем функции принадлежности
    for var in [aggression, delinquency, norms, autoaggression, manipulation]:
        var['low'] = fuzz.zmf(var.universe, 2, 4)
        var['medium'] = fuzz.gaussmf(var.universe, 5, 1.2)
        var['high'] = fuzz.smf(var.universe, 6, 8)

    # Выходная переменная
    deviance = ctrl.Consequent(universe, 'deviance')
    deviance['low'] = fuzz.zmf(deviance.universe, 1.5, 3.5)
    deviance['medium'] = fuzz.gaussmf(deviance.universe, 5, 1.5)
    deviance['high'] = fuzz.smf(deviance.universe, 6.5, 8.5)

    # Правила системы
    rules = [
        # Девиантность явно выражена
        ctrl.Rule(aggression['high'] & delinquency['high'], deviance['high']),
        ctrl.Rule(aggression['high'] & norms['high'], deviance['high']),
        ctrl.Rule(aggression['high'] & autoaggression['high'], deviance['high']),
        ctrl.Rule(aggression['high'] & manipulation['high'], deviance['high']),
        ctrl.Rule(delinquency['high'] & norms['high'], deviance['high']),
        ctrl.Rule(delinquency['high'] & autoaggression['high'], deviance['high']),
        ctrl.Rule(delinquency['high'] & manipulation['high'], deviance['high']),
        ctrl.Rule(norms['high'] & autoaggression['high'], deviance['high']),
        ctrl.Rule(norms['high'] & manipulation['high'], deviance['high']),
        ctrl.Rule(autoaggression['high'] & manipulation['high'], deviance['high']),

        # Критические комбинации с аутоагрессией
        ctrl.Rule(autoaggression['high'] & aggression['medium'], deviance['high']),
        ctrl.Rule(autoaggression['high'] & delinquency['medium'], deviance['high']),
        ctrl.Rule(autoaggression['high'] & norms['high'], deviance['high']),

        # Агрессия + манипуляции
        ctrl.Rule(aggression['high'] & manipulation['medium'], deviance['high']),
        ctrl.Rule(aggression['medium'] & manipulation['high'], deviance['high']),

        # Тройные комбинации
        ctrl.Rule(aggression['high'] & delinquency['high'] & norms['medium'], deviance['high']),
        ctrl.Rule(aggression['high'] & manipulation['high'] & autoaggression['medium'], deviance['high']),

        # Умеренная девиантность
        ctrl.Rule(aggression['medium'] & delinquency['medium'], deviance['medium']),
        ctrl.Rule(aggression['medium'] & norms['medium'], deviance['medium']),
        ctrl.Rule(aggression['medium'] & manipulation['medium'], deviance['medium']),
        ctrl.Rule(delinquency['medium'] & norms['medium'], deviance['medium']),

        # Один высокий показатель
        ctrl.Rule(aggression['high'] & delinquency['low'] & norms['low'], deviance['medium']),
        ctrl.Rule(delinquency['high'] & manipulation['low'] & autoaggression['low'], deviance['medium']),

        # Комбинации с аутоагрессией
        ctrl.Rule(autoaggression['medium'] & aggression['low'], deviance['medium']),
        ctrl.Rule(autoaggression['medium'] & delinquency['low'], deviance['medium']),

        # Нормальное поведение
        ctrl.Rule(aggression['low'] & delinquency['low'] & norms['low'] & autoaggression['low'] & manipulation['low'], deviance['low']),
        ctrl.Rule(aggression['low'] & delinquency['low'] & norms['medium'], deviance['low']),
        ctrl.Rule(aggression['low'] & manipulation['low'] & autoaggression['medium'], deviance['low']),
        ctrl.Rule(autoaggression['high'] & aggression['low'] & delinquency['low'], deviance['low']),

        # Крайние случаи
        ctrl.Rule(autoaggression['high'] & aggression['low'], deviance['medium']),
        ctrl.Rule(manipulation['high'] & delinquency['low'], deviance['medium']),

        # Приоритетные правила
        ctrl.Rule(
            (aggression['high'] | delinquency['high']) &
            (norms['high'] | manipulation['high']) &
            (autoaggression['medium'] | autoaggression['high']),
            deviance['high']
        )
    ]

    deviance_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(deviance_ctrl)

def calculate_deviance(scores, max_scores):
    # Соответствие русских названий английским переменным
    category_mapping = {
        'Агрессивное и рискованное поведение': 'aggression',
        'Делинквентное (противоправное) поведение': 'delinquency',
        'Демонстративное игнорирование социальных норм': 'norms',
        'Аутоагрессия и депрессивные тенденции': 'autoaggression',
        'Манипулятивное и аморальное поведение': 'manipulation'
    }

    # Нормализация баллов к шкале 0-10
    normalized_eng_scores = {}
    normalized_ru_scores = {}
    
    for rus_name, eng_name in category_mapping.items():
        # Вычисляем нормализованное значение
        value = (scores[rus_name] / max_scores[rus_name]) * 10
        
        # Сохраняем с английскими ключами для нечеткой логики
        normalized_eng_scores[eng_name] = value
        
        # Сохраняем с русскими ключами для отображения
        normalized_ru_scores[rus_name] = value

    # Расчет нечеткого вывода
    sim = setup_fuzzy_system()
    for eng_name, value in normalized_eng_scores.items():
        sim.input[eng_name] = value
    sim.compute()

    # Возвращаем только числовые данные
    return {
        'value': sim.output['deviance'],
        'normalized_scores': normalized_ru_scores
    }

def generate_combined_plot(normalized_scores):
    categories = list(normalized_scores.keys())
    values = list(normalized_scores.values())
    
    # Сокращенные названия для отображения
    short_names = {
        'Агрессивное и рискованное поведение': 'Агрессия',
        'Делинквентное (противоправное) поведение': 'Противоправное',
        'Демонстративное игнорирование социальных норм': 'Игнор. норм',
        'Аутоагрессия и депрессивные тенденции': 'Аутоагрессия',
        'Манипулятивное и аморальное поведение': 'Манипуляции'
    }
    
    # Создаем фигуру с увеличенным размером
    plt.rcParams.update({'font.size': 14})  # Базовый размер шрифта
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))  # Увеличиваем размер графиков
    
    # 1. Бар-чарт
    bars = ax1.bar([short_names[cat] for cat in categories], values, 
                  color=['#e74c3c', '#2980b9', '#2ecc71', '#f1c40f', '#9b59b6'])
    ax1.set_ylim(0, 10)
    ax1.set_ylabel('Уровень (0-10)', fontsize=20)  # Увеличиваем подпись оси Y
    ax1.set_title('Нормализованные показатели', fontsize=30, pad=30)  # Увеличиваем заголовок
    
    # Увеличиваем подписи значений
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=20)
    
    # 2. Радарная диаграмма
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax2 = plt.subplot(122, polar=True)
    ax2.fill(angles, values, color='#3498db', alpha=0.25)
    ax2.plot(angles, values, color='#3498db', marker='o', linewidth=2)
    
    # Увеличиваем подписи на радаре
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels([short_names[cat].split()[0] for cat in categories], 
                       fontsize=20)
    ax2.set_yticks(np.arange(0, 11, 2))
    ax2.set_yticklabels(np.arange(0, 11, 2), fontsize=20)  # Увеличиваем цифры на осях
    ax2.set_title('Профиль девиантности', fontsize=30, pad=30)
    
    plt.tight_layout()
    
    # Сохраняем с увеличенным DPI
    img = BytesIO()
    plt.savefig(img, format='png', dpi=120)
    img.seek(0)
    plot_data = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_data