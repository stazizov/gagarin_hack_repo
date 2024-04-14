# Руководство по установке

Убедитесь, что у вас установлен git lfs ([мануал по установке](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)):

```
pip install -r requirements.txt
```

# Наши преймущества
## Стиль
1. Код проходит тесты flake8 ✅
2. Код проходит тесты mypy ✅
## Ускорение
1. Сконвертировали в onnx ✅
2. Сконвертировали веса в формат bfloat16 ✅


# Уникальность итогового решния:
1. Предобучение на русскоязычных инвестиционных каналах 
2. Инициализация векторов компаний при помощи tfidf-like штуки
3. Кастомный механизм с подсовыванием векторов компаний в берта, а затем спиливания их эмбедов
4. Эмбеддинги компаний прогоняются через отдельные головы, которые состоят из задних слоев трансформера и стандартного класс. слоя -> быстрее, точнее. Так мы шарим знания между сентиментом и упоминаниями 
5. Лосс для упоминаний - перевзвешенный BCE
6. Лосс для сентимента - фокал
7. Подход с Gumbel softmax + arcface, идея unsupervised NER-а
