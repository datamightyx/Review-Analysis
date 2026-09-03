# Глибоке порівняння: скоринг програми vs скоринг людини

- **PROG** — `Hamster Sand - 17.08.2026 (1).xlsx` (обробила програма)
- **HUM** — `Hamster sand - 6.07.2026.xlsx` (обробила людина)

---

## 0. Головне застереження: вхідні корпуси РІЗНІ

Файли зроблені по різних наборах товарів. Пряме порівняння обсягів (скільки знайдено відгуків/голосів) **недійсне** — порівнювати можна тільки якість обробки.

| Товар | PROG голоси (Pos) | HUM голоси (Pos) |
|---|---|---|
| Niteangel | **0 (немає взагалі)** | 181 |
| Sukh | 35 (один SKU) | 161 (Sukh white + Sukh yellow) |
| BUCATSTATE | 117 | 38 |
| DR_DUDU | 162 | 202 |
| JFWOD | 196 | 175 |
| Fufuzoie | 124 | 116 |
| Hamiledyi | 124 | 118 |
| Meow&Woof | 19 | 15 |
| Petchardom | 4 | 2 |

- PROG: 8 товарів. HUM: 11 (Niteangel відсутній у PROG, Sukh розділений на 2 SKU).
- Розбіжності йдуть **в обидва боки** (BUCATSTATE у PROG втричі більший, Sukh — вп'ятеро менший) — отже це різні вибірки відгуків, а не «програма загубила дані».
- Перетин дослівних фраз малий: Positive 50 спільних із 308/576 унікальних; Negative 26 із 136/312.

**Висновок:** сумарні цифри (PROG 781 позитивних голосів vs HUM 1008) нічого не доводять.

---

## 1. Гранулярність таксономії

| | PROG | HUM |
|---|---|---|
| Позитивні групи | **39** | 17 |
| Негативні групи | 19 | **21** |
| Usage-групи | 13 (+6 розділів-заголовків) | 17 (плоский список) |
| Improvement-групи | 12 (фактично не згруповано) | 5 |

PROG розбиває позитив утричі дрібніше. Це дає точність, але й породжує сміття:

**Дефекти дроблення в PROG (Positive):**
- **`Litter Box / Potty Use` присутня ДВІЧІ як окрема група** (10 і 3 голоси) — прямий баг дедуплікації USP.
- Семантичні дублі, які мали злитися: `Sand Bathing` (37) vs `Great for Bathing` (2); `Pleasant Scent` (16) vs `Smells Good` (1).
- 12 груп мають ≤3 голоси (`Keeps Pests Away`, `Works as Described`, `Works for Multiple Pets`, `Must Have Product`…) — 25 голосів, 3.2% ваги, але 31% усіх груп.
- **Протікання категорій з листа Usage у Positive**: `Gerbils`, `Craft / Décor Use`, `Litter Box / Potty Use`, `Foraging Enrichment` — це патерни використання, а не USP. У HUM цього нема (крім навмисного `Great as litter box`).

**Дефекти дроблення в HUM (Negative):**
- Тут навпаки — фрагментує людина: 8 груп ≤3 голоси (`Work bad`, `Not American product`, `Bad color`, `Color different than ordered`, `No smell`).
- `No smell` як **негативна** група з 1 голосом — при тому що в Positive є `Pleasant smell/No smell`. Пряма суперечність.

---

## 2. Узгодженість на однакових фразах (найсильніший доказ)

Взяв фрази, дослівно присутні в обох файлах, і порівняв, куди кожен віднесений.

### Positive — 50 спільних фраз
- **33 (66%)** — однакова категорія по суті (`dust free` → Dust Free / Not dusty; `lasts a long time` → Long Lasting / Lasts a long time; `smells good` → Pleasant Scent / Pleasant smell).
- **17 (34%)** — розбіжність. Розподіл вини приблизно рівний:

| Фраза | PROG | HUM | Хто точніший |
|---|---|---|---|
| `clumps the pee very well` | Good Clumping | Easy to clean up | PROG |
| `absorbs wonderfully` | Absorbs Urine Well | Easy to clean up | PROG |
| `its so easy to see when it needs to be cleaned` | Easy to Clean | **Nice color** | PROG |
| `good clean sand for hamster bath` | Sand Bathing | Perfect texture | PROG |
| `the bag is a good size for regular use` | Handy Container | Lasts a long time | PROG |
| `very find sand without dust` | Dust Free | Perfect texture | PROG |
| `its a good size` | Larger Size | Perfect texture | PROG |
| `healthy for my hammies to clean themselves` | Easy to Clean | Safe for hamsters | HUM |
| `it helps keep the hamster clean` | Easy to Clean | Works well for cleaning fur | HUM |
| `much easier to clean their enclosure` | Easy to Clean | Works well for cleaning fur | HUM |
| `very easy to use` | Easy to Clean | Easy to pour and use | HUM |
| `the spout design lets you pour a controlled amount` | Handy Container | Easy to pour and use | HUM |
| `its a lot of sand and it should last me a good while` | Larger Size | Lasts a long time | HUM |

Системна вада PROG: `Easy to Clean` працює як смітник — туди падає і чистка клітки, і чистка шерсті, і зручність використання.
Системна вада HUM: `Perfect texture` — такий самий смітник (туди падають і розмір, і відсутність пилу, і чистота).

### Negative — 26 спільних фраз
- **14 (54%)** — збіг.
- **8 (31%)** — не помилка, а різниця рівня ієрархії: HUM має один кошик `Hallth issues` (73 голоси), PROG розкладає його на `Hamster Died` 21 + `Respiratory Issues` 28 + `Skin/Eye Issues` 9 = 58 (плюс, ймовірно, `Mites/Pests Infestation` 6 — на спільних фразах це не перевірялося). **Перевага PROG** — для товару це різні за критичністю сигнали.
- **3 (12%)** — реальна розбіжність.
- **1 — інверсія змісту в PROG (груба помилка):**
  `it gets solidified into a big piece when my hamster pees` → PROG поставив **`Poor Clumping`** (погано злипається), HUM — `Very clumpy` (занадто злипається). PROG перевернув сенс.

---

## 3. Конкретні помилки, знайдені в кожному файлі

### Помилки PROG
1. **Інверсія сенсу**: `solidified into a big piece` → `Poor Clumping` (див. вище).
2. **Перебільшення тяжкості**: `made my hamster sick` віднесено до **`Hamster Died`**. Хом'як захворів ≠ помер. Штучно роздуває найкритичніший негативний сигнал.
3. **Домислювання**: `my gerbil is sick cause this type of sand` → `Respiratory Issues`. У фразі немає згадки дихальних шляхів.
4. **Дубль групи** `Litter Box / Potty Use` ×2.
5. **Подвійний облік**: 25 пар (фраза + товар) враховані у 2 групах, +27 голосів = 3.5% інфляції позитиву. *Але* перевірено: **усі 25 — крос-групові** (`clean, and pretty sand` → Dust Free + Nice Color), дублів усередині однієї групи — **0**. Тобто це свідома мультиміткова розмітка, а не збій дедуплікації.
6. Категорії використання протікають у лист Positive (п.1).

### Помилки HUM
1. **Орфографія в назвах груп**: `Hallth issues` (73 голоси — найбільший негативний кластер із помилкою в назві), `Prarie dog`, `Chinshilla`, `Hedgehog Approved` як назва зв'язки.
2. **Неконсистентні назви товарів**, які ламають агрегацію: `Dr.Dudu` / `Dr. Dudu` (113 + 7), `Hamiledyi` / `Hamilidyi` (79 + 9). У PROG — єдиний нормалізований `DR_DUDU`, `Hamiledyi`.
3. **Арифметичні помилки в підсумках** (значення забиті руками, не формулами): 4 блоки `Total by product` у Positive не сходяться з сумою рядків — рядки 423 (стоїть 4, треба 5), 433 (6 → 5), 560 (4 → 7), 596 (2 → 3).
4. **Пропущені підсумкові рядки**: 9 блоків `Total by product` мовчки охоплюють два різні товари одночасно. З них **4 — справжнє змішування різних товарів** (рядок 100: Hamiledyi + Fufuzoie; рядок 340: Niteangel + Hamiledyi; рядок 560: Niteangel + Dr. Dudu; рядок 596: Meow&Woof + Sukh yellow), ще **5 — наслідок помилки №2** (два написання одного товару: `Dr.Dudu`/`Dr. Dudu` ×4, `Hamilidyi`/`Hamiledyi` ×1). Тобто «підсумок по товару» місцями неправдивий.
5. **Дублі всередині ОДНІЄЇ групи** — справжній збій дедуплікації: Positive 3 випадки (усі в `Easy to clean up`), Negative 6 (5 із них у `Spilled sand`). Разом **9 груп-дублів, +9 роздутих голосів**. Окремо є ще 8 крос-групових дублів (+9 голосів) — це та сама мультиміткова розмітка, що й у PROG, і помилкою не рахується.
6. **Порожній лист `Products`** — заголовки є, даних нема.
7. Помилки класифікації: `the bag was not resealable` → `Complicated in use` (це дефект упаковки); `it is not good for my chinchilla` → `Animals don't like it` (насправді небезпека для іншого виду).

---

## 4. Механіка й надійність файлу

| Аспект | PROG | HUM |
|---|---|---|
| Підсумки | **живі формули** `=SUM(D2:D12)`, `=SUM(E2:E57)` | хардкод-числа |
| Коректність діапазонів | **0 помилок** із 39+19 груп та всіх продуктових блоків | — |
| Коректність арифметики | — (формули завжди правильні) | **4 розбіжності + 4 змішаних блоки** (+5 блоків зламано різним написанням назви товару) |
| Нормалізація назв товарів | так | ні (4 варіанти написання) |
| Лист Products | заповнений (8 товарів) | порожній |
| Зайві колонки | нема | заявлено 28 колонок (A1:AB896), але за F усе порожнє — просто роздутий діапазон |

Це найчистіша перемога PROG: **жодної арифметичної чи структурної помилки в агрегації**, тоді як у HUM їх 8 прямих (4 арифметичні + 4 змішані блоки) плюс 5 похідних від різного написання назв.

---

## 5. Лист Usage — різні таксономії, не порівнювані напряму

- **PROG**: дворівнева ієрархія з розділами `Animal Type / Usage Behavior / Non-Pet Use / Other / Human Use / Who recommended`. 13 груп, 306 голосів. Ловить *сценарії*: Sand Bathing 33, Litter Box 32, Craft/Décor 7, Sensory Play 1.
- **HUM**: плоский список **лише видів тварин**, 17 груп, 386 голосів. Ловить довгий хвіст видів, який PROG злив у `Other Small Pets` (22 голоси): Chinchilla 6, Hedgehog 4, Quails 3, Mouse 3, Vole, Prairie dog, Rat, Spider, Beetles, Bearded dragon, Rabbit, Lizard, Hermit crabs, Lesser Madagascar Tenrec.

**Взаємодоповнюють.** HUM точніший по видах тварин, PROG — єдиний, хто взагалі витяг патерни використання (купання / туалет / декор / сенсорна гра). PROG злив 14 видів у одну групу `Other Small Pets` — це втрата інформації для товарної стратегії.

---

## 6. Лист Improvement — найслабше місце PROG

| | PROG | HUM |
|---|---|---|
| Групи | 12 | 5 |
| Голоси | 22 | 21 |
| Тип групування | **дослівні цитати, без кластеризації** | канонічні формулювання |

PROG:
```
I wish it were bigger                                6
You'd be better off to go to a pet store and...      3
the packaging could be better                        2
Please please please inspect first before shipping   2
I personally prefer a coarser sand for my hamster    2
I just with it came with a larger quantity...        1   <- дубль "I wish it were bigger" за змістом
wish it was a little cheaper                         1
I will go with a different, more cost effective...   1   <- дубль попереднього за змістом
```
HUM:
```
Bigger amount               8
Reptile sand to avoid dust  7
Small scoop needed          3
Calcium free sand           2
Silica free sand            1
```

PROG на цьому листі фактично **не виконав кластеризацію** — колонка A це назва товару, колонка B — сира цитата. Два змістові дублі «більший об'єм» і два «дешевше» не злиті. HUM дає готові до дії формулювання (`Silica free sand`, `Calcium free sand` — це прямі вимоги до продукту, яких PROG не витяг узагалі).

---

## 7. Підсумок по вимірах

| Вимір | Переможець | Коментар |
|---|---|---|
| Арифметика й цілісність агрегатів | **PROG** | 0 помилок vs 8 прямих (+5 похідних) у HUM |
| Нормалізація назв товарів | **PROG** | HUM має 4 варіанти написання, що ламають підсумки |
| Орфографія назв груп | **PROG** | у HUM `Hallth issues` на найбільшому кластері |
| Деталізація негативу (health-розкладка) | **PROG** | Died / Respiratory / Skin / Mites замість одного кошика |
| Патерни використання (Usage behavior) | **PROG** | HUM не витяг узагалі |
| Точність значення окремої фрази | **нічия** | 66% збіг; помилки з обох боків приблизно рівно |
| Дисципліна таксономії (позитив) | **HUM** | PROG дає 12 сміттєвих груп + дубль групи + протікання Usage |
| Дисципліна таксономії (негатив) | **PROG** | тут фрагментує вже HUM (8 груп ≤3 голоси) |
| Довгий хвіст видів тварин | **HUM** | PROG зліпив 14 видів у `Other Small Pets` |
| Кластеризація Improvement | **HUM** | PROG фактично не кластеризував |
| Відсутність фактичних помилок сенсу | **HUM** | у PROG інверсія `Poor Clumping` і «sick → Died» |

**Загальний висновок:** програма ще не замінює людину, але вже сильніша там, де людина систематично помиляється — механіка, нормалізація, арифметика, деталізація критичного негативу. Людина сильніша там, де потрібне судження: дисципліна назв категорій, злиття семантичних дублів, кластеризація побажань, довгий хвіст видів.

---

## 8. Що конкретно правити в програмі (за пріоритетом)

1. **Дедуплікація USP за назвою** — `Litter Box / Potty Use` не мала з'явитися двічі.
2. **Злиття семантичних дублів груп** перед експортом: `Sand Bathing`+`Great for Bathing`, `Pleasant Scent`+`Smells Good`.
3. **Поріг на хвіст**: групи з 1 голосом (`Works as Described`, `Keeps Pests Away`) або зливати в `Other`, або відсікати.
4. **Заборонити протікання Usage-категорій у Positive** — `Gerbils`, `Craft / Décor Use`, `Foraging Enrichment` мають жити тільки в Usage.
5. **Кластеризація листа Improvement** — зараз її нема взагалі; це найбільша функціональна прогалина.
6. **Полярність у Negative**: розділити «не злипається» і «злипається в камінь» — зараз обидва падають у `Poor Clumping`.
7. **Не підвищувати тяжкість**: «sick» не має мапитися в `Hamster Died`; додати окрему групу `Pet Got Sick`.
8. **Не домислювати механізм**: `sick` без згадки дихання не має ставати `Respiratory Issues`.
9. **Не зливати види тварин** у `Other Small Pets` — виводити кожен вид окремо, як у HUM (це прямий сигнал для розширення асортименту).
