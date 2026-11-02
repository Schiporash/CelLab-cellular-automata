# CelLab

Proiect la Inteligență Artificială, FII Iași, grupa A3
Echipa: Alexie Tudor, Schipor Andrei-Alexandru, Vasilica Anca

Aplicație în care descrii un automat celular în câteva propoziții (grila, stările, regulile), iar programul îl simulează și răspunde la întrebări despre el: cum arată după N generații, când devine stabil, dacă intră într-un ciclu etc. Ideea completă e în [`docs/CelLab_Livrabil1.docx`](docs/CelLab_Livrabil1.docx).

## Ce trebuie să avem la Livrabilul 2 (L6)

Modelul automatului, fără interfață grafică deocamdată:

1. pornim de la o descriere text, din care parserul scoate un model (JSON) cu grila, stările, regulile și parametrii experimentului (seed, nr. de generații, procente inițiale);
2. modelul e validat (stări nedeclarate, reguli care se bat cap în cap etc.);
3. se generează starea inițială și se rulează simularea: citim generația curentă, aplicăm regulile și obținem generația următoare;
4. simularea se poate salva și apoi relua de unde a rămas.

Trebuie să meargă pe cel puțin 2-3 automate. Noi le avem pregătite pe toate cinci în `examples/`: Forest Fire, Game of Life, Cyclic CA, S-I-R și Rule 30.

## Cine ce face

| Cine | Partea lui | Stare |
|---|---|---|
| Andrei | parserul și validarea | gata |
| Tudor | simularea: pasul de la o generație la alta (cu `convolve2d` pentru vecini), Rule 30, regulile cu probabilitate (cu seed), mai târziu întrebările și rețeaua neuronală | de făcut |
| Anca | salvarea și încărcarea: modelul în JSON, generațiile în `.npz` (toate sau doar ultima), reluarea unei simulări salvate; mai târziu interfața în PySide6 | de făcut |

## Ce e gata

**Parserul** (`cellab/parser.py`, gramatica în `cellab/grammar.lark`) ia textul propoziție cu propoziție și construiește modelul. Sinonimele sunt înlocuite înainte de parsare (`neighbors`/`neighbours`, `turns into`/`becomes`, `remains`/`stays` etc.). Dacă o propoziție nu e recunoscută, sunt afișate toate propozițiile cu probleme, cu linia și cuvântul unde s-a oprit.

**Validarea** (`cellab/validation.py`) verifică:
- grila are dimensiuni pozitive și există minim 2 stări, fără duplicate;
- toate stările folosite în reguli și în configurația inițială sunt declarate;
- nicio regulă nu cere mai mulți vecini decât există (ex. `at least 9` la 8 vecini);
- nu există două reguli care se contrazic (aceeași stare, condiții care pot fi adevărate simultan, rezultat diferit);
- probabilitățile sunt între 0 și 1, iar procentele inițiale nu trec de 100;
- la Rule 30: o singură linie, 2 stări și tabelul complet cu 8 cazuri.

Pot fi validate și modelele citite dintr-un JSON salvat, nu doar cele obținute din text.

## Rulare

```bash
pip install -r requirements.txt
python -m pytest
python -m cellab examples/forest_fire.txt              # afișează modelul
python -m cellab examples/game_of_life.txt -o gol.json # îl salvează
```

Din cod:

```python
from cellab import load_description, validate, CelLabError

try:
    model = load_description(text)   # parsare + validare
except CelLabError as err:
    for p in err.problems:
        print(p)
```

## Exemplu de descriere

```
The world is a 20x20 grid. The grid wraps around.
Cells can be ALIVE or DEAD.
An ALIVE cell becomes DEAD if fewer than 2 neighbours are ALIVE.
An ALIVE cell becomes DEAD if more than 3 neighbours are ALIVE.
A DEAD cell becomes ALIVE if exactly 3 neighbours are ALIVE.
Initially 30% of cells are ALIVE. The seed is 42. Run for 100 generations.
```

Mai sunt acceptate `Each cell has 4 neighbours`, `... with probability 0.3`, `... if 2 or 3 neighbours are X`, `... if between 2 and 4 neighbours are X`, `... if any neighbour is X`, `The rule is rule 30` sau tabelul `Pattern 110 becomes 1`. Lista completă se vede în gramatică și în teste.

## Formatul modelului

Formatul pe care îl primește simularea (tot ce produce parserul are forma asta):

```json
{"rows": 5, "cols": 5, "neighbours": 8, "border": "fixed",
 "states": ["TREE", "FIRE", "EMPTY"],
 "rules": [
   {"state": "FIRE", "next": "EMPTY"},
   {"state": "TREE", "next": "FIRE", "if": ["FIRE", ">=", 1]}
 ],
 "seed": 42, "generations": 50, "initial": {"TREE": 60, "FIRE": 1}}
```

- `neighbours`: 8 sau 4; `border`: `fixed` sau `wrap`
- `if`: `[stare_vecin, operator, n]`, operator poate fi `>= <= == > <` sau `in` cu o listă (`["ALIVE", "in", [2, 3]]`)
- `prob`: opțional, probabilitatea ca regula să se aplice
- pentru Rule 30 apare `table`: `{"111": 0, "110": 0, ...}`, unde 0/1 e indexul stării din `states`
- `seed`, `generations`, `initial` sunt opționale

Ce trebuie știut când scriem simularea:
- dacă nicio regulă nu se potrivește, celula rămâne cum era;
- validarea garantează că o celulă nu poate avea două reguli deterministe valabile în același timp, deci ordinea regulilor nu contează;
- două reguli cu probabilitate pentru aceeași stare pot exista doar dacă suma probabilităților e ≤ 1, așa că ajunge un singur număr aleator per celulă.

## Structura

```
cellab/      parser, gramatică, validare
examples/    descrierile celor 5 automate
tests/       teste pentru parser și validare
docs/        Livrabilul 1
```
