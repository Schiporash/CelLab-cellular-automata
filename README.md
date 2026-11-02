# CelLab – automate celulare descrise în limba engleză

Proiect AI, Facultatea de Informatică Iași, grupa A3.
Echipa: **Alexie Tudor** (simulare, întrebări, rețea neuronală), **Schipor Andrei-Alexandru** (parser și validare), **Vasilica Anca** (interfață, salvarea simulărilor).

Utilizatorul descrie un automat celular în engleză (grila, stările, regulile), aplicația transformă textul într-un model JSON, îl validează, simulează generațiile și răspunde la întrebări despre simulare. Detalii în [`docs/CelLab_Livrabil1.docx`](docs/CelLab_Livrabil1.docx).

## Plan

| Laborator | Ce livrăm | Stare |
|---|---|---|
| L3 | Echipa și soluțiile tehnice (Livrabil 1) | ✅ |
| L6 | Modelul automatului: starea lumii și regulile, parametrii experimentului, salvare/încărcare; testat pe 2-3 automate (Livrabil 2) | în lucru: parserul și validarea sunt gata |
| L11 | Set de date de generații + experiment NN/RL (Livrabil 3) | |
| L13 | Parserul pentru descrierea în engleză | ✅ (făcut mai devreme, e necesar pentru L6) |
| L14 | Prezentarea finală | |

## Instalare

```bash
pip install -r requirements.txt
python -m pytest            # rulează testele
python -m cellab examples/forest_fire.txt            # afișează modelul JSON
python -m cellab examples/game_of_life.txt -o gol.json
```

Dacă descrierea are probleme, comanda le afișează pe toate (nu doar prima) și iese cu codul 1:

```
DescriptionError:
  - line 3: "A TREE cell explodes" -> I don't understand the part starting at 'explodes'
ValidationError:
  - rule 'FIRE -> ASH' uses state ASH, which was not declared
  - rule 'TREE -> FIRE if FIRE >= 9' can never apply: a cell has only 8 neighbours
```

## Folosire din cod

```python
from cellab import load_description, validate, CelLabError

try:
    model = load_description(text)      # parsare + validare
except CelLabError as err:
    for problem in err.problems:        # problem.sentence, problem.line, problem.message
        print(problem)

problems = validate(json.load(f))       # validează și un model încărcat dintr-un fișier salvat
```

## Ce propoziții înțelege parserul

Fiecare propoziție se termină cu punct sau rând nou; `#` începe un comentariu. Literele mari/mici nu contează (stările sunt convertite la MAJUSCULE). Numerele pot fi scrise și cu litere (`one` ... `ten`).

| Ce | Exemple |
|---|---|
| Grila | `The world is a 5x5 grid.` · `The grid is 1 by 61.` |
| Stările | `Cells can be TREE, FIRE or EMPTY.` · `The states are A, B and C.` |
| Vecinătatea (implicit 8) | `Each cell has 4 neighbours.` · `The neighbourhood is von Neumann / Moore.` |
| Marginile (implicit fixe) | `The grid wraps around.` · `The border is fixed / toroidal / periodic.` |
| Regulă | `A FIRE cell becomes EMPTY.` · `An EMPTY cell stays EMPTY.` · `A B cell stays the same.` |
| Condiție | `... if at least / at most / exactly / more than / fewer than 2 neighbours are X` · `... if 2 or 3 neighbours are X` · `... if between 2 and 4 neighbours are X` · `... if any neighbour is X` · `... if no neighbours are X` |
| Probabilitate | `... with probability 0.3` · `... with probability 30%` (înainte sau după condiție) |
| Automat 1D (Wolfram) | `The rule is rule 30.` / `Use rule 30.` sau tabelul: `Pattern 110 becomes 1.` (8 propoziții) |
| Parametri | `The seed is 42.` · `Run for 100 generations.` · `Initially 60% of cells are TREE and 1% are FIRE.` |

Sinonime acceptate: `neighbors/neighbours`, `turns into / changes to / transforms into / becomes`, `remains / stays / is unchanged`, `less than / fewer than`, `greater than / more than`, `wraps / toroidal / periodic`, `steps / iterations / generations`, `percent / %`.

## Formatul modelului (JSON)

Exemplul Forest Fire din livrabil:

```json
{"rows": 5, "cols": 5, "neighbours": 8, "border": "fixed",
 "states": ["TREE", "FIRE", "EMPTY"],
 "rules": [
   {"state": "FIRE", "next": "EMPTY"},
   {"state": "TREE", "next": "FIRE", "if": ["FIRE", ">=", 1]},
   {"state": "EMPTY", "next": "EMPTY"}
 ]}
```

| Cheie | Semnificație |
|---|---|
| `rows`, `cols` | dimensiunea grilei |
| `neighbours` | `8` (Moore) sau `4` (von Neumann) |
| `border` | `"fixed"` sau `"wrap"` (grila se continuă pe partea cealaltă) |
| `states` | lista stărilor; indexul din listă poate fi folosit ca valoare în grila NumPy |
| `rules[].state` / `rules[].next` | starea curentă a celulei / starea în generația următoare |
| `rules[].if` | opțional: `[STARE_VECIN, op, n]`, cu `op` ∈ `>= <= == > <` sau `"in"` cu o listă de valori, ex. `["ALIVE", "in", [2, 3]]` |
| `rules[].prob` | opțional: probabilitatea ca regula să se aplice (0 < p ≤ 1) |
| `table` | doar la automate 1D: `{"111": 0, "110": 0, ...}`; 0/1 = indexul stării din `states` |
| `wolfram` | numărul regulii, dacă a fost dat (`table` e deja calculat din el) |
| `seed`, `generations` | parametri de experiment (opționali) |
| `initial` | procent pentru fiecare stare, ex. `{"TREE": 60, "FIRE": 1}` |

**Convenții pentru simulare:** o celulă pentru care nicio regulă nu se aplică își păstrează starea. Validarea garantează că cel mult o regulă deterministă se poate aplica unei celule (deci ordinea regulilor nu contează); două reguli probabiliste cu rezultate diferite se pot suprapune doar dacă suma probabilităților e ≤ 1 (se trage un singur număr aleator).

## Ce verifică validarea

- dimensiunea grilei există și e pozitivă; sunt cel puțin două stări, fără duplicate;
- orice stare folosită într-o regulă, condiție sau în configurația inițială a fost declarată;
- nicio condiție nu cere mai mulți vecini decât are o celulă (ex. `at least 9` cu 8 vecini);
- probabilitățile sunt în (0, 1], procentele inițiale însumează cel mult 100;
- **nu există reguli care se contrazic**: aceeași stare, o vecinătate în care ambele condiții sunt adevărate și rezultate diferite (ex. `A→B if at least 4 B` și `A→C if at least 4 C` se contrazic, pentru că 4+4 ≤ 8 vecini);
- automatele 1D au o singură linie, exact două stări și tabelul complet cu cele 8 cazuri.

## Structura

```
cellab/
  grammar.lark     gramatica Lark (o propoziție)
  normalize.py     împărțirea în propoziții + înlocuirea sinonimelor
  parser.py        text -> model (dict / JSON)
  validation.py    verificările de mai sus
  errors.py        Problem, DescriptionError, ValidationError
  __main__.py      linia de comandă: python -m cellab fisier.txt
examples/          Forest Fire, Game of Life, Cyclic CA, S-I-R, Rule 30 (număr și tabel)
tests/             teste pytest pentru parser și validare
```
