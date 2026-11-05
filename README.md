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
| Tudor | simularea: pasul de la o generație la alta (cu `convolve2d` pentru vecini), Rule 30, regulile cu probabilitate (cu seed); mai târziu întrebările și rețeaua neuronală | simularea gata |
| Anca | salvarea și încărcarea: modelul în JSON, generațiile în `.npz` (toate sau doar ultima), reluarea unei simulări salvate; mai târziu interfața în PySide6 | salvarea gata |

## Ce e gata

**Parserul** (`cellab/parser.py`, gramatica în `cellab/grammar.lark`) ia textul propoziție cu propoziție și construiește modelul. Sinonimele sunt înlocuite înainte de parsare (`neighbors`/`neighbours`, `turns into`/`becomes`, `remains`/`stays` etc.). Dacă o propoziție nu e recunoscută, sunt afișate toate propozițiile cu probleme, cu linia și cuvântul unde s-a oprit.

**Validarea** (`cellab/validation.py`) verifică:
- grila are dimensiuni pozitive și există minim 2 stări, fără duplicate;
- toate stările folosite în reguli și în configurația inițială sunt declarate;
- nicio regulă nu cere mai mulți vecini decât există (ex. `at least 9` la 8 vecini);
- nu există două reguli care se contrazic (aceeași stare, condiții care pot fi adevărate simultan, rezultat diferit);
- probabilitățile sunt între 0 și 1, iar procentele inițiale nu trec de 100;
- regulile cu probabilitate care pot fi valabile în același timp pentru o celulă nu au suma probabilităților mai mare de 1;
- la Rule 30: o singură linie, 2 stări și tabelul complet cu 8 cazuri.

Pot fi validate și modelele citite dintr-un JSON salvat, nu doar cele obținute din text.

**Simularea** (`cellab/simulation.py`):
- grila e un array NumPy cu indexul stării fiecărei celule (0 = prima stare declarată);
- vecinii se numără pentru toată grila odată cu `convolve2d` (8 sau 4 vecini); la marginea `fixed` celulele din afara grilei nu contează ca vecini, la `wrap` grila se continuă pe partea cealaltă;
- starea inițială: dacă sunt date procente, fiecare stare primește atâtea celule (rotunjit, minim una dacă procentul nu e 0), puse aleator; restul celulelor primesc prima stare declarată care nu are procent (la Game of Life: DEAD). Dacă nu e dat niciun procent, toată grila e în prima stare și celula din mijloc în a doua (un singur foc în pădure, o singură celulă ON la Rule 30), ca în Livrabilul 1;
- regulile cu probabilitate folosesc un singur număr aleator per celulă și generație, dintr-un generator pornit cu `seed`, deci același model cu același seed dă mereu aceeași simulare;
- Rule 30 (și orice regulă elementară dată prin tabel) se calculează din tabelul cu 8 cazuri; la marginea `fixed` celulele din afară sunt în starea 0.

**Salvarea și reluarea** (`cellab/storage.py`): o simulare se salvează în două fișiere, `nume.json` (modelul, generația curentă, ce generații s-au păstrat și starea generatorului aleator) și `nume.npz` (generațiile, toate sau doar ultima). La încărcare modelul e validat din nou și se verifică dacă generațiile se potrivesc cu el. Pentru că se salvează și starea generatorului, o simulare reluată continuă exact ca și cum nu s-ar fi oprit (inclusiv la S-I-R, care e cu probabilități).

## Rulare

```bash
pip install -r requirements.txt
python -m pytest
python -m cellab examples/forest_fire.txt              # afișează modelul
python -m cellab examples/game_of_life.txt -o gol.json # îl salvează
python -m cellab examples/rule30.txt --run             # simulează (nr. de generații din descriere)
python -m cellab examples/sir_epidemic.txt --run 50 --save sir   # sir.json + sir.npz
python -m cellab --resume sir --run 150                # continuă de la generația 50
python -m cellab --resume sir --from 20 --run 30       # se întoarce la generația 20 și continuă de acolo
python -m cellab examples/cyclic.txt --run 500 --save cyc --keep last   # doar ultima generație
python -m cellab examples/forest_fire.txt --grid examples/forest_fire_l1.grid --run 4   # pornește de la grila din fișier
python -m cellab examples/forest_fire.txt --grid examples/forest_fire_l1.grid --run 4 --show   # afișează fiecare generație
python -m cellab --run 3 --show                        # scrii descrierea în terminal, un rând gol la final
```

`--from` merge doar cu generațiile păstrate, deci nu și după `--keep last`, iar fișierele salvate nu se schimbă decât dacă folosești `--save` cu același nume. La automatele fără probabilități ajungi din nou la aceleași generații. La cele cu probabilități (S-I-R) simularea o ia pe alt drum, pentru că generatorul aleator e salvat doar pentru ultima generație, nu pentru fiecare.

Fișierul pentru `--grid` are câte o linie pentru fiecare rând și un caracter pentru fiecare celulă: prima literă a stării (`T`, `F`, `E`) sau exact ce afișează programul, deci `.` și `#` la automatele cu două stări și cifre dacă stările încep cu aceeași literă (`S0`, `S1` etc.). Atenție: `#` e a doua stare declarată, iar la Game of Life aia e `DEAD`, așa că acolo e mai clar cu `A` și `D`. `examples/forest_fire_l1.grid` e pornirea din figura din Livrabilul 1.

Din cod:

```python
from cellab import load_description, CelLabError, Simulation, load_grid, save_simulation, load_simulation

try:
    model = load_description(text)   # parsare + validare
except CelLabError as err:
    for p in err.problems:
        print(p)

sim = Simulation(model)        # generația 0
# sau de la o grilă din fișier: Simulation(model, grid=load_grid("start.grid", model))
sim.run(50)                    # încă 50 de generații (fără argument: cât scrie în descriere)
sim.grid, sim.generation       # unde a ajuns
sim.history                    # toate generațiile de până acum
sim.counts()                   # {"ALIVE": 37, "DEAD": 363}
print(sim.render())            # grila ca text

save_simulation(sim, "gol", keep="all")   # sau keep="last"
sim = load_simulation("gol")              # se poate continua cu sim.run(...)
```

Funcțiile separate, dacă e nevoie de ele (ex. pentru întrebări sau pentru setul de date al rețelei): `initial_grid(model)`, `step(grid, model, rng)` și `neighbour_counts(grid, stare, vecini, margine)`.

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
cellab/      parser, gramatică, validare, simulare, salvare
examples/    descrierile celor 5 automate
tests/       teste pentru parser, validare, simulare și salvare
docs/        Livrabilul 1
```
