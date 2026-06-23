# Tőkekalkuláció és kockázati modellek: Bázel kontra kopula

Ez a projekt Hodosi Máté Budapesti Corvinus Egyetemen (Gazdaság- és pénzügy-matematikai elemzés mesterszak) írt szakdolgozatának teljes, reprodukálható kódbázisát tartalmazza. 

A projekt egy komplex, empirikus adatokat használó GARCH-EVT-Kopula-VaR szimulációs motort valósít meg, amelynek célja a banki portfóliók farokeloszlás-fertőzésének (tail contagion) és hálózati diverzifikációs potenciáljának mérése, majd ennek összevetése a statikus Bázeli (ASRF) tőkekövetelmény-számítási módszertannal.

## A projekt felépítése

A kódbázis moduláris felépítésű, a főbb funkcionalitásokat az alábbi fájlok látják el:

* **`master_file.py`**: A szimuláció fő vezérlő szkriptje. Ez a fájl hívja meg a statisztikai modulokat, hajtja végre a Monte Carlo szimulációt, és állítja elő mind a részeredményeket, mind a végső tőkekalkulációt.
* **`tools.py`**: Az empirikus elemzés motorja. Ez a fájl modulárisan tartalmazza a kutatás során használt ökonometriai és statisztikai eszközöket (pl. GARCH illesztése, EVT küszöbértékek számítása, kopulafüggőségek optimalizálása).
* **`capital_comparison.py`**: Vizualizációs modul. Feladata a diverzifikált és diverzifikálatlan belső modellezési eredmények, valamint a bázeli tőkekövetelmények összehasonlításához szükséges végső oszlopdiagram elkészítése.
* **`requirements.txt`**: A futtatáshoz szükséges Python programcsomagok és azok verzióinak listája.
* **`INIT`**: A letisztított kezdeti CDS felárakból implikált PD idősorok. 

## Letöltés és futtatás

A kutatás eredményeinek reprodukálásához az alábbi lépések szükségesek:

1. **A kód és az adatok letöltése:**
    Látogasson el a [hodosimate.github.io](https://hodosimate.github.io) weboldalra, és töltse le a teljes kódbázist, valamint a kiindulási adatokat tartalmazó tömörített (.zip) fájlt.
   
    Miután letöltötte, csomagolja ki a fájlt egy tetszőleges mappába a számítógépén, majd nyissa meg ezt a mappát a parancssorban (terminálban):

```bash
cd a-kicsomagolt-mappa-pontos-utvonala
```

2. **Szükséges csomagok telepítése:**

```bash
pip install -r requirements.txt
```

3. **Fontos megjegyzés a futtatási időről:**
    A modell a szubadditivitás empirikus igazolása és a farokeloszlás-régiók mintavételi zajának kiküszöbölése érdekében 100 000 iterációs Monte Carlo szimulációt hajt végre. Ez a robusztus beállítás jelentős számítási kapacitást igényel.

    A master_file.py teljes lefutása egy átlagos teljesítményű számítógépen nagyságrendileg 4 órát vesz igénybe.

    (Tipp: Amennyiben csak a kód működését vagy az adatok betöltését szeretné tesztelni, a master_file.py fájlban az iterációszám paramétere a futtatás előtt manuálisan csökkenthető.)

4. **A szimuláció elindítása:**
```bash
python master_file.py
```

5. **Várt eredmények:**
    A szkript sikeres lefutása után a konzolon megjelennek a portfóliószintű diverzifikált és diverzifikálatlan VaR mutatók, valamint a capital_comparison.py legenerálja és elmenti a dolgozat Diszkusszió fejezetében is szereplő végső összehasonlító diagramot.
