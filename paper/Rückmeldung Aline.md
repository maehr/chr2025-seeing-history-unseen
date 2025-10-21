## Bezug zur Studie und konkrete Kritik

- **Umfrage UX:**
  - Tastaturnutzung, Fokusreihenfolge, Shortcuts sollten A11y-Standards entsprechen. Sehende sollten mit der Maus bzw. Touchscreen arbeiten können.
  - Zwischenspeichern\*\* sollte möglich sein, um bei Unterbrechungen nicht von vorne zu beginnen.
- **Aufwand zu hoch (600 Urteile):** 100 Bilder × 6 Paarungen ist **zu belastend**. Ermüdung → Rauschen, systematische Dropouts.
- **Redundanz:** Gleiches **Statement 3×** und **Bild 6×** erhöht Frustration und Carry-over-Effekte; gefährdet Unabhängigkeit der Urteile.
- **Vollrandomisierung:** Pros: verteilt Lern-/Reihenfolgeeffekte. Cons: unkontrollierte **Clusterungen** und ungleiche Paarungsfrequenzen; erschwert balanciertes Design.
- **Self-paced + Pausen:** Sinnvoll, aber **im Protokoll nicht spezifiziert** (Autosave, Wiederaufnahme, Timeouts).
- **Intransitive Ränge:** Bei 2AFC sind **Zyklen** erwartbar; Modellierung muss das auffangen (BT ok, aber Varianz ↑ bei Ermüdung).
- **Signifikanz/Power unklar:** **Mindest-N** für gewünschte Präzision der BT-Koeffizienten fehlt; aktuell **unter- oder überdimensioniert**.

## Konsequenzen für das Design

- **Machbarkeits-Pilot** statt Vollstudie:
  - **Items:** ~**20 Bilder**.
  - **Urteilsformat:** **Ranking** statt 2AFC, um **Varianzschätzung** und **Modelldifferenzen** mit weniger Vergleichen zu erhalten.
  - **Stichprobe:** **X Personen** (vorab per Power für ordinales Modell oder gemischtes Linear-Modell bestimmen).
  - **Ziel:** **Modellreduktion** (Top-2 auswählen) und **Varianz-/Reliabilitätsabschätzung**; BT erst in der Hauptstudie.

- **Stimulus-Kontrolle:**
  - **Keine Duplikate** desselben Bilds/Statements im Pilot.
  - **Geblockte Randomisierung** nach Typ/Epoche, nicht voll random.
- **A11y-Spezifikation fürs Survey:**
  - Dokumentierte **Tastenbelegung** (←/→, 1–7), sichtbare **Fokus-Indikatoren**, **ARIA-Labels**, **Speicher-Checkpoint** pro Item, **Fortschrittsbalken**.
- **Ermüdungsmanagement:**
  - **Sessions ≤15 min**, harte **Pausenempfehlung**, Wiederaufnahme-Link.
- **Analyseplan Pilot:**
  - **Interrater:** Kendall’s W oder ICC auf Ratings.
  - **Modelleffekt:** Ordinale gemischte Modelle mit **random rater** und **random item**; Varianzkomponenten schätzen.
  - **Entscheidungskriterium:** Vorab definierte **MCID** in Ratingpunkten oder **OR**-Schwelle; dann Down-select auf 2 Modelle.
- **Power/Min-N:**
  - Vorregistriert: **Ziel-SE** der Modelleffekte oder **Breite des CI**; daraus **N_items × N_raters** ableiten.

## Zusammenfassung der Änderungen am Paper

- Methoden-Abschnitt ergänzen um **Pilot-Phase** mit Ratings, Power-Berechnung, geblockter Randomisierung und A11y-UI-Spezifikation.
- Hauptstudie auf **reduzierte Modellmenge** und ggf. **2AFC** zurückverlagern, nachdem Varianz und Belastung quantifiziert sind.
- Redundanzen im Stimulus-Set entfernen und **Pausen/Autosave** als Teil des Protokolls festschreiben.
