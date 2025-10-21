**Setting**

- 20 Bilder
- 4 LLM generieren je eine Beschreibung
- Jede Versuchsperson rankt die 4 Beschreibungen pro Bild vollständig (1–4)
- Ziel: mittlere Leistungsbewertung der LLM über alle Bilder hinweg („bestes Modell“)

**Empfohlenes Modell**
→ **Hierarchisches Plackett–Luce-Modell (HPLM)**
[
\text{logit}(\Pr(\text{Rangfolge})) = \lambda_j + u_i + v_k
]
mit festen Effekten (\lambda_j) für LLM, Zufallseffekten (u_i) für Bild und (v_k) für Person.
Ergebnis: Worth-Parameter (\lambda_j) mit Konfidenz- oder Posteriorintervallen → bestes Modell = höchster (\lambda_j).

**Empfohlenes Design**

- Volle Rankings pro Bild nutzen (kein Top-1-Verlust).
- Zufallseffekte für Bild und Person, um Varianz zu trennen.
- Likelihood- oder Bayes-Fit (z. B. _R: PlackettLuce_, _brms_; _Python: choicest_, _PyMC_).
- Unsicherheitsintervalle statt bloßer Punktwerte berichten.
- Mehrfachtests via FDR statt Bonferroni.

**Empfohlene Stichprobengrösse**

- Mittelstarker Effekt (≈ 30–32 % Top-1 vs. 22–23 %) → **15–25 Teilnehmende**.
- Grosser Effekt (≈ 40 % vs. 20 %) → **6–10 Teilnehmende**.
- Kleine Effekte (< 28 % vs. 24 %) → **> 50 Teilnehmende**.
- Bei erwarteter Antwortkorrelation ρ≈0.05 Zahl × 2.
- 20 Bilder liefern starke Informationsreduktion des N-Bedarfs.

**Kurzempfehlung**
Für robuste Differenzierung der 4 LLM über 20 Bilder:
→ **Mindestens 20 Teilnehmende**, hierarchisches Plackett–Luce-Modell mit Bild- und Personen-Random-Effekten, Auswertung der Worth-Parameter mit 95 %-Intervallen und FDR-Korrektur.
