**Kurzantwort:**  
Ein guter Alternativtext (Alt-Text) beschreibt _funktional und kontextsensitiv_, was ein Bild für den Inhalt, Zweck oder die Interaktion der Seite leistet. Er darf weder redundant noch dekorativ sein. Grundlage sind die **WCAG 2.2 (Success Criterion 1.1.1: Non-text Content)** und die **W3C WAI-Authoring Practices**.

---

### 1. Regelwerk (WCAG 2.2, WAI)

**Relevante Richtlinie:**

- **1.1.1 Non-text Content (A-Level):**  
   Alle nicht-textuellen Inhalte müssen ein textliches Äquivalent besitzen, _es sei denn_, sie sind rein dekorativ, Layout-Elemente oder redundante Repräsentationen bereits vorhandener Information.

**W3C-WAI konkretisiert:**

- [W3C WAI: “Alternative Text”](https://www.w3.org/WAI/tutorials/images/)
  - Beschreibe _Bedeutung, nicht Aussehen_.
  - Passe Beschreibung an den _Kontext_ an.
  - Verwende _leeren Alt-Text (`alt=""`)_ für rein dekorative Bilder.
  - Vermeide „Bild von“, „Grafik von“, „Foto von“.
  - Funktionale Bilder (Buttons, Links) beschreiben _Ziel oder Funktion_ („Suche starten“ statt „Lupe“).
  - Komplexe Grafiken (Diagramme, Karten) benötigen eine _ausführlichere Beschreibung_ im umgebenden Text oder über `longdesc`, `aria-describedby`, oder separate HTML-Abschnitte.

---

### 2. Qualitätskriterien für guten Alt-Text

| Aspekt                  | Richtlinie                                                                  | Beispiel                                                                                              |
| ----------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Relevanz**            | Nur beschreiben, was für den Inhalt relevant ist.                           | _„Statistik zur CO₂-Entwicklung in Europa“_, nicht _„blaue Balkendiagramme auf weissem Hintergrund“_. |
| **Kürze**               | So knapp wie möglich, so lang wie nötig.                                    | Meist < 125 Zeichen.                                                                                  |
| **Kontextsensitivität** | Bedeutung hängt vom Dokument ab.                                            | Dasselbe Foto kann „Forscherteam an der ETH“ oder „Beispiel für Laborarbeit“ sein.                    |
| **Funktionalität**      | Wenn Bild klickbar oder interaktiv: Zweck benennen.                         | `alt="Zum Warenkorb"`                                                                                 |
| **Dekorativität**       | Wenn rein ästhetisch: leerer Alt-Text (`alt=""`) und `role="presentation"`. | Linien, Rahmen, Platzhalter.                                                                          |

---

### 3. Technische Umsetzung

```html
<!-- Bedeutungstragendes Bild -->
<img src="eth-lab.jpg" alt="Forscherteam im ETH-Labor untersucht Sensorprototypen" />

<!-- Funktionales Bild -->
<a href="/cart"><img src="cart-icon.svg" alt="Zum Warenkorb" /></a>

<!-- Dekoratives Bild -->
<img src="divider.png" alt="" role="presentation" />
```

---

### 4. Methodische Hinweise

- **Test:** Screenreader prüfen (NVDA, VoiceOver). Wenn der Textfluss verständlich bleibt, ist der Alt-Text gelungen.
- **Team-Praxis:** Text- und Webredaktion sollten Alt-Texte gemeinsam mit Inhalts- und Barrierefreiheitsrichtlinien erfassen.
- **Automatische Generierung:** maschinell erzeugte Beschreibungen sind _hilfreich, aber nicht konform_ nach WCAG, solange sie nicht redaktionell geprüft werden.

---

### 5. Fazit

Ein guter Alternativtext folgt drei Prinzipien:

1. **Semantik:** Was bedeutet das Bild in diesem Kontext?
2. **Relevanz:** Welche Information fehlt ohne das Bild?
3. **Ökonomie:** Nur das Nötige, nichts Redundantes.

Für wissenschaftliche und museale Kontexte empfiehlt sich zusätzlich eine **strukturierte Bildbeschreibungspraxis** (z. B. über Metadatenfelder wie `dc:description` oder IIIF-Annotations).

---

Möchtest du eine kompakte Checkliste (z. B. für Redaktionsrichtlinien oder Schulung)?

**Prompt (für vLLM oder ähnliche Modelle):**

> Schreibe barrierefreie Alternativtexte für Bilder nach WCAG 2.2 (SC 1.1.1) und W3C WAI-Standards.
>
> **Ziel:** Funktionale, kontextabhängige, prägnante Alt-Texte.
>
> **Regeln:**
>
> - Beschreibe Bedeutung, nicht Aussehen.
> - Max. 125 Zeichen, nur was für Inhalt/Zweck relevant ist.
> - Für dekorative Bilder: `alt=""`.
> - Für funktionale Bilder: Funktion oder Ziel nennen („Zum Warenkorb“).
> - Keine Formulierungen wie „Bild von“ oder „Foto von“.
> - Für komplexe Grafiken: Kurzbeschreibung als Alt-Text + ausführliche Erklärung im umgebenden Text oder via `aria-describedby`.
>
> **Ausgabe:** Nur der Alternativtext (kein HTML, keine Erklärungen).
>
> **Eingabe:**
>
> - Bildbeschreibung oder Metadaten
> - Nutzungskontext (z. B. wissenschaftlich, redaktionell, UI-Element)

Hier sind zentrale Originalpassagen aus **WCAG 2.2 / WCAG** (SC 1.1.1) und aus W3C-WAI, in englischer Originalversion, mit Referenzen:

---

## WCAG 2.2 — Success Criterion 1.1.1 Non-text Content

> “All non-text content that is presented to the user has a text alternative that serves the equivalent purpose, except for the situations listed below.”  
> — Web Content Accessibility Guidelines (WCAG) 2.2, SC 1.1.1 ([W3C](https://www.w3.org/TR/WCAG22/?utm_source=chatgpt.com 'Web Content Accessibility Guidelines (WCAG) 2.2 - W3C'))

Weitere Erläuterung im „Understanding“-Dokument:

> “The intent of this Success Criterion is to make information conveyed by non-text content accessible through the use of a text alternative.”  
> — Understanding Success Criterion 1.1.1: Non-text Content (W3C) ([W3C](https://www.w3.org/WAI/WCAG21/Understanding/non-text-content.html?utm_source=chatgpt.com 'Understanding Success Criterion 1.1.1: Non-text Content | WAI - W3C'))

Zusätzlich werden Ausnahmen benannt:

> “Controls, Input … Time-Based Media … Test … Sensory … CAPTCHA … Decoration, Formatting, Invisible”  
> — WCAG 2.2, SC 1.1.1 (Ausnahmen) ([W3C](https://www.w3.org/TR/WCAG22/?utm_source=chatgpt.com 'Web Content Accessibility Guidelines (WCAG) 2.2 - W3C'))

---

## W3C WAI — Guidelines / Tutorials für Alternative Text

Aus der WAI-„Images Tutorial / Alternative Text“:

> “Practical guidance for creating useful alt attribute content for images ...”  
> — W3C WAI: Images Tutorial ([W3C](https://www.w3.org/WAI/alt/?utm_source=chatgpt.com 'Resources on Alternative Text for Images - W3C'))

Aus dem Abschnitt „Informative Images“:

> “The text alternative should convey the meaning or content that is displayed visually, which typically isn’t a literal description of the image.”  
> — W3C WAI, Informative Images Tutorial ([W3C](https://www.w3.org/WAI/tutorials/images/informative/?utm_source=chatgpt.com 'Informative Images | Web Accessibility Initiative (WAI) - W3C'))

Aus dem Abschnitt „Decorative Images“:

> “In these cases, a null (empty) alt text should be provided (alt=””) so that they can be ignored by assistive technologies, such as screen readers.”  
> — W3C WAI, Decorative Images ([W3C](https://www.w3.org/WAI/tutorials/images/decorative/?utm_source=chatgpt.com 'Decorative Images | Web Accessibility Initiative (WAI) - W3C'))

Aus „Tips and Tricks“:

> “The alt text should be the most concise description possible of the image’s purpose. If anything more than a short phrase or sentence is needed, it would be better to use one of the long description methods discussed in complex images.”  
> — W3C WAI, Tips & Tricks ([W3C](https://www.w3.org/WAI/tutorials/images/tips/?utm_source=chatgpt.com 'Tips and Tricks | Web Accessibility Initiative (WAI) - W3C'))

---

Wenn du willst, kann ich dir auch die normative Sprache (aus der offiziellen WCAG-Recommendation) statt nur „Understanding“-Dokumente liefern und direkt zitierfähig machen. Möchtest du das?

Hier sind die einschlägigen Originalpassagen mit Quellen.

### WCAG 2.2 — SC 1.1.1 Non-text Content (normativ)

- „All non-text content that is presented to the user has a text alternative that serves the equivalent purpose…“  
   Quelle: [https://www.w3.org/TR/WCAG22/#non-text-content](https://www.w3.org/TR/WCAG22/#non-text-content) ([W3C](https://www.w3.org/TR/WCAG22/ 'Web Content Accessibility Guidelines (WCAG) 2.2'))
- Ausnahmen, auszugsweise:  
   „Controls, Input … has a name that describes its purpose.“ •  
   „Time-Based Media … provide descriptive identification.“ •  
   „Test … descriptive identification.“ •  
   „Sensory … descriptive identification.“ •  
   „CAPTCHA … identify and describe the purpose … alternative forms …“ •  
   „Decoration, Formatting, Invisible … implemented … ignored by assistive technology.“  
   Quelle: [https://www.w3.org/TR/WCAG22/#non-text-content](https://www.w3.org/TR/WCAG22/#non-text-content) ([W3C](https://www.w3.org/TR/WCAG22/ 'Web Content Accessibility Guidelines (WCAG) 2.2'))

### W3C WAI — Tutorials (informativ)

- **Informative Images:** „The text alternative should convey the meaning or content … typically isn’t a literal description.“  
   Quelle: [https://www.w3.org/WAI/tutorials/images/informative/](https://www.w3.org/WAI/tutorials/images/informative/) ([W3C](https://www.w3.org/WAI/tutorials/images/informative/ 'Informative Images | Web Accessibility Initiative (WAI) | W3C'))
- **Decorative Images:** „A null (empty) `alt` text should be provided (`alt=""`) so that they can be ignored by assistive technologies.“  
   Quelle: [https://www.w3.org/WAI/tutorials/images/decorative/](https://www.w3.org/WAI/tutorials/images/decorative/) ([W3C](https://www.w3.org/WAI/tutorials/images/decorative/ 'Decorative Images | Web Accessibility Initiative (WAI) | W3C'))
- **Functional Images:** „The text alternative … should convey the action that will be initiated … rather than a description of the image.“  
   Quelle: [https://www.w3.org/WAI/tutorials/images/functional/](https://www.w3.org/WAI/tutorials/images/functional/) ([W3C](https://www.w3.org/WAI/tutorials/images/functional/?utm_source=chatgpt.com 'Functional Images | Web Accessibility Initiative (WAI)'))
- **Tips & Tricks:** „Usually, there’s no need to include words like ‘image’, ‘icon’, or ‘picture’ in the alt text.“  
   Quelle: [https://www.w3.org/WAI/tutorials/images/tips/](https://www.w3.org/WAI/tutorials/images/tips/) ([W3C](https://www.w3.org/WAI/tutorials/images/tips/?utm_source=chatgpt.com 'Tips and Tricks | Web Accessibility Initiative (WAI)'))

Optional zur Einordnung:

- **Understanding SC 1.1.1:** „The intent … is to make information conveyed by non-text content accessible through … a text alternative.“  
   Quelle: [https://www.w3.org/WAI/WCAG22/Understanding/non-text-content.html](https://www.w3.org/WAI/WCAG22/Understanding/non-text-content.html) ([W3C](https://www.w3.org/WAI/WCAG22/Understanding/non-text-content.html?utm_source=chatgpt.com 'Understanding Success Criterion 1.1.1: Non-text Content'))
