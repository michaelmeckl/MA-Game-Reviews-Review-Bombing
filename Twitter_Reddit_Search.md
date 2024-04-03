#  Search Results

- Queries auf Twitter und Reddit sind per default immer case-insensitiv

## Twitter

- (lang:de OR lang:en) ist leider nicht immer präzise und filtert manchmal auch englischsprachige 
  Tweets raus, wenn sie wenige englische Wörter und viele Eigennamen / Zahlen / etc. enthalten  => **weglassen**
- Anführungszeichen bei Spiel engen noch weiter ein  => **weglassen** (wenn nicht zwingend nötig)
- **Wildcards** bei einzelnen Wörtern scheinen überhaupt **nicht** zu funktionieren auf Twitter (nur in ganzen Phrasen, z.B. “This game is * bad”); versch. Wortendungen müssen leider separat genannt werden
- implizit ist immer ein AND dazwischen (egal ob in Klammern oder nicht); Klammern scheinen insgesamt nicht so wichtig zu sein
  - “review (bomb OR fake)” == “(review bomb OR fake)” == “review AND (bomb OR fake)”
- Queries mit Zeitangaben (z.B. since:2023-02-06 until:2023-02-07) nur sinnvoll, wenn für das ganze Spiel gesucht wird (sonst zu wenig), aber wenn nur nach dem Spielnamen gesucht wird, gibt es teilweise über 10.000 Tweets pro Tag für bekannte Spiele (wie Hogwarts Legacy) und die meisten davon sind nicht besonders relevant
  - weitere Eingrenzung mit Keyword ‘review’ grenzt zwar deutlich ein, aber einige Ergebnisse nicht relevant
  - Query mit ‘(good OR bad OR negative OR positive)’ statt review bringt nochmal deutlich mehr Ergebnisse, sinnvoller als nur mit ‘review’ zu suchen vielleicht ? (vor allem in Kombination mit Query unten)
  - das until - date ist immer exklusiv! (until:02-07 bedeutet bis zum 02-06)
- für kleinere Spiele ist die Zeitsuche nur mit dem Spielenamen in 2-3 Wochen nach Release schon machbar, z.B. Ghostwire Tokyo 8449 Tweets für 1 Woche und Kunai 259 Tweets für die 3 Wochen nach Release

| Query                                                        | Ergebnisse - Cyberpunk 2077 | Ergebnisse - Hogwarts Legacy |
| ------------------------------------------------------------ | --------------------------- | ---------------------------- |
| [GAME] (lang:de OR lang:en) review bomb                      | 113                         | 182                          |
| [GAME] (review bomb OR "boycott* " OR "controvers* " OR manipulat* " OR fake review) | 55                          | 183                          |
| [GAME] **(lang:de OR lang:en)** (review bomb OR "boycott* " OR "controvers* " OR manipulat* " OR fake review) | 49                          | 176                          |
| **“[GAME]”** (review bomb OR "boycott* " OR "controvers* " OR manipulat* " OR fake review) | 46                          | 133                          |
| [GAME] (lang:de OR lang:en) (**"review bomb* "** OR "boycott* " OR "controvers* " OR manipulat* " OR **"fake* "**) | 0                           | 0                            |
| [GAME] (lang:de OR lang:en) (review bomb OR "boycott* " OR "controvers* " OR manipulat* " OR "fake* " **OR "sabotage* " OR review spam**) | 24                          | 69                           |
| [GAME] **review (bomb** OR boycott OR controvers OR manipulate OR fake OR sabotage OR spam) | 59                          | 187                          |
| [GAME] **(review bomb** OR boycott OR controvers OR manipulate OR fake OR sabotage OR spam) | 59                          | 187                          |
| [GAME] (review bomb OR "boycott* " OR "controvers* " OR manipulat* " **OR hate** OR fake) | 141                         | 287                          |
| [GAME] review (bomb **OR bombs OR bombing** OR boycott OR boycotting OR controvers **OR controversy** OR manipulate **OR manipulation** OR fake OR sabotage OR sabotaging OR spam OR hate) | 312                         | 523                          |

=> **beste Query:**  ```[GAME] review (bomb OR bombs OR bombing OR boycott OR boycotting OR controvers OR controversy OR manipulate OR manipulation OR fake OR sabotage OR sabotaging OR spam OR hate)```

-  "hate" bringt zwar nochmal einige Ergebnisse mehr, aber sehr viel davon auch nicht relevant

**Insgesamt:** Query oberhalb für alle verwenden; für kleinere / unbekanntere Spiele Suche nur mit Spielname + Zeitraum möglich, für bekanntere Spiele wenn dann nur mit Zusatz im Zeitraum suchen ( “(good OR bad OR positive OR negative)” ist z.B. ganz guter Kompromiss)

## Reddit

- Suche in bestimmten Feldern möglich: selftext:review OR title:review
- Suche in bestimmter Zeitspanne geht in Reddit nicht (zumindest im Browser und in offizieller API); muss manuell implementiert werden beim Filtern der Ergebnisse
- U.a. folgende Queries in Cyberpunk Subreddits getestet:
  - `'ReviewBomb OR review-bomb OR "review bomb"'`, `'"ReviewBomb*" OR "review-bomb*" OR "review bomb*"'`, `'"review bomb"'` => Wildcard & versch. Schreibweisen machen keinen Unterschied
  - auch mit Hogwarts Legacy in r/all und in Subreddit getestet für Queries ` “review bomb” / review bomb / “review * bomb * ” / “review bomb*” / review bombing / Hogwarts Legacy AND "review bomb" / Hogwarts Legacy AND "review bomb*` -> ebenfalls keine Auswirkungen der Wildcard
  - **=> Wildcards werden ignoriert**, aber es werden grundsätzlich verschiedene Varianten / Endungen gefunden
    - "review bomb" findet auch Review Bombing und review-bomb, aber nicht ReviewBomb
    - ReviewBomb hingegen findet alle (auch review-bomb or review bombing)
- Kein Unterschied zwischen 'review bombing' und 'review bomb' in subreddit sowie in r/all
- controversy oder controversial liefern die gleichen Ergebnisse, controvers (nur Wortstamm) liefert aber 0
- “fake review” und “review fake” liefern die selben Ergebnisse => Reihenfolge nicht relevant
  - allerdings bringt “review AND fake” 2 Ergebnisse mehr als “review fake”  => **explizites AND sinnvoll** (?)
- Verschachtelte Klammerung funktioniert und richtige Klammernsetzung ist wichtig !

| Query                                                        | Subreddit                                            |              Ergebnisse - Cyberpunk 2077               | Ergebnisse - Hogwarts Legacy |
| ------------------------------------------------------------ | ---------------------------------------------------- | :----------------------------------------------------: | :--------------------------: |
| review bomb                                                  | HarryPotterGame                                      |                           /                            |              4               |
| “review bomb*****“                                           | HarryPotterGame                                      |                           /                            |              4               |
| “review bomb”                                                | HarryPotterGame                                      |                           /                            |              4               |
| Hogwarts Legacy AND review bomb                              | all                                                  |                           /                            |              15              |
| Hogwarts Legacy AND "review bomb"                            | all                                                  |                           /                            |              12              |
| Hogwarts Legacy AND "review bombing"                         | all                                                  |                           /                            |              12              |
| “ReviewBomb**\*"** OR "review-bomb\*" OR "review bomb**\*"** | Cyberpunk - Subreddits                               |                           38                           |              /               |
| ReviewBomb OR review-bomb OR "review bomb"                   | Cyberpunk - Subreddits                               |                           46                           |              /               |
| ReviewBomb* OR review-bomb* OR \"review bomb*"               | Cyberpunk - Subreddits                               |                           46                           |              /               |
| "review bomb"                                                | Cyberpunk - Subreddits                               |                           37                           |              /               |
| ReviewBomb                                                   | Cyberpunk - Subreddits                               | **46 (gleiche Ergebnisse wie bei den anderen beiden)** |              /               |
| review (bomb OR bombing OR bombs)                            | Cyberpunk - Subreddits                               |    47 (findet aber z.B. auch review und time bomb)     |              /               |
| controversy                                                  | Cyberpunk - Subreddits                               |                          228                           |              /               |
| ReviewBomb OR review AND (fake OR spam OR controversy OR negative OR hate OR boycott) | HogwartsLegacy - Subreddits                          |                           /                            |              51              |
| ReviewBomb OR **(**review AND (fake OR spam OR controversy OR negative OR hate OR boycott)**)** | HogwartsLegacy - Subreddits                          |                           /                            |              54              |
| ReviewBomb OR fake **OR spam** OR controversy **OR negative** OR hate OR boycott | HogwartsLegacy - Subreddits / Cyberpunk - Subreddits |                          240                           |             240              |
| ReviewBomb OR controversy OR hate OR boycott OR fake         | HogwartsLegacy - Subreddits / Cyberpunk - Subreddits |                          236                           |             242              |
| ReviewBomb OR boycott OR controversy OR fake OR **sabotage OR manipulate OR spam** | HogwartsLegacy - Subreddits / Cyberpunk - Subreddits |                          242                           |             236              |
| ReviewBomb OR boycott OR controversy OR fake OR sabotage OR manipulate OR spam **OR hate** | HogwartsLegacy - Subreddits / Cyberpunk - Subreddits |                          241                           |             240              |
| review AND (good OR bad OR negative OR positive)             | HogwartsLegacy - Subreddits / Cyberpunk - Subreddits |                          233                           |             148              |
| **Hogwarts Legacy AND** review AND (good OR bad OR negative OR positive) | all                                                  |                           /                            |             209              |
| **[GAME] AND** (ReviewBomb OR controversy OR hate OR boycott OR fake) | all                                                  |                          142                           |             208              |
| **[GAME] AND** (ReviewBomb OR boycott OR controversy OR fake OR sabotage OR manipulate OR spam OR hate) | all                                                  |                          166                           |             228              |

- <u>HogwartsLegacy - Subreddits</u>: “HarryPotterGame" + "hogwartslegacyJKR" + "HogwartsLegacyGaming”
- <u>Cyberpunk - Subreddits</u>: "cyberpunkgame" + "CyberpunkTheGame" + "LowSodiumCyberpunk"

=> **beste Query:**  `ReviewBomb` ist beste Variante für Review Bombing; für Suche ist Query `ReviewBomb OR boycott OR controversy OR fake OR sabotage OR manipulate OR spam OR hate`, bzw. in r/all `[GAME] AND ...` am besten

=> **für Kommentare** andere Query nötig: “ReviewBomb OR boycott OR controversy OR fake OR sabotage OR manipulate OR spam OR hate" liefert überall 0 Ergebnisse; **Queries**: `ReviewBomb OR "review bombing"` ganz gut (hier gibt es ein paar wenige Unterschiede zu nur ‘ReviewBomb’ (z.B. werden ein paar auseinandergeschriebene bei der 1. zusätzlich erkannt), andere Variationen davon scheinen aber nichts zu bringen); `review AND (good OR bad OR negative OR positive OR hate)` bringt auch einige (andere) gute Ergebnisse

=> Suche in Zeitraum oft gar nicht nötig (außer bei Kommentaren), da insgesamt nicht so viele Ergebnisse



# Spezifische Queries

Queries für Reddit sollten etwas allgemeiner als bei Twitter formuliert werden, da Reddit sowieso deutlich weniger Ergebnisse liefert (v.a. bei Kommentar-Suche muss die Query möglichst breit sein)

### CDPR & andere + Russia - Ukraine - Conflict

- Zeitraum Ende Februar 2022  -  Ende April 2022; betroffen u.a. waren CDPR - Spiele, Frostpunk, S.T.A.L.K.E.R, This War of Mine und einige andere Spiele

<u>Twitter:</u>

- (ReviewBomb OR “review bombing” OR "negative game review") AND (russia OR ukraine)

- (CD Projekt Red OR CDPR OR Witcher OR Gwent OR “Cyberpunk 2077” OR Frostpunk OR “STALKER game”) AND (russia OR ukraine)

- ("CD Projekt Red" OR "CD Project Red" OR CDPR OR Witcher OR Gwent OR Thronebreaker OR "Cyberpunk 2077" OR Frostpunk OR ("STALKER" OR "S.T.A.L.K.E.R")) AND (russia OR ukraine)

- ("CD Projekt Red" OR "CD Project Red" OR CDPR OR Witcher OR Gwent OR Thronebreaker OR "Cyberpunk 2077" OR Frostpunk OR ("STALKER" OR "S.T.A.L.K.E.R")) AND (sales OR support) AND (russia OR ukraine)

  => alle mit  “since:2022-02-24 until:2022-04-01”

  => die letzten beiden Queries sind am besten

<u>Reddit:</u>

- r/all: (ukraine OR russia) AND (ReviewBomb OR "review bombing" OR game review)

- In Subreddits:

  - (ReviewBomb OR "review bombing") AND (ukraine OR russia)       => in cyberpunk subreddit - Kommentare
  - sales (ukraine OR russia)   => in r/CDProjektRed und r/gwent
  - review (ukraine OR russia)  => in Witcher Subreddits

  => überall aber nur sehr wenige Ergebnisse (am meisten bei Cyberpunk)

  => finale Query: ```(review OR support OR sales) AND (ukraine OR russia)```

### Borderlands & Metro - Serie

- Zeitraum für Metro war Ende Januar - Februar 2019 und Februar 2020 (für Metro Exodus), bei Borderlands April 2019 und März 2020 (für Borderlands 3)

<u>Twitter:</u>

- borderlands 3 AND ("epic store" OR "epic games") since:2019-04-01 until:2019-05-01

  - borderlands 3 AND ("epic store" OR "epic games" OR exclusive)

- (metro OR “metro exodus”) AND ("epic store" OR "epic games") since:2019-01-27 until:2019-03-01

- (metro OR “metro exodus”) AND ("epic store" OR "epic games") since:2020-02-01 until:2020-03-01

  => Zeitraum eingrenzen ist sehr wichtig auf Twitter, da sonst viel zu viele Tweets (hier in einem Monat schon über 4000 Tweets)

<u>Reddit:</u>

- r/all: 

  - (metro OR “metro exodus”) ReviewBomb AND ("epic store" OR "epic games")
  - borderlands 3 ReviewBomb AND ("epic store" OR "epic games")

  => am besten einfach nur ```[GAME] AND ("epic store" OR "epic games" OR exclusive)```

- subreddits: 

  - Queries wie ```(ReviewBomb OR review OR "negative review" OR hate) AND ("epic store" OR "epic games")``` schränken zu sehr ein, stattdessen einfach nur ```"epic store" OR "epic games" OR exclusive ```
  - für Kommentare ist einfache Query wie ```epic games store``` am besten, bei Borderlands aber 0 Ergebnisse im Zeitraum, deswegen kein End-Datum etwas weiter nach hinten setzen

### Firewatch

- Zeitraum September 2017

<u>Twitter:</u>

- Firewatch (DCMA OR pewdiepie) since:2017-09-10 until:2017-11-01

<u>Reddit:</u>

- r/all: Firewatch AND (DCMA OR takedown OR pewdiepie)
- r/Firewatch: fast keine Ergebnisse zu Review Bombing, DCMA oder PewDiePie (oder nur die gleichen wie in all)
  - überhaupt keine Kommentar-Ergebnisse im Subreddit

### Overwatch 2

- Zeitraum im Prinzip seit Early Access im Oktober 2022, aber vor allem seit Steam - Release im August 2023

<u>Twitter</u>: 

- "Overwatch 2" (promise OR shutdown OR greed OR monetization OR microtranscation)

- "Overwatch 2" (review (bomb OR bombs OR bombing))

  => Zeiträume “since:2023-08-10 until:2023-09-01” und “since:2022-10-04 until:2022-11-01”

<u>Reddit:</u> 

- r/all: "Overwatch 2" AND (promise OR shutdown OR greed OR monetization OR microtranscation)
- subreddits: promise OR shutdown OR greed OR monetization OR microtranscation
  - comments: (negative OR hate) AND (promise OR shutdown OR greed OR monetization OR microtranscation)

### Cyberpunk 2077

- außer dem Review Bombing im März 2022 (s. oben), noch im Zeitraum Dezember 2020 und Januar 2023
- Für 2020 Keywords wie “Lie” / “Fraud” / “Scam” / “Broken” / “Disappoint”; für 2023 “Labor of Love”  & (Steam) Award

<u>Twitter</u>: 

- "Cyberpunk 2077" (lie OR fraud OR scam OR broken OR disappoint OR disappointment OR disappointing) until:2021-01-01 since:2020-12-09
- "Cyberpunk 2077" (negative OR hate OR review) (lie OR fraud OR scam OR broken OR disappoint OR disappointment OR disappointing)
  - deutlich weniger als bei der ersten Query aber noch ein paar relevante neue
- "Cyberpunk 2077" (award OR “Steam Awards” OR “Labor of Love”) until:2023-02-01 since:2023-01-01

<u>Reddit:</u> 

- r/all: "Cyberpunk 2077" AND (lie OR fraud OR scam OR broken OR disappoint OR disappointment OR disappointing)
- subreddits: lie OR fraud OR scam OR broken OR disappoint OR disappointment OR disappointing
  - im Zeitraum extrem wenige Kommentare dazu (mit weniger Keywords offenbar ein bisschen besser)
- für Steam - “Labor of Love” - RB im Zeitraum Januar 2023:
  - r/all: "Cyberpunk 2077" AND (award OR "Steam Awards" OR "Labor of Love")
  - subreddits: award OR "Steam Awards" OR "Labor of Love"
  - für comments ist die Query ‘steam OR award OR "Labor of Love"’ besser 