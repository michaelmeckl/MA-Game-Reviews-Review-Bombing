#  Search Results

- Queries auf Twitter und Reddit sind per default immer case-insensitiv

## Twitter

- (lang:de OR lang:en) ist leider nicht immer präzise und filtert manchmal auch englischsprachige 
  Tweets raus, wenn sie wenige englische Wörter und viele Eigennamen / Zahlen / etc. enthalten  => **weglassen**
- Anführungszeichen bei Spiel engen noch weiter ein  => **weglassen** (wenn nicht zwingend nötig)
- **Wildcards** bei einzelnen Wörtern scheinen überhaupt **nicht** zu funktionieren auf Twitter (nur in ganzen Phrasen, z.B. “This game is * bad”); versch. Wortendungen müssen leider separat genannt werden
- implizit ist immer ein AND dazwischen (egal ob in Klammern oder nicht); Klammern scheinen insgesamt nicht so wichtig zu sein
  - “review (bomb OR fake)” == “(review bomb OR fake)” => “review AND (bomb OR fake)”
- Queries mit Zeitangaben (z.B. since:2023-02-06 until:2023-02-07) nur sinnvoll, wenn für das ganze Spiel gesucht wird (sonst zu wenig), aber wenn nur nach dem Spielnamen gesucht wird, gibt es teilweise über 10.000 Tweets pro Tag für bekannte Spiele (wie Hogwarts Legacy) und die meisten davon sind nicht besonders relevant
  - weitere Eingrenzung mit Keyword ‘review’ grenzt zwar deutlich ein, aber einige Ergebnisse nicht relevant
  - Query mit ‘(good OR bad OR negative OR positive)’ statt review bringt nochmal deutlich mehr Ergebnisse, sinnvoller als nur mit ‘review’ zu suchen vielleicht ? (vor allem in Kombination mit Query unten)
  - das until - date ist immer exklusiv! (until:02-07 bedeutet bis zum 02-06)
- für kleinere Spiele ist die Zeitsuche nur mit dem Spielenamen in 2-3 Wochen nach Release machbar, z.B. Ghostwire Tokyo 8449 Tweets für 1 Woche und Kunai 259 Tweets für die 3 Wochen nach Release

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
- Verschachtelte Klammerung funktioniert und richtige Klammern sind wichtig !

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

