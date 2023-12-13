# Überblick
Verschiedene Tests, um geeignete Tweets / Posts von Twitter und Reddit sowie Videospiele-(Nutzer)Reviews von 
Steam und Metacritic zu erhalten.

## Steam
Steam Web API funktioniert gut, um an grundlegende Spieleinformationen (z.B. AppList, AppReviewHistogramm) und die 
Nutzerreviews heranzukommen.

**100 % fertig**:
* [get_steam_reviews.py](./Steam/get_steam_reviews.py): enthält eine Funktion zum Laden der kompletten AppListe von 
  Steam (App/Spiel + App_Id) sowie eine Funktion zum Laden der Reviews & Review Histogramm-Daten für bestimmte Spiele
  * Default-Parameter - Einstellungen sind max. 1000 Reviews pro Spiel (in 100er Batches), nur negative, maximal 1h 
    Spielzeit zum Zeitpunkt des Reviews, auch off-topic-Reviews und nach Datum sortiert

zusätzlich (veraltet, inzwischen befindet sich eine neuere Methode im anderen Code oben):
* [scrape_steam_games.py](./Steam/_scrape_steam_games.py) führt den Code aus dem heruntergeladenen 
  [Steam-Games-Scraper - Repository](https://github.com/FronkonGames/Steam-Games-Scraper) aus, um einige Daten zu Steam - Spielen von Steam & SteamSpy zu laden; enthält außerdem eine 
  Methode, um die resultierende `games.json` zu parsen.

**Ablauf :**
1. (nur einmal, falls noch nicht vorhanden) Steam-AppListe laden (Games + ID)
2. Nutzerreviews für best. Spiele laden (vgl. `get_steam_reviews.py`)
   1. (im Moment) Liste mit Spielen direkt im Code anpassen
   2. ggf. Start- & Enddatum unterhalb anpassen / weglassen
   3. falls mehrere Reviews mit versch. Parametern / Zeiträumen extrahiert werden sollen, nach jedem Durchgang die 
      `current_progress.txt` - Datei löschen

---
## Metacritic
Da keine offizielle API und keine guten, aktuellen Scraper gefunden, eigenen Scraper geschrieben.

**Anmerkung:** seit ca. 10.09.2023 funktioniert der Scraper nicht mehr, da die gesamte Metacritic-Webseite komplett 
modernisiert und re-designt wurde :(

**Update:** inzwischen existiert in [metacritic_scraper_new.py](./Metacritic/metacritic_scraper_new.py) ein neuer 
Scraper für die neue Metacritic-Seite

**100 % fertig:**
* Scrapen von Spielen (allgemeine Informationen zum Spiel + Reviews + User Informationen) für die angegebenen 
  Plattformen: entweder alle Reviews (sortiert nach Datum) oder Suche nach allen Reviews in einem angegebenen Zeitraum
* es werden alle Reviews (nicht nur schlecht bewertete) nach Datum sortiert extrahiert
* in [filter_metacritic_reviews.py](./Metacritic/filter_metacritic_reviews.py) werden die extrahierten Reviews 
  gefiltert (nur englisch oder deutsch, nur Ratings 0-2 und (optional) Keyword-Search)
* **BUG**: Time Period Search funktioniert oft nicht richtig, da manchmal die Filterfunktion auf Metacritic die Reviews 
  nicht immer in der richtigen Reihenfolge anzeigt :(

**Ablauf :**
1. in [metacritic_scraper_new.py](./Metacritic/metacritic_scraper_new.py) oben in Dictionary Spielname (vgl. Name in 
   Metacritic-Url) eintragen mit Plattform
2. alle Reviews oder nur in Zeitraum - Flag entsprechend ändern (+ ggf. Zeitraum)
3. danach mit [filter_metacritic_reviews.py](./Metacritic/filter_metacritic_reviews.py) die extrahierten Reviews filtern

---
## Reddit
PRAW - API - Wrapper funktioniert sehr gut für das meiste, bis auf die Suche nach Kommentaren (da die Kommentarsuche 
auch noch nicht in der offiziellen Reddit API im Moment enthalten ist).
Pushshift API ist nicht mehr aktiv seit Mai 2023 und eigenes Scrapen der Result Page sehr problematisch.
RedditWarp API scheint aber für Kommentarsuche gut zu funktionieren (nutzt den undokumentierten GraphQL-Endpunkt).

**92 % fertig** (Kommentare von submissions werden aktuell noch ignoriert!)
* Extrahieren / Parsen von Submissions und Kommentaren in bestimmten Subreddits, bzw. "r/all" (sowie den Kommentaren zu 
  einer Submission)
* Keyword-Suche in Subreddits nach Submissions mithilfe von PRAW (bzw. Reddit Search Queries) und nach Kommentaren 
  mit RedditWarp
* Timeperiod-Search nach Submissions und Kommentaren vorhanden
* aktuell wird für Submissions grundsätzlich auch "r/all" durchsucht, für Kommentare aber nicht (da vermutlich zu viele)

**Ablauf :**
1. in [reddit_api.py](./Reddit/reddit_api.py) existierenden Code in `get_reddit_data` - Methode anpassen
   1. Game, zugehörige Subreddits & Timeperiod anpassen
   2. Queries für Suche nach Submissions anpassen
   3. Query anpassen für Kommentarsuche mit RedditWarp

---
## Twitter
Offizielle API nicht verwendbar (sehr teuer, kein akademischer Tier mehr vorhanden & starke Limits bzgl. Menge an 
Tweets und historische Tweets); die meisten Tools / Libraries funktionieren nach den wiederholten Änderungen an der 
Twitter API nicht mehr. SNScrape war das beste Tool, dass ohne Twitter Dev - Account funktionierte (geht seit Juli aber auch nicht 
mehr).
Rettiwt-API (Typescript) kann als aktuell noch für ein paar Daten (deutlich weniger als SNScrape) genutzt werden.
Tweety ist im Moment die einzige Library, mit der es zu funktionieren scheint, aufgrund der Rate-Limits müssen zwar 
regelmäßig Accounts gewechselt werden, aber ansonsten kann man damit die Twitter-Advanced-Search verwenden.

**99 % fertig**:
* mit Tweety funktioniert fast alles, nur Replies / Kommentare können leider nicht extrahiert werden

**Ablauf :**
1. in [get_tweets.py](./Twitter/get_tweets.py) unten die Games & Queries anpassen, oder single_query auf True 
   stellen und query unten anpassen (die Konfiguration ganz oben 
   wird im Moment nicht benutzt)
    1. regelmäßig Console überprüfen bzgl. Rate-Limits und ggf. Account unten in Methode wechseln

---
# Aktuell verfügbare Daten
vgl. [data_for_analysis](./data_for_analysis) - Ordner

* **Steam:**
  * ca. 250 negative Reviews zu Hogwarts Legacy (7.02.2023 - 21.02.2023) (noch ohne User Infos)
  * negative User Reviews (keine min/max review time und bei den meisten bei Sprachen "all", bei manchen auf "english, 
    german" eingeschränkt) mit zugehörigen User Informationen + Review-Histogramm-Daten + allgemeine Infos zu den 
    Spielen: (für Zeiträume s. Dict im Code)
    * Metro 2033 Redux & Metro: Last Light Redux & Metro Exodus
    * Borderlands GOTY & Borderlands GOTY Enhanced & Borderlands The Pre-Sequel & Borderlands 2 & Borderlands 3
    * Firewatch
    * Overwatch 2
    * Cyberpunk 2077
      * 09.12.2020 - 16.12.2020 (46820 Reviews)  &  01.03.2022 - 13.03.2022 (4977 negative Reviews)
      * 01.01.2023 - 31.01.2023
      * 17.12.2020 - 02.01.2021  &  8.06.2023 - 19.06.2023 (bei diesen nur eingeschränkte review time bis 1h)
  * negative User Reviews (keine min/max review time, language "all") + zugehörige User Informationen + 
  Review-Histogramm-Daten + allgemeine Infos zu den Spielen:
    * The Elder Scrolls V: Skyrim & Skyrim Special Edition
    * Fallout 4
    * Grand Theft Auto V (hier bei language nur "english, german" statt "all", da sonst zu viele)
    * Total War: ROME II - Emperor Edition
    * Mortal Kombat 11
    * Assassin's Creed Unity (hier stattdessen nur die positiven User Reviews, da positives Review Bombing)
  * negative & positive User Reviews (keine min/max review time, language "all") + zugehörige User Informationen + 
    Review-Histogramm-Daten + allgemeine Infos zu den Spielen aus dem Russland-Ukraine-Review Bombing 
    (S.T.A.L.K.E.R-Serie, Witcher 1 & 2 & 3, Gwent, Thronebreaker und Frostpunk)
  
* **Metacritic:**
  * allgemeine Informationen + 407 UserReviews zu Hogwarts Legacy im Zeitraum 7.02.2023 - 21.02.2023
  * allgemeine Informationen + alle PC - UserReviews zu den Spielen "Metro 2033, Metro: Last Light, Metro 2033 Redux, 
    Metro: Last Light Redux, Metro Exodus, Borderlands, Borderlands The Pre-Sequel, Borderlands 2, Borderlands 3, 
    Firewatch und Overwatch 2" + zugehörige User Informationen
    * die gleichen Informationen wie oben zu Cyberpunk 2077 aus den Zeiträumen 10.12.2020 - 02.01.2021,
      01.03.2022 - 31.03.2022 und 01.01.2023 - 31.01.2023
  * allgemeine Informationen + alle PC - UserReviews + zugehörige User Informationen zu den Spielen aus dem 
    Russland-Ukraine-Review Bombing aus dem Zeitraum 24.02.2022 - 31.03.2022 (S.T.A.L.K.E.R-Serie, Witcher 1 
    (& 1 Enhanced Edition) & 2 & 3 (& 3 Complete Edition), Gwent, Thronebreaker und Frostpunk)
  * allgemeine Informationen + alle PC - UserReviews + zugehörige User Informationen zu 
    * "The Elder Scrolls V: Skyrim" aus dem Zeitraum 23.04.2015 - 01.05.2015
    * "Grand Theft Auto V" aus dem Zeitraum 14.06.2017 - 31.07.2017
    * "The Elder Scrolls V: Skyrim Special Edition" und "Fallout 4" aus dem Zeitraum 29.08.2017 - 01.12.2017
    * "Total War: Rome II" aus dem Zeitraum 20.09.2018 - 31.10.2018
    * "Mortal Kombat 11" aus dem Zeitraum 22.04.2019 - 01.05.2019
    * "Assassin's Creed Unity" aus dem Zeitraum 19.04.2019 - 31.05.2019
  
* **Reddit:**
  * Submissions & Kommentare mit zugehörigen User Informationen für die Default - Queries (s. Code) zu den 
    Spielen "Cyberpunk 2077, Metro Exodus, Borderlands 3, Firewatch und Overwatch 2" sowie für das Russland-Ukraine-Review Bombing
  * Submissions & Kommentare mit zugehörigen User Informationen für spezifische Queries für die Spiele "Cyberpunk 2077, 
    Metro Exodus, Borderlands 3, Firewatch und Overwatch 2" sowie für das Russland-Ukraine-Review Bombing (für 
    Queries & Zeiträume, s. Code; manchmal auch Ergebnisse mehrerer Queries kombiniert)
  
* **Twitter:**
  * Tweets für die allg. Query `[GAME] review (bomb OR bombs OR bombing OR boycott OR boycotting OR controvers OR 
    controversy OR manipulate OR manipulation OR fake OR sabotage OR sabotaging OR spam OR hate)` zu den Spielen 
    "Cyberpunk 2077, Metro Exodus, Borderlands (3), Firewatch und Overwatch 2" sowie für das Russland-Ukraine-Review 
    Bombing (und die davon betroffenen Spiele wie die Witcher- oder S.T.A.L.K.E.R-Serie)
  * Tweets zu spezifischen Queries für die Spiele "Cyberpunk 2077, Metro Exodus, Borderlands 3, Firewatch und 
    Overwatch 2" sowie für das Russland-Ukraine-Review Bombing (für Queries & Zeiträume, s. Code)

* **Old Data:**
  * **Reddit:**
    * 49 Submissions (ohne Kommentare) zur Query `("ReviewBomb*" OR "review-bomb*" OR "review bomb*")` für Cyberpunk 2077 
      (aus Subreddits und r/all) aus dem Zeitraum 10.12.2020 - 27.06.2023
    * 8 Submissions (ohne Kommentare) zur Query `("ReviewBomb*" OR "review-bomb*" OR "review bomb*")` für Hogwarts 
      Legacy (aus Subreddits und r/all) aus dem Zeitraum 06.02.2023 - 27.06.2023
    * Submissions und Kommentare zu den Spielen Hogwarts Legacy, Cyberpunk 2077, Elden Ring, Ghostwire Tokyo, The Last 
      of Us Part II, Borderlands Series und Titan Souls für die Queries `ReviewBomb OR boycott OR controversy 
      OR fake OR sabotage OR manipulate OR spam OR hate` (Submissions) sowie `ReviewBomb OR "review bombing"`(Kommentare)
  * **Twitter:**
    * Hogwarts Legacy:
      * ca 3000-4000 Tweets zu verschiedenen Queries aus Februar 2022 + ca. 8000 Tweets für 06.02
    * Cyberpunk 2077:
      * ca 2000 Tweets von bestimmten Tagen im Dezember 2020 und März 2022
    * Tweets für die Query `[GAME] review (bomb OR bombs OR bombing OR boycott OR boycotting OR controvers OR controversy OR manipulate OR manipulation OR fake OR sabotage OR sabotaging OR spam OR hate)` zu den Spielen Hogwarts Legacy, Cyberpunk 2077, Elden Ring, Ghostwire Tokyo, The 
      Last of Us Part II, Borderlands Series, Titan Souls und Kunai
    * 8449 Tweets zu Ghostwire Tokyo für die Query `Ghostwire Tokyo (lang:en OR lang:de) since:2023-04-11 
      until:2023-04-19`
