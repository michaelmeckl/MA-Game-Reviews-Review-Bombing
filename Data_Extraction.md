# Überblick
Überblick über Code, um geeignete Tweets / Posts von Twitter und Reddit sowie Videospiele-(Nutzer)Reviews von 
Steam und Metacritic herunterzuladen.

## Steam
Die Steam Web API funktioniert gut, um grundlegende Spieleinformationen (z.B. AppList, AppReviewHistogramm) und die 
Nutzerreviews zu bekommen.

**100 % fertig**:
* [get_steam_reviews.py](./Steam/get_steam_reviews.py): enthält eine Funktion zum Laden der kompletten AppListe von 
  Steam (App/Spiel + App_Id) sowie eine Funktion zum Laden der Reviews (mit Nutzerinfos), Informationen zum Spiel & 
  Review-Histogramm-Daten für bestimmte Spiele
  * Default-Parameter - Einstellungen sind Download in 100er Batches im angegebenen Zeitraum pro Spiel, nur negative, 
    kein Limit für die Spielzeit zum Zeitpunkt des Reviews, nach Datum sortiert und off-topic-Reviews sind mit enthalten
  * Language - Filter ist per default auf "all", für Spiele mit mehr Reviews aber sinnvoll auf "english" einzugrenzen

**Verwendung :**
1. (nur einmal falls noch nicht vorhanden) Steam-AppListe laden (Games + ID): in Main-Methode Bool-Flag auf True stellen
   1. wichtig, um schnell die richtigen App-IDs für Spiele auf Steam zu finden
2. Nutzerreviews für best. Spiele laden (vgl. `get_steam_reviews.py`)
   1. Liste mit Spielen direkt im Code anpassen
   2. ggf. Start- & Enddatum anpassen / weglassen 
   3. falls für ein Spiel mehrere Reviews mit versch. Parametern / Zeiträumen extrahiert werden sollen, nach jedem 
      Durchgang die `current_progress.txt` - Datei löschen (ansonsten wird mit dem zwischengespeicherten Cursor 
      weitergemacht)

---
## Metacritic
Da keine offizielle API und keine guten, aktuellen Scraper gefunden, wurde ein eigener Scraper geschrieben.

**Anmerkung:** seit ca. 10.09.2023 funktioniert der Scraper nicht mehr, da die gesamte Metacritic-Webseite komplett 
modernisiert und re-designt wurde :(

**Update:** inzwischen existiert in [metacritic_scraper_new.py](./Metacritic/metacritic_scraper_new.py) ein neuer 
Scraper für die neue Metacritic-Seite

**100 % fertig:**
* Scrapen von Spielen (allgemeine Informationen zum Spiel + Reviews + User Informationen) für die angegebenen 
  Plattformen: entweder alle Reviews (sortiert nach Datum) oder Suche nach allen Reviews in einem angegebenen Zeitraum
* es werden alle Reviews (nicht nur schlecht bewertete) extrahiert (nach Datum sortiert)
* in [filter_metacritic_reviews.py](./Metacritic/filter_metacritic_reviews.py) können die extrahierten Reviews 
  gefiltert werden (nur englisch oder deutsch, nur Ratings 0-2 und (optional) Keyword-Search)


**Verwendung :**
1. in [metacritic_scraper_new.py](./Metacritic/metacritic_scraper_new.py) oben in Dictionary Spielname (vgl. Name in 
   Metacritic-Url) eintragen mit gewünschter Plattform
2. per default werden alle Reviews geladen, ansonsten den Zeitraum entsprechend ändern
3. (optional) danach mit [filter_metacritic_reviews.py](./Metacritic/filter_metacritic_reviews.py) die extrahierten 
   Reviews filtern
   - wurde letztendlich nicht verwendet (Filtern wurde später separat für Label Studio implementiert)

---
## Reddit
PRAW - API - Wrapper funktioniert sehr gut für das meiste, bis auf die Suche nach Kommentaren (da die Kommentarsuche 
auch noch nicht in der offiziellen Reddit API im Moment enthalten ist).
Pushshift API ist nicht mehr aktiv seit Mai 2023 und eigenes Scrapen der Result Page stellte sich als sehr 
schwierig heraus. Deshalb wurde die RedditWarp API verwendet, diese scheint für Kommentarsuche gut zu 
funktionieren (nutzt den undokumentierten GraphQL-Endpunkt von Reddit).

**99 % fertig**
* Extrahieren / Parsen von Submissions und Kommentaren in bestimmten Subreddits, bzw. "r/all" (sowie den Kommentaren zu 
  einer Submission)
* Keyword-Suche in Subreddits nach Submissions mithilfe von PRAW (bzw. Reddit Search Queries) und nach Kommentaren 
  mit RedditWarp
* Timeperiod-Search nach Submissions und Kommentaren vorhanden
* aktuell wird für Submissions grundsätzlich auch "r/all" durchsucht, für Kommentare nicht (da zu viele)
* Code, um Kommentare zu Submissions zu extrahieren ist vorhanden, aber separat und nicht standardmäßig im Ablauf 
  integriert (für schnelleren Download der Reddit Submissions und aufgrund der Rate Limits) 

**Verwendung :**
1. in [reddit_api.py](./Reddit/reddit_api.py) existierenden Code in `get_reddit_data_for_games` - Methode anpassen
   1. Game, zugehörige Subreddits & Timeperiod für Suche in Dictionary anpassen
   2. Queries für Suche nach Submissions anpassen sowohl in Subreddits als auch r/all (für r/all eignet sich 
      meistens "game_name AND query_subreddit")
   3. Query anpassen für Kommentarsuche mit RedditWarp

---
## Twitter
* Offizielle API nicht verwendbar (sehr teuer, kein Academic-Tier mehr vorhanden & starke Limits bzgl. Menge an 
Tweets und historische Tweets)
* die meisten Tools / Libraries funktionieren nach den wiederholten Änderungen an der Twitter API nicht mehr :(
  * SNScrape war das beste Tool, dass ohne Twitter Dev - Account funktionierte (geht 
    seit Juli 2023 aber auch nicht mehr). 
  * Rettiwt-API (Typescript) kann aktuell (September 2023) noch für ein paar Daten (deutlich weniger als SNScrape) 
    genutzt werden.
* Tweety ist im Moment die einzige Library, mit der es zu funktionieren scheint, aufgrund der Rate-Limits müssen zwar 
  regelmäßig Accounts gewechselt werden, aber ansonsten kann man damit die Twitter-Advanced-Search verwenden.

**95 % fertig**:
* mit Tweety funktioniert fast alles, nur Replies / Kommentare können leider nicht extrahiert werden
* **Update - April 2024**: offenbar funktioniert inzwischen mit Tweety die exakte Suche in Anführungszeichen nicht 
  mehr (obwohl es auf Twitter weiterhin zu funktionieren scheint)  -> Queries im Code funktionieren deshalb nicht mehr

**Verwendung :**
1. in [get_tweets.py](./Twitter/get_tweets.py) unten im Dictionary die Games & Queries anpassen (oder single_query auf 
   True stellen und query unten anpassen, gut geeignet zum Testen)
   1. die Zeiträume sind immer am Ende der jeweiligen Queries mit in den Queries enthalten
   2. regelmäßig Console überprüfen bzgl. Rate-Limits und ggf. Account unten in Methode wechseln (mindestens zwei 
       Twitter/X - Accounts sind nötig, sonst dauert es ewig)

---
# Aktuell extrahierte Daten
vgl. [data_for_analysis](./data_for_analysis) - Ordner (im Moment nicht in Github hochgeladen)

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
  * negative & positive User Reviews (keine min/max review time, language "all") + zugehörige User Informationen + 
    Review-Histogramm-Daten + allgemeine Infos zu den Spielen aus dem Russland-Ukraine-Review Bombing 
    (S.T.A.L.K.E.R-Serie, Witcher 1 & 2 & 3, Gwent, Thronebreaker und Frostpunk)
  * negative User Reviews (keine min/max review time, language "all") + zugehörige User Informationen + 
  Review-Histogramm-Daten + allgemeine Infos zu den Spielen:
    * The Elder Scrolls V: Skyrim & Skyrim Special Edition
    * Fallout 4
    * Grand Theft Auto V (hier bei language nur "english, german" statt "all", da sonst zu viele)
    * Total War: ROME II - Emperor Edition
    * Mortal Kombat 11
    * Assassin's Creed Unity (hier stattdessen nur die positiven User Reviews, da positives Review Bombing)
    * Crusader Kings II
    * The Long Dark
    * Superhot VR
    * Hogwarts Legacy (nur vom 11. - 21.02, auch wenn Release schon am 07.02 war, da sonst zu viele Reviews in den ersten Tagen)
    * No Man's Sky (nur vom 12. - 19.08, auch wenn Release schon am 09.08 war, da sonst zu viele Reviews in den ersten Tagen)

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
    * "The Elder Scrolls V: Skyrim" aus dem Zeitraum 23.04.2015 - 31.07.2015
    * "Grand Theft Auto V" aus dem Zeitraum 14.06.2017 - 01.10.2017
    * "The Elder Scrolls V: Skyrim Special Edition" und "Fallout 4" aus dem Zeitraum 29.08.2017 - 01.01.2018
    * "Total War: Rome II" aus dem Zeitraum 20.09.2018 - 31.12.2018
    * "Mortal Kombat 11" aus dem Zeitraum 22.04.2019 - 01.09.2019
    * "Assassin's Creed Unity" aus dem Zeitraum 16.04.2019 - 30.11.2019
    * "Crusader Kings II" aus dem Zeitraum 01.10.2019 - 31.12.2019
    * "The Long Dark" aus dem Zeitraum 01.03.2020 - 31.12.2020
    * "Superhot VR" aus dem gesamten Zeitraum
    * "Hogwarts Legacy" aus dem Zeitraum 07.02.2023 - 01.03.2023
    * "No Man's Sky" aus dem Zeitraum 09.08.2016 - 01.09.2016
  
* **Reddit:**
  * Submissions & Kommentare mit zugehörigen User Informationen für die Default - Queries (s. Code) zu den 
    Spielen *Cyberpunk 2077, Metro Exodus, Borderlands 3, Firewatch, Overwatch 2, The Elder Scrolls V: Skyrim, 
    Grand Theft Auto V, Total War: Rome II, Mortal Kombat 11 und Assassin's Creed Unity* sowie für das Bethesda 
    Creation Club - Review Bombing (Skyrim & Fallout 4) und das Russland-Ukraine-Review Bombing
  * Submissions & Kommentare mit zugehörigen User Informationen für spezifische Queries für die Spiele *Cyberpunk 2077, 
    Metro Exodus, Borderlands 3, Firewatch und Overwatch 2, The Elder Scrolls V: Skyrim, Grand Theft Auto V, Total 
    War: Rome II, Mortal Kombat 11 und Assassin's Creed Unity* sowie für das Bethesda Creation Club - Review Bombing 
    (Skyrim & Fallout 4) sowie für das Russland-Ukraine-Review Bombing (für Queries & Zeiträume, s. Code; manchmal 
    auch Ergebnisse mehrerer Queries kombiniert)
  
* **Twitter:**
  * Tweets für die allg. Query `[GAME] review (bomb OR bombs OR bombing OR boycott OR boycotting OR controvers OR 
    controversy OR manipulate OR manipulation OR fake OR sabotage OR sabotaging OR spam OR hate)` zu den Spielen 
    *Cyberpunk 2077, Metro Exodus, Borderlands (3), Firewatch und Overwatch 2, The Elder Scrolls V: Skyrim, 
    Grand Theft Auto V, Total War: Rome II, Mortal Kombat 11 und Assassin's Creed Unity* sowie für das Bethesda 
    Creation Club - Review Bombing (Skyrim & Fallout 4) sowie für das Russland-Ukraine-Review Bombing (und die davon 
    betroffenen Spiele wie die Witcher- oder S.T.A.L.K.E.R-Serie)
  * Tweets zu spezifischen Queries für die Spiele *Cyberpunk 2077, Metro Exodus, Borderlands 3, Firewatch und 
    Overwatch 2, The Elder Scrolls V: Skyrim, Grand Theft Auto V, Total War: Rome II, Mortal Kombat 11 und 
    Assassin's Creed Unity* sowie für das Bethesda Creation Club - Review Bombing (Skyrim & Fallout 4) sowie für 
    das Russland-Ukraine-Review Bombing (für Queries & Zeiträume, s. Code)
